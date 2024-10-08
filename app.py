# app.py
from datetime import datetime
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

def display_beautified_dataframe(df, no_index=True):
    # Copy the DataFrame to avoid modifying the original
    df_display = df.copy()
    hide_index = no_index

    if hide_index:
        # remove id column from display
        if 'id' in df_display.columns:
            df_display = df_display.drop(columns=['id'])
        
        for col in df_display.columns:
            if 'id' in col:
                df_display = df_display.drop(columns=[col])
    
    # Beautify column names: Replace underscores with spaces and convert to Title Case
    df_display.columns = df_display.columns.str.replace('_', ' ').str.title()
    
    # Define column configurations
    column_configs = {}
    for col in df_display.columns:
        if pd.api.types.is_numeric_dtype(df_display[col]):
            # For numeric columns, apply number formatting with thousand separators
            if pd.api.types.is_float_dtype(df_display[col]):
                number_format = "%,.2f"  # Format for floats
            else:
                number_format = "%,d"    # Format for integers
            column_configs[col] = st.column_config.NumberColumn(
                col
            )
        elif pd.api.types.is_datetime64_any_dtype(df_display[col]):
            # For datetime columns, format as desired
            column_configs[col] = st.column_config.DatetimeColumn(
                col,
                format="YYYY-MM-DD HH:mm:ss",
                min_width=150,
            )
        else:
            # For text columns, set minimum width
            column_configs[col] = st.column_config.TextColumn(
                col
            )
    
    # Display the DataFrame with the column configurations
    st.dataframe(
        data=df_display,
        column_config=column_configs,
        use_container_width=True,
        hide_index=hide_index
    )


@st.cache_resource
def get_similarity_matrix(customers, products):
    # Encode categorical variables
    le = LabelEncoder()
    customers['risk_profile_encoded'] = le.fit_transform(customers['risk_profile'])
    products['risk_level_encoded'] = le.transform(products['risk_level'])
    products['min_investment_normalized'] = products['min_investment'] / products['min_investment'].max()

    # Combine risk levels for consistent encoding
    risk_levels = pd.concat([customers['risk_profile'], products['risk_level']], axis=0)
    le = LabelEncoder()
    le.fit(risk_levels)

    # Encode risk profiles
    customers['risk_profile_encoded'] = le.transform(customers['risk_profile'])
    products['risk_level_encoded'] = le.transform(products['risk_level'])

    # Normalize net worth for similarity calculation
    customers['net_worth_normalized'] = customers['net_worth'] / customers['net_worth'].max()

    # Create feature vectors
    # For customers
    customer_features = customers[['net_worth_normalized', 'risk_profile_encoded']].values
    # For products
    product_features = products[['min_investment_normalized', 'risk_level_encoded']].values


    # Calculate similarity
    similarity_matrix = cosine_similarity(customer_features, product_features)

    return similarity_matrix

def get_recommendations(customer_id, top_n=2):
    customers = get_customers()
    products = get_products()

    customer_idx = customers[customers['id'] == customer_id].index[0]
    similarity_scores = get_similarity_matrix(customers, products)[customer_idx]
    product_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommended_products = products.iloc[product_indices]
    return recommended_products

REFERRAL_STATUS_OPTIONS = ['Pending', 'Completed', 'Failed']
PRODUCTS = ['Investment Package', 'Savings Account', 'Credit Card', 'Insurance Plan', 'Personal Loan']
AGREEMENT_TERMS = [
    "10% commission on successful referrals",
    "5% cashback for referrers on first purchase",
    "Referral bonus upon account activation",
    "Flat $50 reward for every successful referral",
    "Tiered reward system based on customer investment"
]

def create_tables():
    # Connect to SQLite database
    conn = sqlite3.connect('recommendation.db')
    c = conn.cursor()

    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS referrers (
    user_id INTEGER PRIMARY KEY,
    name TEXT,
    points INTEGER,
    badges TEXT
    )
    ''')

    c.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name TEXT,
        net_worth INTEGER,
        risk_profile TEXT,
        purchase_history TEXT
    )
    ''')

    c.execute('''CREATE TABLE IF NOT EXISTS referrals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        referrer_name TEXT,
        referral_date TEXT,
        customer_name TEXT,
        product_referred TEXT,
        referral_status TEXT,
        referrer_points INTEGER
    )
    ''')

    c.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT,
        agreement_terms TEXT,
        referral_reward INTEGER,
        risk_level TEXT, 
        min_investment INTEGER
    )
    ''')

    conn.commit()
    conn.close()

def run_read_sql(query):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('recommendation.db')
                
        # Read the data into a DataFrame
        df = pd.read_sql_query(query, conn)
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        df = pd.DataFrame()  # Return an empty DataFrame in case of error
        
    finally:
        # Close the database connection
        if conn:
            conn.close()
                
    return df



@st.cache_resource
def get_customers():
    return run_read_sql("SELECT * FROM customers")

@st.cache_resource
def get_products():
    return run_read_sql("SELECT * FROM products")

@st.cache_resource
def get_referrals():
    return run_read_sql("SELECT * FROM referrals")

@st.cache_resource
def get_referrers():
    return run_read_sql("SELECT * FROM referrers")

@st.cache_resource
def get_referrers_dict():
    referrers = get_referrers()[['user_id', 'name']]
    referrers_dict = referrers.set_index('name')['user_id'].to_dict()
    return referrers_dict

@st.cache_resource
def get_products_dict():
    products = get_products()[['id', 'product_name']]
    product_dict = products.set_index('product_name')['id'].to_dict()
    return product_dict

@st.cache_resource
def get_customers_dict(reversed=True):
    customers = get_customers()[['id', 'customer_name']]
    if reversed:
        customers_dict = customers.set_index('customer_name')['id'].to_dict()
    else:
        customers_dict = customers.set_index('id')['customer_name'].to_dict()
        
    return customers_dict

@st.cache_resource
def get_customers_risk_profile():
    customers = get_customers()[['id', 'risk_profile']]
    customers_dict = customers.set_index('id')['risk_profile'].to_dict()
    
    return customers_dict

def add_customer(name, net_worth, risk_profile):
    try:
        conn = sqlite3.connect('recommendation.db')
        c = conn.cursor()
        # Insert customer details into the customers table
        c.execute('INSERT INTO customers (name, net_worth, risk_profile) VALUES (?, ?, ?)',
                  (name, net_worth, risk_profile))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"An error occurred: {e}")
        return False

@st.cache_resource
def insert_synthetic_data():
    conn = sqlite3.connect('recommendation.db')
    c = conn.cursor()

    c.execute('''
    INSERT INTO products (product_name, agreement_terms, referral_reward, risk_level, min_investment)
    VALUES 
    ('Investment Package', '10% commission on successful referrals', 300, 'High', 750000),
    ('Savings Account', '5% cashback for referrers on first purchase', 150, 'High', 10000),
    ('Credit Card', 'Referral bonus upon account activation', 200, 'Medium', 0),
    ('Insurance Plan', 'Flat $50 reward for every successful referral', 50, 'Low', 200),
    ('Personal Loan', 'Tiered reward system based on customer investment', 400, 'Low', 500);
    ''')

    c.execute('''
    INSERT INTO referrals (referrer_name, referral_date, customer_name, product_referred, referral_status, referrer_points)
    VALUES
    ('John Smith', '2024-01-15', 'Alex Brown', 'Investment Package', 'Completed', 300),
    ('Jane Johnson', '2024-02-05', 'Taylor Davis', 'Savings Account', 'Pending', 0),
    ('Chris Williams', '2024-03-20', 'Jordan Garcia', 'Credit Card', 'Completed', 200),
    ('Alex Brown', '2024-04-10', 'Sam Miller', 'Insurance Plan', 'Failed', 0),
    ('Taylor Davis', '2024-05-22', 'John Smith', 'Personal Loan', 'Completed', 400),
    ('Jordan Garcia', '2024-06-14', 'Chris Williams', 'Investment Package', 'Pending', 0),
    ('Sam Miller', '2024-07-09', 'Jane Johnson', 'Savings Account', 'Completed', 150),
    ('Morgan Brown', '2024-08-03', 'Alex Brown', 'Credit Card', 'Failed', 0),
    ('Jordan Smith', '2024-09-12', 'Chris Garcia', 'Insurance Plan', 'Completed', 50),
    ('Sam Williams', '2024-10-01', 'Taylor Jones', 'Personal Loan', 'Pending', 0);
    ''')

    c.execute('''
    INSERT INTO customers (customer_name, net_worth, risk_profile, purchase_history)
    VALUES
    ('Alex Brown', 150000, 'High', 'Investment Package, Credit Card'),
    ('Taylor Davis', 50000, 'Low', 'Savings Account'),
    ('Jordan Garcia', 250000, 'Medium', 'Personal Loan, Insurance Plan'),
    ('Sam Miller', 80000, 'Low', 'Savings Account, Insurance Plan'),
    ('Chris Williams', 300000, 'High', 'Investment Package, Personal Loan'),
    ('John Smith', 120000, 'Medium', 'Credit Card, Savings Account'),
    ('Morgan Brown', 60000, 'Low', 'Insurance Plan'),
    ('Jane Johnson', 200000, 'High', 'Investment Package, Credit Card'),
    ('Chris Garcia', 90000, 'Medium', 'Personal Loan'),
    ('Taylor Jones', 75000, 'Low', 'Savings Account');
    ''')

    c.execute('''
    INSERT INTO referrers (name, points, badges)
    VALUES
    ('John Smith', 1500, 'Bronze, Silver'),
    ('Jane Johnson', 800, 'Gold, Bronze'),
    ('Chris Williams', 2000, 'Platinum, Diamond'),
    ('Alex Brown', 1200, 'Bronze'),
    ('Taylor Davis', 500, 'Silver, Gold'),
    ('Jordan Garcia', 1300, 'Gold, Platinum'),
    ('Sam Miller', 600, 'Bronze'),
    ('Morgan Brown', 1700, 'Platinum, Diamond, Gold'),
    ('Jordan Smith', 900, 'Silver, Bronze'),
    ('Taylor Jones', 1000, 'Gold');
    ''')

    conn.commit()
    conn.close()

def dashboard():

    st.title("Welcome to AI Driven Recommendation MVP")

    col1, col2, col3, _, _ = st.columns(5)
    
    if col1.button("🔄 Refresh Data"):
        get_customers.clear()
        get_products.clear()
        get_referrals.clear()
        get_referrers.clear()

    if col2.button("🔗 New Referral"):
        st.title("Submit a Referral")

        referrer_names = list(get_referrers_dict().keys())
        product_names = list(get_products_dict().keys())

        with st.form("referral_form"):
            referrer_name = st.selectbox("Select Referrer", referrer_names)
            customer_name = st.text_input("Customer Name")
            product_name = st.selectbox("Select Product", product_names)
            submitted = st.form_submit_button("Submit Referral")

            if submitted:
                if not customer_name.strip():
                    st.error("Please enter a customer name.")
                else:
                    
                    # Map names back to IDs
                    referrer_id = get_referrers_dict()[referrer_name]
                    product_id = get_products_dict()[product_name]
                    # Insert into the referrals table
                    conn = sqlite3.connect('recommendation.db')
                    c = conn.cursor()
                    c.execute('INSERT INTO referrals (user_id, customer_name, product_id, status, timestamp) VALUES (?, ?, ?, ?, ?)',
                            (referrer_id, customer_name, product_id, 'Pending', datetime.now()))
                    conn.commit()
                    conn.close()
                    st.success("Referral submitted successfully!")

                    get_referrals.clear()

    if col3.button("🆕 New Customer"):
        st.title("Submit a New Customer")

        with st.form("customer_form"):
            customer_name = st.text_input("Customer Name")
            net_worth = st.number_input("Net Worth", min_value=0, step=1000)
            risk_profile = st.selectbox("Risk Profile", ["Low", "Medium", "High"])
            
            # Submit button for the form
            submitted = st.form_submit_button("Add Customer")

            if submitted:
                if not customer_name.strip():
                    st.error("Customer Name is required.")
                else:
                    # Add the new customer to the database
                    success = add_customer(customer_name, net_worth, risk_profile)
                    if success:
                        st.success("Customer added successfully!")
            get_customers.clear()

    st.header("Referrals Overview")
    display_beautified_dataframe(get_referrals())

    col1, col2 = st.columns(2)

    with col1:
        st.header("Current Pipeline")
        st.bar_chart(get_referrals()['referral_status'].value_counts())

    with col2:
        st.header("Top Referrers")
        display_beautified_dataframe(get_referrers().sort_values(by='points', ascending=False))

    col1, col2 = st.columns(2)

    with col1:
        st.header("Customers")
        display_beautified_dataframe(get_customers()[['customer_name', 'net_worth', 'risk_profile', 'purchase_history']])

    with col2:
        st.header("AI-Driven Product Recommendations")
        #display_beautified_dataframe(get_referrers().sort_values(by='points', ascending=False))
        product_recommendations()

    if st.button("Clear Cache"):
        st.cache_resource.clear()

def leaderboard():
    st.title("Leaderboard")
    # Fetch users and points from database
    # Display top users

import streamlit as st

def product_recommendations():
    recommendations_list = []

    customers = get_customers()
    # for customer in customers['id']:
    #         recommendations = get_recommendations(customer)
    #         for _, row in recommendations.iterrows():
    #             st.write(f"**{row['product_name']}** - Risk Level: {row['risk_level']}")
    

    for customer_id in customers['id']:
        recommendations = get_recommendations(customer_id)
        for _, row in recommendations.iterrows():
            # Append details as a dictionary to the list
            recommendations_list.append({
                'customer_name': get_customers_dict(reversed=False)[customer_id],
                'customer_risk_profile': get_customers_risk_profile()[customer_id],
                'product_name': row['product_name'],
                'product_risk_level': row['risk_level']
            })

    # Convert the list of recommendations to a DataFrame
    recommendations_df = pd.DataFrame(recommendations_list)

    # Display the DataFrame in Streamlit
    if not recommendations_df.empty:
        display_beautified_dataframe(recommendations_df)
    else:
        st.write("No recommendations available.")

def analytics():
    st.title("Analytics")
    # Display KPIs and charts

def main():
    st.sidebar.title("Navigation")
    radio_selection = st.sidebar.radio("Go to", ["Dashboard", "Analytics", "Admin"])

    st.query_params.update(selection=radio_selection)
    selection = st.query_params.get("selection", "Dashboard")

    if selection == "Dashboard":
        dashboard()
    elif selection == "Analytics":
        analytics()
    elif selection == "Admin":
        if st.button("Insert Syntetic Data"):
            insert_synthetic_data()
    else:
        st.error("Something went wrong!")

if __name__ == '__main__':
    create_tables()
    st.set_page_config(page_title="recommendation MVP", page_icon="🚀", initial_sidebar_state="collapsed", layout="wide")
    main()
