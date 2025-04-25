import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load Real SQL Query Logs (Updated Caching Method)
@st.cache_data
def load_data_from_csv(file):
    # The file is passed in directly from the uploader widget
    return pd.read_csv(file)

# 2. Preprocess Data and Train the Model
def preprocess_and_train_model(df):
    # Ensure 'avg_exec_time_ms' is numeric, coercing any errors into NaN
    df['avg_exec_time_ms'] = pd.to_numeric(df['avg_exec_time_ms'], errors='coerce')

    # Handle NaN values: either drop or fill them
    df.dropna(subset=['avg_exec_time_ms'], inplace=True)  # Drop rows with NaN in 'avg_exec_time_ms'
    # Or you could fill NaN with a default value:
    # df['avg_exec_time_ms'].fillna(0, inplace=True)

    # Define 'slow' query threshold (avg_exec_time_ms > 1000 ms)
    df['is_slow'] = df['avg_exec_time_ms'] > 1000
    features = ['query_length', 'num_joins', 'has_subquery', 'uses_index']

    # Check if all expected features are present
    missing_columns = [col for col in features if col not in df.columns]
    if missing_columns:
        st.error(f"Error: The following expected columns are missing: {', '.join(missing_columns)}")
        return None  # Exit early if any required feature columns are missing

    # Extract features and target
    X = df[features]
    y = df['is_slow'].astype(int)

    # Check if data is numeric and no NaN values
    st.write("Features (X):", X.head())
    st.write("Target (y):", y.head())
    st.write("Data Types:", X.dtypes)

    # Drop rows with NaN values in the feature set
    X = X.dropna()
    y = y[X.index]  # Make sure to align y with X after dropping NaN rows

    # Verify that X and y are properly shaped
    st.write("Shape of X:", X.shape)
    st.write("Shape of y:", y.shape)

    if X.shape[0] == 0 or y.shape[0] == 0:
        st.error("Error: No valid data available for training. Check the input CSV.")
        return None

    # Train a RandomForest model
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# 3. Add a Recommendation Engine
def recommend_tips(query):
    tips = []
    if query['query_length'] > 800:
        tips.append("🔍 Query is long — consider breaking it into smaller chunks.")
    if query['num_joins'] > 3:
        tips.append("🪢 Too many JOINs — simplify joins or add proper indexing.")
    if query['has_subquery']:
        tips.append("🧠 Subquery detected — flatten subqueries if possible.")
    if not query['uses_index']:
        tips.append("⚡ Index not used — create indexes on filter/join columns.")

    if not tips:
        tips.append("✅ Query structure looks optimized.")
    return tips

# 4. Streamlit App Interface
def main():
    st.title("SQL Query Performance Predictor")

    # Step 1: Load the Data
    uploaded_file = st.file_uploader("Upload your SQL query log file (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = load_data_from_csv(uploaded_file)  # This now happens after the upload
    else:
        st.warning("Please upload a CSV file.")
        return

    # Display the column names and preview the data for debugging
    st.subheader("Query Logs Preview")
    st.write(df.head())  # Show the first few rows of the data
    st.write("Data Columns:", df.columns)  # Show the column names

    # Step 2: Train the Model
    model = preprocess_and_train_model(df)

    if model is None:
        return  # Stop execution if the model couldn't be trained due to missing data

    # Step 3: User Input for Query Analysis
    st.subheader("Enter Your SQL Query")
    query_text = st.text_area("SQL Query", height=150)

    if query_text:
        # Process the query to extract features
        query_length = len(query_text)
        num_joins = (query_text.lower().count('join') // 4)  # Approximation
        has_subquery = 1 if 'select' in query_text.lower() and 'from' in query_text.lower() and 'select' in query_text.lower() else 0
        
        # Dummy logic to determine if an index is used — you can extend this logic with actual parsing
        uses_index = 1 if "index" in query_text.lower() else 0
        
        query_features = pd.DataFrame({
            'query_length': [query_length],
            'num_joins': [num_joins],
            'has_subquery': [has_subquery],
            'uses_index': [uses_index]
        })

        # Step 4: Prediction
        prediction = model.predict(query_features)[0]

        # Show result
        if prediction == 1:
            st.error("🛑 This query is likely to be **Slow**.")
        else:
            st.success("✅ This query is likely to be **Fast**.")

        # Show optimization recommendations
        st.subheader("🛠️ Optimization Tips")
        recommendations = recommend_tips(query_features.iloc[0])
        for tip in recommendations:
            st.write(tip)

# Run the Streamlit app
if __name__ == '__main__':
    main()
