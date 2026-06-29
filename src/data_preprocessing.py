import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    columns_to_drop = ['Unnamed: 0', 'flight']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    return df

def perform_feature_engineering(df):
    class_mapping = {'Economy': 0, 'Business': 1}
    df['class'] = df['class'].map(class_mapping)
    
    stops_mapping = {'zero': 0, 'one': 1, 'two_or_more': 2}
    df['stops'] = df['stops'].map(stops_mapping)
    
    #  One-Hot Encoding 
    categorical_columns = ['airline', 'source_city', 'destination_city', 'departure_time', 'arrival_time']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    return df_encoded

#"random_state" sets a constant for randomness in the system.
def split_data(df, target_column='price', test_size=0.2, random_state=42):
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    raw_data_path = "data/raw/flights.csv"
    print("Data processing is starting...")
    
    df_clean = load_and_clean_data(raw_data_path)
    print("Unnecessary columns removed.")
    
    df_processed = perform_feature_engineering(df_clean)
    print(f"Categorical data has been converted to numbers. New column count: {df_processed.shape[1]}")

    X_train, X_test, y_train, y_test = split_data(df_processed)
    print(f"Data was split. Training set size: {X_train.shape[0]} rows, Test set size: {X_test.shape[0]} rows.")
