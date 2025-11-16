import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(df):
    """
    Applies feature engineering and splits data into training/testing sets.

    In a real scenario, this is where you would perform:
    - Text vectorization (TF-IDF or BERT embeddings)
    - Domain reputation lookups
    - Header analysis (SPF/DKIM checks)
    
    For this simulation, we apply One-Hot Encoding on the categorical feature.
    """
    
    # Separate features (X) from the target label (y)
    X = df.drop('label', axis=1)
    y = df['label']
    
    print("\n[Feature Engineering] Applying One-Hot Encoding to 'request_type'...")
    # One-Hot Encode the categorical 'request_type' feature
    X = pd.get_dummies(X, columns=['request_type'], prefix='req')
    
    # Drop the original numerical request_type column if it wasn't dropped by get_dummies
    if 'request_type' in X.columns:
        X = X.drop('request_type', axis=1)
    
    # Convert DataFrame to NumPy arrays for scikit-learn
    X_processed = X.values
    y_processed = y.values
    
    print(f"[Feature Engineering] Final features shape: {X_processed.shape}")
    print(f"[Feature Engineering] Target labels shape: {y_processed.shape}")
    
    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__': # <-- THIS IS THE CORRECTED LINE
    # This block is for testing the feature engineer independently
    try:
        df = pd.read_csv('simulated_emails.csv')
        X_train, X_test, y_train, y_test = prepare_data(df)
        print("\nFeature Engineering Test Successful.")
        print(f"Training set size: {len(X_train)} samples")
        print(f"Testing set size: {len(X_test)} samples")
    except FileNotFoundError:
        print("Error: 'simulated_emails.csv' not found. Please run data_simulator.py first.")