import joblib
import pandas as pd
import numpy as np

# Define the filename where the trained model was saved
# --- FIX: Changed from 'bec_detector_model.pkl' to 'model.pkl' to match your training output. ---
MODEL_FILENAME = 'model.pkl'

def predict_new_email(new_email_data):
    """
    Loads the trained model and predicts the class (Legitimate/Malicious) 
    for a single, new email record.
    """
    try:
        # Load the trained model
        model = joblib.load(MODEL_FILENAME)
        print(f"✅ Model '{MODEL_FILENAME}' successfully loaded.")
        
        # 1. Prepare the new email data for prediction
        
        # Convert the dictionary to a DataFrame
        df_new = pd.DataFrame([new_email_data])
        
        # Manually apply the same One-Hot Encoding used in feature_engineer.py
        
        # First, ensure the request_type is numeric
        df_new['request_type'] = df_new['request_type'].astype(int)
        
        # Create dummy data template to ensure all 3 OHE columns exist
        temp_data = {'request_type': [0, 1, 2]}
        df_template = pd.DataFrame(temp_data)
        df_template = pd.get_dummies(df_template, columns=['request_type'], prefix='req')
        
        # Filter down the template columns to only keep the OHE columns (req_X)
        expected_cols = [col for col in df_template.columns if col.startswith('req_')]
        
        # Now apply OHE to the real data
        df_processed = pd.get_dummies(df_new, columns=['request_type'], prefix='req')
        
        # Add missing columns (which will be 0) to match the expected training feature set
        for col in expected_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # IMPORTANT: This order must match the feature names used during training!
        # Your original features likely included 'body_length' and 'capitalization_ratio' 
        # which were generated but then dropped in the test case below.
        # Since we are focusing on the features passed in the dictionary, we use a subset.
        # This list must match the column order the model expects.
        feature_order = [
            'urgency_score', 
            'domain_similarity_score', 
            'financial_keyword_count', 
            'sender_anomaly', 
            'req_0',  
            'req_1',  
            'req_2'   
        ]
        
        # Filter and reorder
        X_predict = df_processed.reindex(columns=feature_order, fill_value=0)
        
        # Convert to numpy array for the model
        X_predict_array = X_predict.values
        
        # 2. Make Prediction
        
        # Predict the class (0 or 1)
        prediction = model.predict(X_predict_array)[0]
        
        # Get the probability of being Malicious (class 1)
        probability = model.predict_proba(X_predict_array)[0][1]
        
        print("\n--- Prediction Result ---")
        if prediction == 1:
            print(f"🚨 CLASSIFICATION: MALICIOUS (PHISHING/BEC ATTEMPT)")
            print(f"   Confidence Score (Prob. of Malicious): {probability:.2f}")
        else:
            print(f"✅ CLASSIFICATION: LEGITIMATE EMAIL")
            print(f"   Confidence Score (Prob. of Malicious): {probability:.2f}")

    except FileNotFoundError:
        print(f"\n❌ Error: Model file '{MODEL_FILENAME}' not found.")
        print("Please ensure train_model.py ran successfully and saved the file.")
    except Exception as e:
        print(f"\n❌ An error occurred during prediction: {e}")
        print("This might be due to a mismatch in feature columns or the structure of the model.")

if __name__ == '__main__':
    
    # --- SIMULATE A HIGH-RISK INCOMING EMAIL ---
    # Testing a highly suspicious email
    suspicious_email = {
        'urgency_score': 0.95,        
        'domain_similarity_score': 0.90, 
        'financial_keyword_count': 5, 
        'request_type': 2,            
        'sender_anomaly': 1           
    }

    print("--- Simulating New Email Data ---")
    print(f"Testing a suspicious email: {suspicious_email}")
    predict_new_email(suspicious_email)