import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import feature_engineer as fe # Import the feature engineering module

# Define the filename where the trained model will be saved
MODEL_FILENAME = 'model.pkl'

def train_and_evaluate_model():
    """Loads data, trains the model, and reports performance."""
    
    try:
        print("--- BEC Phishing Detection Model Training Initiated ---")
        
        # 1. Load Data
        print("1. Loading and pre-processing data...")
        df = pd.read_csv('simulated_emails.csv')
        
        # 2. Feature Engineering & Split
        X_train, X_test, y_train, y_test = fe.prepare_data(df)
        
        # 3. Model Training (Using Random Forest Classifier)
        print("\n3. Training Random Forest Classifier (Class Weight Adjusted for Recall)...")
        
        # --- CRITICAL CHANGE FOR RECALL ---
        # We explicitly set class weights to penalize misclassifying the malicious class (1)
        # much more heavily than the legitimate class (0). This forces the model to 
        # prioritize avoiding False Negatives (missed attacks).
        class_weights = {0: 1, 1: 5} # Malicious class is 5x more important
        
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight=class_weights # Use the custom weights dictionary
        )
        model.fit(X_train, y_train)
        print("   Training complete.")
        
        # 4. Model Evaluation
        print("\n4. Evaluating model performance on the test set...")
        y_pred = model.predict(X_test)
        
        # Report
        print("\n--- Classification Report (AFTER WEIGHT ADJUSTMENT) ---")
        print("NOTICE: Recall for 'Malicious' should be much higher now.")
        print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Malicious (1)']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n--- Confusion Matrix (Actual vs. Predicted) ---")
        print(f"           Predicted 0  |  Predicted 1")
        print(f"Actual 0:  {cm[0, 0]:<12} |  {cm[0, 1]:<11}")
        print(f"Actual 1:  {cm[1, 0]:<12} |  {cm[1, 1]:<11}")
        
        # 5. Save Model
        joblib.dump(model, MODEL_FILENAME)
        print(f"\n5. ✅ Model successfully trained and saved to '{MODEL_FILENAME}'")
        
    except FileNotFoundError:
        print("\n❌ Error: 'simulated_emails.csv' not found.")
        print("Please ensure you have run 'python data_simulator.py' first.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during training: {e}")

if __name__ == '__main__': # <-- THIS IS THE CORRECTED LINE
    train_and_evaluate_model()