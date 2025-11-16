import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

def generate_bec_data(n_samples=2000):
    """Generates synthetic data for BEC/Phishing detection."""
    
    # 1. Feature Generation
    
    # Urgency Score (0 to 1) - Higher for malicious emails
    urgency = np.random.beta(a=3, b=5, size=n_samples) # Biased toward lower (legitimate)
    
    # Domain Similarity Score (0 to 1) - Higher means closer to known good domain (malicious uses typosquatting)
    # Malicious emails will have a score clustered around 0.8 (close but not perfect match)
    domain_sim = np.random.beta(a=5, b=2, size=n_samples) # Biased toward higher (spoofing)
    
    # Financial Keyword Count - Higher for malicious
    finance_keywords = np.random.poisson(lam=2, size=n_samples)
    
    # Request Type (Categorical Feature)
    # 0: None, 1: Wire Transfer, 2: Credential Update
    request_type = np.random.randint(0, 3, size=n_samples)
    
    # Anomalous Sender History (0 or 1) - Malicious emails often come from new/unknown senders
    sender_anomaly = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # 2. Label Generation (The Target: 1=Malicious, 0=Legitimate)
    
    # Create a base probability for being malicious
    # High urgency, high domain similarity (typosquatting), and specific request types increase probability
    base_prob = (0.2 
                 + 0.5 * urgency 
                 + 0.3 * (domain_sim > 0.7) 
                 + 0.2 * (request_type > 0) 
                 + 0.4 * sender_anomaly)
    
    # Clamp probabilities between 0 and 1
    base_prob = np.clip(base_prob, 0.05, 0.95)
    
    # Apply a slight increase in probability for higher financial keyword counts
    for i in range(n_samples):
        if finance_keywords[i] > 3:
            base_prob[i] *= 1.2
            
    # Determine the final label (1 or 0) based on the calculated probability
    labels = (np.random.rand(n_samples) < base_prob).astype(int)
    
    # 3. Assemble DataFrame
    df = pd.DataFrame({
        'urgency_score': urgency,
        'domain_similarity_score': domain_sim,
        'financial_keyword_count': finance_keywords,
        'request_type': request_type,
        'sender_anomaly': sender_anomaly,
        'label': labels
    })

    # Ensure a reasonable balance (BEC is usually imbalanced, but we need enough samples to train)
    
    return df

if __name__ == '__main__': # <-- THIS LINE IS THE FIX
    df = generate_bec_data(n_samples=2000)
    df.to_csv('simulated_emails.csv', index=False)
    print(f"Generating {len(df)} synthetic email records...")
    print("Successfully created simulated_emails.csv with synthetic data.")