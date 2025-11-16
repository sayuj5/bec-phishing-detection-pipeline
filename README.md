🚨 BEC/Phishing Detection Pipeline

This project is an **end-to-end Machine Learning pipeline** built to classify emails as Legitimate (0) or Malicious (1), focusing on **Business Email Compromise (BEC)** and phishing attempts.

The core is a **Random Forest Classifier** which has been strategically tuned using class weights to achieve **high Recall (92%+)**. This minimizes the risk of missing a real threat (False Negative), prioritizing security above all else.

## 🚀 Quick Run

1.  `python data_simulator.py`
2.  `python train_model.py` (Trains the high-recall model)
3.  `python predict_email.py` (Tests a suspicious email)
