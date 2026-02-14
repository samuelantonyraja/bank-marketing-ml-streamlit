# Bank Marketing ML Classification

## Problem Statement
Predict whether a client subscribes to a term deposit using classification models.

## Dataset Description
- Source: UCI Bank Marketing Dataset
- Instances: 45,211
- Features: 16 input features
- Target: y (Yes/No subscription)

## Models Used
Logistic Regression  
Decision Tree  
KNN  
Naive Bayes  
Random Forest  
XGBoost  

## Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|------|-----------|--------|-----|-----|
| Logistic Regression | 0.8987 | 0.6507 | 0.6291 | 0.3270 | 0.4303 | 0.4054 |
| Decision Tree | 0.8646 | 0.6855 | 0.4260 | 0.4518 | 0.4385 | 0.3618 |
| KNN | 0.8938 | 0.6438 | 0.5853 | 0.3175 | 0.4117 | 0.3793 |
| Naive Bayes | 0.8659 | 0.6961 | 0.4335 | 0.4744 | 0.4530 | 0.3773 |
| Random Forest | 0.9065 | 0.6834 | 0.6726 | 0.3922 | 0.4955 | 0.4677 |
| XGBoost | 0.9026 | 0.7038 | 0.6167 | 0.4442 | 0.5164 | 0.4717 |

## Observations
XGBoost achieved the best overall performance in terms of AUC, F1 score, and MCC.  
Random Forest also performed strongly.  
Logistic Regression provided a strong baseline.  
Naive Bayes showed better recall than Logistic Regression.  
KNN and Decision Tree performed moderately due to class imbalance.
