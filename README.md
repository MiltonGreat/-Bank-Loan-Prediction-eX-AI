# Loan Approval Prediction & Fairness Analysis

![screenshot-localhost_8888-2025 04 11-11_41_37](https://github.com/user-attachments/assets/0d7c2ca9-7673-498b-be1d-6130d659a208)

### Overview

This project analyzes loan approval decisions using Logistic Regression and Random Forest models, with a focus on:

- Predictive performance
- Feature importance analysis
- Fairness evaluation across education levels

The goal is to understand key approval drivers while assessing potential biases in model predictions.

### Dataset

#### Columns
- Demographics: Gender, Married, Dependents, Education, Self_Employed
- Financials: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
- Property: Property_Area (Urban/Rural/Semiurban)
- Target: Loan_Status (Y/N)

### Key Steps

1. Data Preprocessing
Handled missing values:
- Categorical: Filled with mode (e.g., Gender, Self_Employed).
- Numerical: Filled with median/mode (e.g., LoanAmount, Credit_History).

Feature engineering:
- Created Income_per_dependent, EMI, Balance_Income.
- Log-transformed skewed features (LoanAmount, Total_Income).

2. Model Training & Evaluation
- Logistic Regression	82.00% (Accuracy), 	84%	52% (Precision)
- Random Forest	83.00% (Accuracy), 	79%	61% (Precision)
- Tuned Random Forest	84.00% (Accuracy), 	89%	55% (Precision)
- Best Model: Tuned Random Forest

### Key Takeaways

**1. Top Features Influencing Approval**
- Credit History is the strongest predictor (30% weight in Logistic Regression).
- Income-related features (e.g., Balance_Income) are more impactful in Random Forest.

**2. Fairness Metrics by Education Level**
- Random Forest is fairer in approvals for qualified applicants (lower Equal Opportunity bias).
- Logistic Regression has slightly better overall approval parity.
- Credit History dominates approval decisions, but income stability matters in Random Forest.
- Education bias exists but is small (≤9% disparity). Random Forest reduces bias in qualified approvals.

### Conclusion

While both models perform well, the choice between them depends on the prioritization of fairness metrics (equal opportunity vs. demographic parity) and interpretability. This analysis provides a framework for ethical AI deployment in lending, balancing predictive power with equitable outcomes.

### Source

[Loan Application Data from Kaggle](https://www.kaggle.com/datasets/vipin20/loan-application-data)


