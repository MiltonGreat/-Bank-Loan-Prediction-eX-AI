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

### Workflow Outline

1. Data Loading & Preprocessing
- Handle missing values (categorical: mode imputation; numerical: median/mode imputation).
- Clean and convert Total_Income (remove currency symbols, convert to float).
- Encode categorical variables (e.g., Loan_Status: Y=1, N=0).
- Feature engineering: Create Income_per_dependent, EMI, Balance_Income, and log-transformed features.

2. Data Splitting & Scaling
- Split data into train (80%) and test (20%) sets.
- Scale features for Logistic Regression (StandardScaler).
- Keep data unscaled for Random Forest (tree-based models don’t require scaling).

3. Model Training
- Logistic Regression: Trained on scaled features.
- Random Forest: Trained on raw features.

4. Feature Importance Analysis
- Extract and normalize Logistic Regression coefficients.
- Use built-in Random Forest feature importances.
- Visualize top features for both models (bar plots).

5. Model Interpretability
- Partial Dependence Plots (PDPs):
- Continuous features (LoanAmount_Log, Total_Income_Log).
- Binary features (Credit_History, Education, Gender).

Individual Conditional Expectation (ICE) Plots:
- Analyze LoanAmount_Log impact on individual predictions.

6. Fairness Evaluation

Metrics Calculated:
- Demographic Parity: Difference and ratio in approval rates between graduates/non-graduates.
- Equal Opportunity: Difference and ratio in true positive rates.
- Visualization: Bar plots comparing fairness metrics for both models.

7. Education & Income Impact Analysis
- Plot predicted approval probabilities vs. Total_Income for:
- Graduates (Education=1).
- Non-graduates (Education=0).

8. Results & Recommendations
- Compare model performance, interpretability, and fairness.
- Suggest mitigation strategies for bias (e.g., threshold adjustment).

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


