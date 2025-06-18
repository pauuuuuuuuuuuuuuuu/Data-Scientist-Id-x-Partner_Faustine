#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[6]:


from matplotlib import pyplot as plt


# In[7]:


get_ipython().system('pip install pandas numpy matplotlib seaborn scikit-learn plotly')


# In[9]:


import pandas as pd

df = pd.read_csv("loan_data_2007_2014.csv")
df.head()


# In[5]:


df = pd.read_csv('loan_data_2007_2014.csv')

df.shape
df.info()
df.describe()
df.isnull().sum()


# In[57]:


df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['loan_status'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

# Check if term is already numeric, if not then extract the number
if df['term'].dtype == 'object':
    df['term'] = df['term'].str.extract(r'(\d+)').astype(float)  # Fixed the regex by removing extra backslash

# Check if int_rate is already numeric, if not then convert it
if df['int_rate'].dtype == 'object':
    df['int_rate'] = df['int_rate'].str.replace('%','').astype(float)

df = df.dropna(subset=['loan_amnt','term','int_rate','installment','annual_inc'])


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({
    'loan_status': ['Approved', 'Rejected', 'Approved', 'Approved', 'Rejected'],
    'loan_amnt': [5000, 10000, 7500, 12000, 8000]
})

sns.set_theme(style='darkgrid')

if 'loan_status' in df.columns and not df['loan_status'].empty:
    sns.countplot(x='loan_status', data=df, palette='Blues')
    plt.title('Loan Status Distribution')
    plt.show()
else:
    print("Error: 'loan_status' column is missing or empty")

if 'loan_amnt' in df.columns and not df['loan_amnt'].empty:
    sns.histplot(df['loan_amnt'], kde=True, color='royalblue')
    plt.title('Loan Amount Distribution')
    plt.show()
else:
    print("Error: 'loan_amnt' column is missing or empty")


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("Available columns:", df.columns.tolist())

X = df[['loan_amnt']]  
y = df['loan_status']  

# Check the distribution of classes in y
print("Target variable distribution:")
print(y.value_counts())

# If there's only one class, you might need to use a different dataset or feature
# For now, let's modify the code to avoid the ROC AUC calculation

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# Only calculate ROC AUC if there are two classes
if len(set(y_test)) > 1:
    from sklearn.metrics import roc_auc_score
    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
else:
    print("ROC AUC Score cannot be calculated: only one class present in the target variable")


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Display current confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print(f"Current class distribution in test data: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# SOLUTION: Create a new train/test split from your original data
# Make sure you have access to your original dataset (X and y)

# First, check the overall class distribution
print(f"Overall class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Create a new stratified split with a larger test size to ensure both classes are present
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Verify the new test set has both classes
print(f"New test set class distribution: {dict(zip(*np.unique(y_test_new, return_counts=True)))}")

# Retrain your model with the new split
model.fit(X_train_new, y_train_new)
y_pred_new = model.predict(X_test_new)

# Calculate and display alternative metrics
print(f"Accuracy: {accuracy_score(y_test_new, y_pred_new):.4f}")
print(f"Precision: {precision_score(y_test_new, y_pred_new, pos_label='Rejected'):.4f}")
print(f"Recall: {recall_score(y_test_new, y_pred_new, pos_label='Rejected'):.4f}")
print(f"F1 Score: {f1_score(y_test_new, y_pred_new, pos_label='Rejected'):.4f}")

# Now calculate ROC curve and AUC with the new split
from sklearn.metrics import roc_curve, roc_auc_score

# Check if we have at least two classes in the new test data
if len(set(y_test_new)) >= 2:
    # Get the probability predictions
    y_prob_new = model.predict_proba(X_test_new)[:,1]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test_new == 'Rejected', y_prob_new)
    auc_score = roc_auc_score(y_test_new == 'Rejected', y_prob_new)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Model (AUC = {auc_score:.2f})")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
else:
    print("Still unable to generate ROC curve: not enough class diversity in the data")


# In[26]:


# First, define and train your Random Forest model
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor

# Create and train the model (assuming X and y are your features and target)
rf_model = RandomForestClassifier()  # Add parameters as needed
rf_model.fit(X, y)  # Fit the model with your data

# Now extract feature importances
importances = rf_model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(8,5))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='Blues')
plt.title('Top Features - Random Forest')
plt.tight_layout()
plt.show()


# In[41]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# 1. Load the data with the correct filename
try:
    # Use your specific CSV filename
    df = pd.read_csv('loan_data_2007_2014.csv')

    # 2. Check if data was loaded correctly
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())

    # 3. Check for missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # 4. Handle missing values if needed
    # df = df.dropna()  # or df.fillna(value)

    # 5. Check if required columns exist
    required_columns = ['loan_amnt', 'loan_status']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"\nWarning: The following required columns are missing: {missing_columns}")
        print("Available columns:", df.columns.tolist())
    else:
        # 6. Prepare features and target
        X = df[['loan_amnt']]  # Only 'loan_amnt' is available as a feature
        y = df['loan_status']

        # Check if we have data before proceeding
        if len(X) > 0:
            # 7. Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 8. Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # 9. Logistic Regression
            log_model = LogisticRegression(max_iter=1000)
            log_model.fit(X_train, y_train)
            y_pred_log = log_model.predict(X_test)

            # 10. Random Forest Classifier
            rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)

            # 11. Evaluation
            print("\n=== Logistic Regression ===")
            print(classification_report(y_test, y_pred_log))
            print("ROC AUC:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:,1]))

            print("\n=== Random Forest Classifier ===")
            print(classification_report(y_test, y_pred_rf))
            print("ROC AUC:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))

            # 12. Confusion Matrix & ROC Curve
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix (Random Forest)')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

            plt.figure(figsize=(10, 8))
            fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
            plt.plot(fpr, tpr, label='Random Forest (AUC = %.2f)' % roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1]))
            plt.plot([0,1], [0,1], 'k--')
            plt.title('ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.grid(True)
            plt.show()

            # 13. Feature Importance Visualization
            importances = rf_model.feature_importances_
            features = ['loan_amnt']  # Only one feature
            feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})

            plt.figure(figsize=(8,5))
            sns.barplot(data=feat_df, x='Importance', y='Feature', palette='Blues')
            plt.title('Feature Importance - Random Forest')
            plt.tight_layout()
            plt.show()
        else:
            print("\nNo data available in the selected columns.")

except FileNotFoundError:
    print("Error: Data file 'loan_data_2007_2014.csv' not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred while loading or processing the data: {str(e)}")


# In[42]:


print("Dataset Shape:", df.shape)
print("\nTipe Data Tiap Kolom:")
print(df.dtypes.value_counts())

print("\nStatistik Deskriptif Kolom Numerik:")
print(df.describe())

print("\nJumlah Nilai Unik per Kolom:")
print(df.nunique())

print("\nJumlah Missing Values per Kolom:")
print(df.isnull().sum())


# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='Blues')
plt.title('Correlation Matrix')
plt.show()


# In[47]:


date_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d']

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

for col in date_cols:
    df[col + '_num'] = df[col].dt.year * 100 + df[col].dt.month  # Format: YYYYMM

# Create the correlation matrix from the numeric date columns
date_num_cols = [col + '_num' for col in date_cols]
corr_matrix = df[date_num_cols].corr()  # This creates the correlation matrix

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='magma', vmin=0, vmax=1)
plt.title('Correlation of Date Features and Loan Status')
plt.show()


# In[58]:


# Struktur data
df.shape
df.info()
df.describe()
df.isnull().sum().sort_values(ascending=False)


# In[60]:


# Korelasi numerik
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Features)')
plt.show()


# In[ ]:




