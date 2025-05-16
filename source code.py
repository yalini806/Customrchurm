
# Install libraries (run this only once)
# !pip install pandas scikit-learn

# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 2. Load Dataset (local file)
df = pd.read_csv("Telco-Customer-Churn.csv")

# 3. Basic Preprocessing
df = df[df['TotalCharges'] != " "]
df['TotalCharges'] = df['TotalCharges'].astype(float)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.drop(['customerID'], axis=1, inplace=True)

# Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 4. Train/Test Split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Save Predictions to CSV
results_df = X_test.copy()
results_df['Actual_Churn'] = y_test.values
results_df['Predicted_Churn'] = y_pred
results_df.to_csv('churn_predictions.csv', index=False)

print("Predictions saved to 'churn_predictions.csv'")
