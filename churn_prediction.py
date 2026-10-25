import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score

def generate_dummy_data():
    '''Generates synthetic customer data to ensure the script runs out of the box.'''
    np.random.seed(42)
    n = 1000
    data = {
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_charges': np.random.uniform(20, 120, n),
        'total_charges': np.random.uniform(20, 8000, n),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'churn': np.random.choice(['Yes', 'No'], n, p=[0.26, 0.74])
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("📊 Loading Data...")
    df = generate_dummy_data()

    print("🧹 Preprocessing Data...")
    le = LabelEncoder()
    categorical_cols = ['contract_type', 'tech_support', 'churn']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop('churn', axis=1)
    y = df['churn']

    scaler = StandardScaler()
    X[['tenure_months', 'monthly_charges', 'total_charges']] = scaler.fit_transform(
        X[['tenure_months', 'monthly_charges', 'total_charges']]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("🤖 Training Model (Random Forest)...")
    # Using class_weight='balanced' to handle churn imbalance
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("📈 Evaluating Model...")
    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    recall = recall_score(y_test, y_pred)
    print(f"\n💡 Model Recall for Churn: {recall:.2f}")
    print("Business Note: High recall is prioritized to catch as many potential churners as possible!")
