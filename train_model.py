import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib



df = pd.read_csv(r"C:\Users\akars\OneDrive\Documents\loan-approval-cleaned.csv")



if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)


df = df.dropna()
print(df["Loan_Status"].value_counts())

y = df["Loan_Status"]


X = df.drop("Loan_Status", axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

joblib.dump(model, "loan_model.pkl")

print("Model saved successfully!")