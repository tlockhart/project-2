import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

file_location = "./Resources/data_science_salaries.csv"
salary_df = pd.read_csv(file_location)

# Binning function
def bin_y(salary_df):
    bins = [1500, 50000, 156000, 176000, 750000]
    group_names = ["very low", "low", "average", "high"]
    salary_df["salary_binned"] = pd.cut(salary_df["salary_in_usd"], bins, labels=group_names, include_lowest=True)
    return salary_df

salary_df = bin_y(salary_df)

# Preprocess
X = salary_df.drop(columns=["salary", "salary_currency", "salary_in_usd", "work_year", "salary_binned"], axis=1)
y = salary_df["salary_binned"]

# Encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, random_state=1)

best_params = {'n_estimators': 70, 'learning_rate': 1.0, 'algorithm': 'SAMME'}
model = AdaBoostClassifier(**best_params)
model.fit(X_train, y_train)

# Save model, encoder, and feature names
joblib.dump(model, 'salary_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X_encoded.columns, 'feature_names.pkl')

print("Model, Label Encoder, and Feature Names Saved Successfully!")