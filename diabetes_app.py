import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ------------------- Load and Prepare Dataset -------------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)

# Clean the data
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    median = data[col].median()
    data[col] = data[col].replace(0, median)

# Split features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train_sm)

# ------------------- Streamlit App -------------------
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk:")

# Input form
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=30)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    # Prepare input
    new_patient = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                               columns=X.columns)
    new_patient_scaled = scaler.transform(new_patient)
    prediction = rf_model.predict(new_patient_scaled)[0]

    # Display result
    if prediction == 1:
        st.error("Prediction: Diabetic (High Risk)")
    else:
        st.success("Prediction: Non-Diabetic (Low Risk)")
