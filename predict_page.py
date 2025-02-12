import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor = data["model"]

# Define specified categories
countries = (
    "United States",
    "India", 
    "United Kingdom",
    "Germany",
    "Canada",
    "Brazil",
    "France",
    "Spain",
    "Australia",
    "Netherlands",
    "Poland",
    "Italy",
    "Russian Federation",
    "Sweden",
)

education = (
    "Less than a Bachelors",
    "Bachelor's degree",
    "Master's degree",
    "Post grad",
)

# Create new label encoders with specified categories
new_le_country = LabelEncoder()
new_le_education = LabelEncoder()
new_le_country.fit(countries)
new_le_education.fit(education)

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need some information to predict the salary""")

    country = st.selectbox("Country", countries)
    education_level = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education_level, experience]])
        X[:, 0] = new_le_country.transform(X[:,0])
        X[:, 1] = new_le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")