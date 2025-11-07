import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import streamlit as st
import pickle


with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

st.title("Customer Churn Prediction")

# Taking input from the User
st.sidebar.header("Enter Customer Details")

credit_score = st.sidebar.number_input('Credit Score')
geography = st.sidebar.selectbox('Geography', ["Germany", "Spain", "France"])
gender = st.sidebar.selectbox('Gender', ["Male", "Female"])
age = st.sidebar.slider('Age', 18, 92)
tenure = st.sidebar.slider('Tenure', 0, 10)
balance = st.sidebar.number_input('Balance')
num_of_products = st.sidebar.slider('Number of Products', 1, 4)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])
estimated_salary = st.sidebar.number_input('Estimated Salary')


# Convert into dataframe to feed the model
user_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


cat_cols = ['Geography', 'Gender']

# encode categorical col
encoded_cols = encoder.transform(user_data[cat_cols])
encoded_cols_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out())

# drop original cat columns + concat encoded
user_data = pd.concat([user_data.drop(columns=cat_cols), encoded_cols_df], axis=1)

# # scale
user_data = scaler.transform(user_data)

model = tf.keras.models.load_model("NN_model.h5")

if st.button("Predict"):
    prediction = model.predict(user_data)
    prediction_probability = prediction[0][0]

    st.write(f'Churn Probability: {prediction_probability:.3f}')

    if prediction_probability > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')