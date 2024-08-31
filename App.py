import pandas as pd
import joblib
import streamlit as st
import sklearn


st.title("Insurance Preminum Prediction")

model = joblib.load('reg_model.joblib')
enc = joblib.load('encoder.joblib')
le_sex = joblib.load('le_sex.joblib')
le_smk = joblib.load('le_smk.joblib')

age = st.number_input('Enter Age',min_value=18)

Sex = st.selectbox('Sex',['male','female'])

bmi = st.number_input('Enter BMI',min_value=20)

children = st.number_input('Number of Kids',min_value=0)

smoker = st.selectbox('Smoker?',['yes','no'])

region = st.selectbox('Region',['northeast', 'northwest', 'southeast', 'southwest'])

data = {"age":age,
        "sex":Sex,
        "bmi":bmi,
        "children":children,
        "smoker":smoker,
        "region":region}

df = pd.DataFrame(data, index=[0])

one_hot = enc.transform(df[['region']]).toarray()
df[["northeast","northwest","southeast","southwest"]] = one_hot
df['sex'] = le_sex.transform(df[['sex']])
df['smoker'] = le_smk.transform(df[['smoker']])
df = df.drop(columns='region')

Buttton = st.button('Predict')

if Buttton == True:
    Prediction = model.predict(df.head(1))
    if Prediction < 0:
        st.info(f'${0:0,.2f}')
    else:
        st.info(f'${Prediction[0]:0,.2f}')
