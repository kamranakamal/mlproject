import streamlit as st
import requests

gender= st.selectbox("Gender",["male","female"])
race_ethnicity = st.selectbox(
    "Race/Ethnicity",
    ['group B' ,'group C', 'group A' ,'group D', 'group E'])

parental_level_of_education=st.selectbox(
    "Parents Level of Education",
    ["bachelor's degree", "some college", "master's degree", "associate's degree",
"high school", "some high school"])

lunch=st.selectbox("Lunch",['standard', 'free/reduced'])
test_preparation_course = st.selectbox("Test Preparation Course",['none', 'completed'])
reading_score = st.slider("Reading Score",0,100)
writing_score = st.slider("Writing Score",0,100)


data = {
    "gender": gender,
    "race_ethnicity": race_ethnicity,
    "parental_level_of_education": parental_level_of_education,
    "lunch": lunch,
    "test_preparation_course": test_preparation_course,
    "reading_score": reading_score,
    "writing_score": writing_score
}
if st.button("predict"):
    response = requests.post('http://127.0.0.1:8000/predict',json=data)
    output = response.json()['prediction']
    if output>100:
        output = 100
    st.success(f"Predicted Math score  is {output}")
    