import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('mnb_spam_detector.pkl','rb'))

# Define the input features
input_features = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", 
                  "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
                  "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", 
                  "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", 
                  "Income"]

feature_labels = {
    "HighBP": "High Blood Pressure. 0 = no 1 = yes",
    "HighChol": "High Cholesterol. 0 = no no 1 = yes",
    "CholCheck": "Cholesterol Check in 5 Years 0 = no 1 = yes ",
    "BMI": "Body Mass Index",
    "Smoker": "Smoker. Have you smoked at least 100 cigarettes in your entire life? 0 = no 1 = yes",
    "Stroke": "Stroke. (Ever told) you had a stroke. 0 = no 1 = yes",
    "HeartDiseaseorAttack": "Coronary heart disease (CHD) or Myocardial infarction (MI) 0 = no 1 = yes",
    "PhysActivity": "Physical Activity. Physical Activity in past 30 days - not including job 0 = no 1 = yes",
    "Fruits": "Fruits. Consume Fruit 1 or more times per day 0 = no 1 = yes",
    "Veggies": "Vegetables. Consume Vegetables 1 or more times per day 0 = no 1 = yes",
    "HvyAlcoholConsump": "Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 0 = no 1 yes",
    "AnyHealthcare": "Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no 1 = yes",
    "NoDocbcCost": "Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no 1 = yes",
    "GenHlth": "General Health. Would you say that in general your health is 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor",
    "MentHlth": "Mental Health. Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? 0-30",
    "PhysHlth": "Physical Health. Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? 0-30",
    "DiffWalk": "Serious difficulty walking or climbing stairs means you cannot walk or have a lot of difficulty walking or climbing stairs. 0 = no 1 = yes",
    "Sex": "0 = Female 1 = Male",
    "Age": "13-level age category (_AGEG5YR see codebook) 1 = 18-24 9 = 60-64 13 = 80 or older", 
    "Education": "Highest level of education completed. 1 = never attended school or only kindergarten 2 = grades 1 through 8 (Elementary) 3 = grades 9 through 11 (Some high school) 4 = grade 12 or GED (High school graduate) 5 = some college or 2-year degree 6 = college graduate",
    "Income": "Annual household income. 1 = Less than $10,000 2 = $10,000 to less than $15,000 3 = $15,000 to less than $20,000 4 = $20,000 to less than $25,000 5 = $25,000 to less than $35,000 6 = $35,000 to less than $50,000 7 = $50,000 to less than $75,000 8 = $75,000 or more",
}


# Collect input values from the user
input_values = []
for feature in input_features:
    label = feature_labels.get(feature, feature)
    value = st.number_input(f"Enter {label}", step=0.01)
    input_values.append(value)

# Perform prediction when the user clicks the button
if st.button('Predict'):
    # Convert input_values to numpy array and reshape
    input_data = np.array(input_values).reshape(1, -1)
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction result
    if prediction == 0:
        st.header("Negative for Diabetes")
    elif prediction == 1:
        st.header("Prediabetes")
    else:
        st.header("Diabetes")