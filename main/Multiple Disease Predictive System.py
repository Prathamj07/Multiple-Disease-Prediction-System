import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import openai
import sqlite3
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Disease Prediction System",
                   layout="wide",
                   page_icon="🧑‍⚕️")

diabetes_model = pickle.load(open("D:/Study/Mini Project/Main/main/saved models/diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("D:/Study/Mini Project/Main/main/saved models/heart_disease_model.sav", 'rb'))
parkinsons_model = pickle.load(open("D:/Study/Mini Project/Main/main/saved models/parkinsons_model.sav", 'rb'))
svc = pickle.load(open("D:/Study/Mini Project/Main/main/saved models/rf.pkl", 'rb'))  

precaution = pd.read_csv("D:/Study/Mini Project/Main/dataset/precautions_df.csv")
workout = pd.read_csv("D:/Study/Mini Project/Main/dataset/workout_df.csv")
medication = pd.read_csv('D:/Study/Mini Project/Main/dataset/medications.csv')
diets = pd.read_csv('D:/Study/Mini Project/Main/dataset/diets.csv')
description = pd.read_csv("D:/Study/Mini Project/Main/dataset/description.csv")

db_dir = 'D:/Study/Mini Project/Main/database/'
db_path = os.path.join(db_dir, 'disease_predictions.db')


# Check if the directory exists, if not, create it
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

# Now connect to the database
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        disease_type TEXT,
        user_input TEXT,
        prediction_result TEXT
    )
''')

# Function to store input and prediction result in the database
def store_prediction(disease_type, user_input, prediction_result):
    user_input_str = ', '.join([str(x) for x in user_input])
    c.execute('''
        INSERT INTO predictions (disease_type, user_input, prediction_result) 
        VALUES (?, ?, ?)
    ''', (disease_type, user_input_str, prediction_result))
    conn.commit()

# Function for symptom-based disease prediction
def get_predicted_value(patient_symptoms):
    symptom_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
    diseases_list = {15: 'Fungal infection', 20: 'Hepatitis C', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 21: 'Hepatitis D', 19: 'Hepatitis B', 4: 'Allergy', 40: 'hepatitis A', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 27: 'Impetigo'}
    
    input_vector = np.zeros(len(symptom_dict))
    for item in patient_symptoms:
        input_vector[symptom_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

def helper(dis):
    print(f"Fetching details for disease: {dis}")
    
    descr = description[description['Disease'] == dis]['Description']
    descr = " ".join({ w for w in descr})
    print(f"Description: {descr}")
    
    pre = precaution[precaution['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    print(f"Precautions: {pre}")
    
    die = diets[diets['Disease'] == dis]['Diet']
    die = [diet for diet in die.values]
    print(f"Diet: {die}")
    
    work = workout[workout['disease'] == dis]['workout']
    print(f"Workouts: {work}")
    
    med = medication[medication['Disease'] == dis]['Medication']
    med = [med for med in med.values]
    
    if len(med) == 0:
        med = ["No specific medication available. Consult a doctor."]
    print(f"Medications: {med}")
    
    return descr, pre, die, med, work



# Side menu
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Helper','Dr. Friend'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'capsule'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
            
        store_prediction('Diabetes', user_input, diab_diagnosis)
                
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex ( 1 for Male and 0 for Female')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
        store_prediction('Heart Disease', user_input, heart_diagnosis)
    st.success(heart_diagnosis)
# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

   
    parkinsons_diagnosis = ''

   
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
        store_prediction("Parkinson's", user_input, parkinsons_diagnosis)
    st.success(parkinsons_diagnosis)

# Symptom-based Disease Prediction Page
if selected == "Helper":
    st.title("Helper")

    # User input: symptoms
    symptoms_input = st.text_input('Enter Your Symptoms (separated by commas)')

    # Adding an "Enter" button to trigger the prediction
    if st.button('Enter'):
        if symptoms_input:
            user_symptoms = [s.strip() for s in symptoms_input.split(',')]

            # Get the predicted disease
            predicted_disease = get_predicted_value(user_symptoms)

            # Get recommendation details
            descr, pre, die, med, work = helper(predicted_disease)

            # Display the results
            st.subheader('Predicted Disease')
            st.write(predicted_disease)

            st.subheader('Description')
            st.write(descr)

            st.subheader('Precautions')
            for i, precaution in enumerate(pre[0], start=1):
                st.write(f"{i}: {precaution}")

            st.subheader('Diets')
            for diet in die:
                st.write(diet)

            st.subheader('Medications')
            for i, medication in enumerate(med, start=1):
                st.write(f"{i}: {medication}")

            st.subheader('Recommended Workouts')
            for i, workout in enumerate(work, start=1):
                st.write(f"{i}: {workout}")



openai.api_key = "Enter the API Key"

import asyncio



def get_health_assistant_response(user_query):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a health assistant that provides general health information and advice."},
                {"role": "user", "content": user_query}
            ],
            max_tokens=100,  # Limit the length of the response
        )
        return completion['choices'][0]['message']['content']
    except Exception as e:
        return "Sorry, I couldn't process that request. Please try again later."

if selected == "Dr. Friend":
    st.title("Dr. Friend: Your Health Assistant Chatbot")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    for role, content in st.session_state.chat_history:
        if role == "You":
            st.markdown(
                f"<div style='text-align: right; background-color: #CFE2FF; padding: 10px; border-radius: 15px; margin-bottom: 5px; display: inline-block; max-width: 70%;'>"
                f"<strong>You:</strong> {content}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='text-align: left; background-color: #D1E7DD; padding: 10px; border-radius: 15px; margin-bottom: 5px; display: inline-block; max-width: 70%;'>"
                f"<strong>Health Assistant:</strong> {content}</div>",
                unsafe_allow_html=True
            )

    # Create input box
    user_input = st.text_input("You:", "")

    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(("You", user_input))
            with st.spinner("Generating response..."):
                assistant_response = get_health_assistant_response(user_input)
            st.session_state.chat_history.append(("Health Assistant", assistant_response))
            st.session_state.user_input = ""  # Reset not necessary, but here for clarity
            st.text_input("You:", "", key="input_box")  
