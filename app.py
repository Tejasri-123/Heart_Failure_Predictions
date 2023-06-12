

import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



st.markdown("<h6 style='text-align:center;color:RosyBrown;'>Here we are going to predict whether a person has heart failure or not</h6>", unsafe_allow_html=True)

# Load the trained model
with open('random_forest.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

with open('logr.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)

with open('decision_tree.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

# Function to predict heart failure
def predict_random_forest(data):
    prediction = random_forest_model.predict(data)
    return prediction

def predict_logistic_regression(data):
    prediction = logistic_regression_model.predict(data)
    return prediction

def predict_decision_tree(data):
    prediction = decision_tree_model.predict(data)
    return prediction

data = pd.read_csv('heart_failure (1).csv')

# Set up the Streamlit app
def home():
    st.markdown("<h1 style='text-align:center;color:Red;'>Heart Failure Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='text-align:center;color:Violet;'>Welcome to the Heart Failure Prediction System!</h4>", unsafe_allow_html=True)
    col1,col2 = st.columns([1, 1])
    with col2:
        image = Image.open('images.jfif')
        st.image(image , caption='Know the health of your heart')

    with col1:
        st.subheader("Let's Learn about the some Facts about Heart . . .")
        st.write("* Everyday your heart beats 100000 times ")
        st.write("* Each minute your heart pumps 5 liters of blood")
        # st.write("* Happiness helps to lower risk of heart disease")
    
    st.markdown("<h3 style='text-align:center;color:Blue;'>General Statistics</h3>", unsafe_allow_html=True)

    gender_counts = data['sex'].value_counts()

# Filter the data for men and women separately
    men_data = data[data['sex'] == 1]
    women_data = data[data['sex'] == 0]

# Count the number of men and women with heart failure
    men_heart_failure_counts = men_data['DEATH_EVENT'].value_counts()
    women_heart_failure_counts = women_data['DEATH_EVENT'].value_counts()

# Calculate heart failure rates for men and women
    men_total = gender_counts[1]
    women_total = gender_counts[0]
    men_heart_failure_rate = men_heart_failure_counts[1] / men_total
    women_heart_failure_rate = women_heart_failure_counts[1] / women_total

# Create a bar chart for men and women
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as desired
    bar_width = 0.55  # Adjust the bar width as desired
    # opacity = 0.8  # Adjust the opacity as desired
    ax.bar(['No Heart Failure', 'Heart Failure'], men_heart_failure_counts, label='Men',
       color='lightblue', width=bar_width)
    ax.bar(['No Heart Failure', 'Heart Failure'], women_heart_failure_counts, label='Women',
       color='lightgreen',width=bar_width, bottom=men_heart_failure_counts)
    ax.set_xlabel('Heart Failure')
    ax.set_ylabel('Count')
    ax.set_title('Heart Failure Count by Gender')
    ax.legend()

    # Adjust the font size of x-axis and y-axis labels
    ax.tick_params(axis='both', labelsize=10)  # Adjust the font size as desired

    st.pyplot(fig)

# Display the heart failure rates for men and women
    st.write(f"Heart Failure Rate for Men: {men_heart_failure_rate:.2%}")
    st.write(f"Heart Failure Rate for Women: {women_heart_failure_rate:.2%}")

    st.markdown("<h5 style='color:CornflowerBlue;'>If you to know about your heart condition go to input page 🎯 and fill your details. . .</h5>", unsafe_allow_html=True)
    
def input_page():
    st.markdown("<h1 style='text-align:center;color:Red;'>Heart Failure Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;color:Teal;'>Let me Know your details</h1>", unsafe_allow_html=True)
     
    col1,col2 = st.columns([1, 1])

    # Input fields for the user to enter the required data
    with col1:
        
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        anaemia = st.selectbox('Anaemia', ['No', 'Yes'])
        creatinine_phosphokinase=st.number_input('Creatinine_Phosphokinase',min_value=0,max_value = 20000)
        diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
        ejection_fraction = st.number_input('Ejection Fraction', min_value=0, max_value=100, value=50)
        high_blood_pressure = st.selectbox('High Blood Pressure', ['No', 'Yes'])

    with col2:
        
        platelets = st.number_input('Platelets', min_value=0, value=200000)
        serum_creatinine = st.number_input('Serum Creatinine', min_value=0.0, value=1.0)
        serum_sodium = st.number_input('Serum Sodium', min_value=0, value=140)
        sex = st.selectbox('Sex', ['Female', 'Male'])
        time = st.number_input('Time',min_value=0,max_value=500)
        smoking = st.selectbox('Smoking', ['No', 'Yes'])

    # Create a dictionary with the user input
    data = {
        'age': age,
        'anaemia': 1 if anaemia == 'Yes' else 0,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': 1 if diabetes == 'Yes' else 0,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': 1 if high_blood_pressure == 'Yes' else 0,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': 1 if sex == 'Male' else 0,
        'smoking': 1 if smoking == 'Yes' else 0,
        'time':time
    }

    # Convert the data to a DataFrame
    input_df = pd.DataFrame([data])

    random_forest_prediction = predict_random_forest(input_df)
    logistic_regression_prediction = predict_logistic_regression(input_df)
    decision_tree_prediction = predict_decision_tree(input_df)

    

    # st.subheader('Model Accuracy')
    # st.write(f'Random Forest Model Accuracy: {91.6318:.2f}')
    # st.write(f'Logistic Regression Model Accuracy: {79.4979:.2f}')
    # st.write(f'Decision Tree Model Accuracy: {92.887:.2f}')

    st.subheader('Prediction')
    st.subheader('Model Information')
    model_names = ['Random Forest', 'Logistic Regression', 'Decision Tree']
    model_distribution = [0.4, 0.3, 0.3]
    fig, ax = plt.subplots()
    ax.pie(model_distribution, labels=model_names, autopct='%1.1f%%')
    ax.set_title('Model Distribution')
    st.pyplot(fig)
    st.markdown("<h3 style='color:DarkKhaki;'>Select the model you like. . .</h3>", unsafe_allow_html=True)  

    model_selection = st.selectbox('Select Model', ['Random Forest Model', 'Logistic Regression Model', 'Decision Tree Model'])
    if model_selection == 'Random Forest Model':
        prediction = random_forest_prediction
        accuracy = 91.6318

    elif model_selection == 'Logistic Regression Model':
        prediction = logistic_regression_prediction
        accuracy = 79.4979

    else:
        prediction = decision_tree_prediction
        accuracy = 92.887
    st.markdown("<h5 style='color:MediumOrchid;'>Click on predict button to predict the result👆</h3>", unsafe_allow_html=True)  

    if st.button('Predict'):
        st.session_state['prediction'] = prediction
        st.session_state['accuracy'] = accuracy
        st.session_state['model_selection'] = model_selection

    st.markdown("<h6 style='color:LightSalmon;'>If you want to know your result navigate to result page🎯</h6>", unsafe_allow_html=True)  


def result_page():
    st.title('Heart Failure Prediction - Result')
    prediction = st.session_state.get('prediction')
    accuracy = st.session_state.get('accuracy')
    model_selection = st.session_state.get('model_selection')

    if prediction is not None:
        # Display the prediction
        if prediction[0] == 1:
            st.error('Sorry 😥😔 Heart failure is predicted.')
        else:
            st.success('Congratulations🎉🎇 Heart failure is not predicted.')

        # Display the prediction accuracy
        st.write(f'Prediction Accuracy for {model_selection} : {accuracy:.2f}')

    X = data.drop("DEATH_EVENT", axis=1) # Specify the column(s) representing your features
    y = data["DEATH_EVENT"] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    def compute_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return cm

    if model_selection == 'Random Forest Model':
        selected_model_predictions = predict_random_forest(X_test)
    elif model_selection == 'Logistic Regression Model':
        selected_model_predictions = predict_logistic_regression(X_test)
    else:
        selected_model_predictions = predict_decision_tree(X_test)

    selected_model_cm = compute_confusion_matrix(y_test, selected_model_predictions)

    st.subheader(f'Confusion Matrix for {model_selection}')
    # st.dataframe(pd.DataFrame(selected_model_cm, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive']))
    plt.figure(figsize=(6, 4))
    sns.heatmap(selected_model_cm, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)
    
    st.markdown("<h6 style='color:Red;'>Disclaimer : These are just predictions only</h6>", unsafe_allow_html=True)
    
    

    

def main():
    # st.sidebar.title('Navigation')
    
    # page = st.sidebar.radio('Go to', ['Home', 'Input', 'Result'])

    # if page == 'Home':
    #     home()
    # elif page == 'Input':
    #     input_page()
    # elif page == 'Result':
    #     result_page()

    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        home()
        if st.button('Go to Input Page'):
            st.session_state.page = 'input'
    elif st.session_state.page == 'input':
        input_page()
        if st.button('Go to Result Page'):
            st.session_state.page = 'result'
    elif st.session_state.page == 'result':
        result_page()
        if st.button('Go to Home Page'):
            st.session_state.page = 'home'

if __name__ == '__main__':
    main()
