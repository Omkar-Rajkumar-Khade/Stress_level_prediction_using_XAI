import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

# Load the saved RandomForestRegressor model
with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

data=pd.read_csv("Dataset\Sleep_health_and_lifestyle_dataset.csv")
data=data.drop(["Blood Pressure","Sleep Disorder","Person ID"],axis=1)
# Specify the categorical columns to be one-hot encoded
categorical_columns = ['Gender', 'Occupation', 'BMI Category']
# Use get_dummies to one-hot encode the categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_columns)
replace_dict = {True: 1, False: 0}
data_encoded= data_encoded.replace(replace_dict)
X = data_encoded.drop(columns=['Stress Level'])
y = data_encoded['Stress Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get the list of feature names
feature_names = X_train.columns.tolist()

# Disable the Matplotlib global figure object warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define the input fields
st.title("Stress Level Prediction")
st.write("Enter the following information:")

# Age
age = st.slider("Age", 18, 100, 30)

# Sleep Duration
sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 24.0, 7.0)

# Quality of Sleep
quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 5)

# Physical Activity Level
physical_activity_level = st.slider("Physical Activity Level (1-100)", 1, 100, 50)

# Heart Rate
heart_rate = st.slider("Heart Rate", 40, 200, 75)

# Daily Steps
daily_steps = st.slider("Daily Steps", 0, 25000, 5000)

# Gender
gender = st.radio("Gender", ("Female", "Male"))

# Occupation
occupation = st.radio("Occupation", ("Accountant", "Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Sales Representative",
                                     "Salesperson", "Scientist", "Software Engineer", "Teacher"))

# BMI Category
bmi_category = st.radio("BMI Category", ("Normal", "Normal Weight", "Obese", "Overweight"))

# Encode categorical variables based on user input
gender_encoded = 1 if gender == "Female" else 0
occupation_encoded = occupation
bmi_encoded = bmi_category

# Create a DataFrame from the user input
user_input_df = pd.DataFrame({
    'Age': [age],
    'Sleep Duration': [sleep_duration],
    'Quality of Sleep': [quality_of_sleep],
    'Physical Activity Level': [physical_activity_level],
    'Heart Rate': [heart_rate],
    'Daily Steps': [daily_steps],
    'Gender_Female': [gender_encoded],
    'Gender_Male': [1 - gender_encoded],
    'Occupation_Accountant': [1 if occupation_encoded == "Accountant" else 0],
    'Occupation_Doctor': [1 if occupation_encoded == "Doctor" else 0],
    'Occupation_Engineer': [1 if occupation_encoded == "Engineer" else 0],
    'Occupation_Lawyer': [1 if occupation_encoded == "Lawyer" else 0],
    'Occupation_Manager': [1 if occupation_encoded == "Manager" else 0],
    'Occupation_Nurse': [1 if occupation_encoded == "Nurse" else 0],
    'Occupation_Sales Representative': [1 if occupation_encoded == "Sales Representative" else 0],
    'Occupation_Salesperson': [1 if occupation_encoded == "Salesperson" else 0],
    'Occupation_Scientist': [1 if occupation_encoded == "Scientist" else 0],
    'Occupation_Software Engineer': [1 if occupation_encoded == "Software Engineer" else 0],
    'Occupation_Teacher': [1 if occupation_encoded == "Teacher" else 0],
    'BMI Category_Normal': [1 if bmi_encoded == "Normal" else 0],
    'BMI Category_Normal Weight': [1 if bmi_encoded == "Normal Weight" else 0],
    'BMI Category_Obese': [1 if bmi_encoded == "Obese" else 0],
    'BMI Category_Overweight': [1 if bmi_encoded == "Overweight" else 0]
})

# Add a "Predict" button
if st.button("Predict"):
    # Make predictions
    predicted_stress_level = model.predict(user_input_df)

    # Display the prediction with a larger font size using HTML tags
    st.markdown(f"<h2>Predicted Stress Level: {predicted_stress_level[0]}</h2>", unsafe_allow_html=True)

   # Step 3: Use SHAP for model interpretation
    explainer_shap = shap.Explainer(model, X_train)
    shap_values = explainer_shap.shap_values(user_input_df)

    # Display the SHAP summary plot using st.pyplot()
    st.write("SHAP Summary Plot")
    fig_summary_shap, ax_summary_shap = plt.subplots()
    shap.summary_plot(shap_values, user_input_df, plot_type="bar", show=False)
    st.pyplot(fig_summary_shap)  # Display the Matplotlib plot using st.pyplot()
    # # Create the SHAP force plot using st.pyplot()
    # st.write("SHAP Force Plot")
    # fig_force_shap, ax_force_shap = plt.subplots()
    # shap.force_plot(explainer_shap.expected_value, shap_values[0], user_input_df, show=False)
    # st.pyplot(fig_force_shap)  # Display the Matplotlib plot using st.pyplot()

      
    # Step 4: Use LIME for model interpretation
    explainer_lime = LimeTabularExplainer(X_train.values, mode="regression", training_labels=y_train, feature_names=feature_names)

    # Generate LIME explanations for the Random Forest model
    lime_explanation_rf = explainer_lime.explain_instance(user_input_df.values[0], model.predict)

    # Get LIME explanation data
    lime_data = lime_explanation_rf.as_list()

    # Extract feature names and contribution values
    feature_names_lime = [item[0] for item in lime_data]
    contributions = [item[1] for item in lime_data]

    # Create a bar chart to visualize feature contributions
    st.write("LIME Explanation (Graphical)")
    fig_lime, ax_lime = plt.subplots()
    ax_lime.barh(feature_names_lime, contributions)
    st.pyplot(fig_lime)  # Display the Matplotlib plot using st.pyplot()







