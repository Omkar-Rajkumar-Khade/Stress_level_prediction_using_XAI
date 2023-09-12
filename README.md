## Stress Level Prediction Using Explainble AI

This software predicts stress levels based on user-provided input data. It uses a machine learning model trained on a dataset that includes factors such as age, sleep duration, quality of sleep, physical activity level, heart rate, daily steps, gender, occupation, and BMI category. The software utilizes a Random Forest Regressor model for predictions and provides explanations for model predictions using both SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-Agnostic Explanations) techniques.

## Installation
To run this application, you need Python and several libraries, which can be installed using the following command:
```bash
pip install streamlit pandas scikit-learn shap lime matplotlib numpy
```
## Usage
To use the Stress Level Prediction app:

1) Clone this repository to your local machine.
```bash
git clone https://github.com/Omkar-Rajkumar-Khade/Stress_level_prediction_using_XAI.git
```
2) Navigate to the project directory.

3) Run the Streamlit app with the following command:
bash
```bash
streamlit run app.py
```
4) Open a web browser and go to the URL provided by Streamlit (usually http://localhost:8501).

5) You will see a user interface with input fields for various features.

    *Age*: Slide the age slider to input your age.
    *Sleep Duration (hours)*: Slide the sleep duration slider to input the number of hours you sleep per night.
    *Quality of Sleep (1-10)*: Slide the quality of sleep slider to rate your sleep quality on a scale of 1 to 10.
    *Physical Activity Level (1-100)*: Slide the physical activity level slider to specify your activity level.
    *Heart Rate*: Slide the heart rate slider to input your heart rate.
    *Daily Steps*: Slide the daily steps slider to specify the number of steps you take per day.
    *Gender*: Choose your gender (Female or Male) from the radio buttons.
    *Occupation*: Select your occupation from the available options.
    *BMI Category*: Choose your BMI category from the available options (Normal, Normal Weight, Obese, Overweight).
    After entering your information, click the "Predict" button.

6) Click the "Predict" button to see your predicted stress level.

7) The app will display the predicted stress level based on the provided data