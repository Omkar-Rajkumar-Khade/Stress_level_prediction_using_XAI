## Stress Level Prediction Using Explainble AI

This software predicts stress levels based on user-provided input data. It uses a machine learning model trained on a dataset that includes factors such as age, sleep duration, quality of sleep, physical activity level, heart rate, daily steps, gender, occupation, and BMI category. The software utilizes a Random Forest Regressor model for predictions and provides explanations for model predictions using both SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-Agnostic Explanations) techniques.

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

5) Use the sliders and radio buttons to input your information.

6) Click the "Predict" button to see your predicted stress level.