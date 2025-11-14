import streamlit as st

# Create tabs
tab1, tab2 = st.tabs(["Test", "Charts"])

with tab1:
    st.header("Sleep Data Entry Form")

    with st.form(key='sleep_form'):

        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
        with col2:
            age = st.number_input("Age", min_value=20, max_value=60, value=40)
        with col3:
            occupation = st.selectbox("Occupation", ['Doctor', 'Teacher', 'Software Engineer', 'Lawyer', 'Engineer','Accountant', 'Nurse', 'Scientist', 'Manager', 'Salesperson','Sales Representative']);


        # Sleep duration and quality
        col1, col2 = st.columns(2)
        with col1:
            sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0)
        with col2:
            sleep_quality = st.slider("Sleep Quality (1 to 10)", min_value=1, max_value=10, value=7)

        # Physical activity and stress
        col1, col2 = st.columns(2)
        with col1:
            physical_activity = st.slider("Physical Activity Level (1 to 5)", min_value=1, max_value=5, value=3)
        with col2:
            stress_level = st.slider("Stress Level (1 to 5)", min_value=1, max_value=5, value=2)

        # BMI and blood pressure
        col1, col2 = st.columns(2)
        with col1:
            bmi_category = st.selectbox("BMI Category", ['Normal Weight', 'Normal', 'Overweight', 'Obese'])
        with col2:
            blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)")

        # Heart rate and daily steps
        col1, col2 = st.columns(2)
        with col1:
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=70)
        with col2:
            daily_steps = st.number_input("Daily Steps", min_value=0, max_value=100000, value=5000)

        submit_button = st.form_submit_button(label="Run Analysis")

    if submit_button:
        st.success("Data received!")

        st.write({
            "Gender": gender,
            "Age": age,
            "Occupation": occupation,
            "Sleep Duration": sleep_duration,
            "Sleep Quality": sleep_quality,
            "Physical Activity Level": physical_activity,
            "Stress Level": stress_level,
            "BMI Category": bmi_category,
            "Blood Pressure": blood_pressure,
            "Heart Rate": heart_rate,
            "Daily Steps": daily_steps,
        })

        # TODO Faire analyse et afficher résultats