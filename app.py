import streamlit as st
import time

st.set_page_config(layout="wide", page_title="Sleep Analysis")

st.markdown("""
<style>
    /* Réduit le padding par défaut en haut de la page */
    .stMainBlockContainer {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Optionnel : Ajuste la hauteur des tabs pour qu'ils soient plus propres */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Test", "Conversion", "Charts"])

with tab1:
    st.header("Sleep Data Entry Form")

    with st.form(key='sleep_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
        with col2:
            age = st.number_input("Age", min_value=20, max_value=60, value=40)
        with col3:
            occupation = st.selectbox("Occupation",
                                      ['Doctor', 'Teacher', 'Software Engineer', 'Lawyer', 'Engineer', 'Accountant',
                                       'Nurse', 'Scientist', 'Manager', 'Salesperson', 'Sales Representative'])

        col1, col2 = st.columns(2)
        with col1:
            sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.0)
        with col2:
            sleep_quality = st.slider("Sleep Quality (1 to 10)", min_value=1, max_value=10, value=7)

        col1, col2 = st.columns(2)
        with col1:
            physical_activity = st.slider("Physical Activity Level (1 to 5)", min_value=1, max_value=5, value=3)
        with col2:
            stress_level = st.slider("Stress Level (1 to 5)", min_value=1, max_value=5, value=2)

        col1, col2 = st.columns(2)
        with col1:
            bmi_category = st.selectbox("BMI Category", ['Normal Weight', 'Normal', 'Overweight', 'Obese'])
        with col2:
            blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)")

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
            "Daily Steps": daily_steps,
        })


def handle_user_message(message: str) -> str:
    time.sleep(0.5)
    return f"**Gemina :** J'ai bien reçu : '{message}'. Comment puis-je approfondir ?"


with tab2:
    st.header("Conversation")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    chat_container = st.container(height=450, border=True)

    with chat_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_input = st.chat_input(placeholder="Comment peut-on vous aider ?", key="user_message")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        response = handle_user_message(user_input)
        st.session_state["messages"].append({"role": "assistant", "content": response})

        st.rerun()

with tab3:
    st.header("Charts")
    st.info("Graphiques à venir...")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Répartition du sommeil")
        st.bar_chart({"Data": [1, 2, 3]})
    with col2:
        st.markdown("### Niveau de stress")
        st.line_chart({"Data": [10, 20, 15]})
