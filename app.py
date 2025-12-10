import streamlit as st
import time
import os
from google import genai
import json
import dotenv

dotenv.load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
generation_config = {
    "temperature": 0.2,
    "response_mime_type": "application/json"
}

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

MODEL_TO_USE = "gemini-flash-latest"

analysis_prompt = """
Agis comme un assistant virtuel expert en hygiène de sommeil. En t'appuyant sur les données du fichier 'Sleep_Data_Sampled.csv', analyse les liens entre le style de vie (niveau de stress, IMC, activité physique) et le type de trouble du sommeil identifié dans la colonne 'Sleep Disorder'.

Si un utilisateur présente un trouble spécifique comme l'insomnie ou l'apnée du sommeil, fournis des recommandations factuelles et pratiques basées sur les tendances observées dans les données ou sur des principes de santé générale reconnus. Concentre-toi sur des ajustements simples du quotidien.

Adopte un ton professionnel, concis et serviable, typique d'un bot d'aide en ligne. Évite le jargon médical complexe. Ne fais aucune liste à puces ; structure ta réponse uniquement sous forme de paragraphes fluides et cohérents.
"""

chat_prompt = """
Tu es un assistant IA spécialisé dans la collecte et la structuration de données médicales pour l'analyse du sommeil. **Ton rôle est celui d'un clinicien ou d'un chercheur expert, visant à rendre le processus de collecte d'informations sur le sommeil aussi agréable et rapide que possible.** Ton objectif est de dialoguer avec l'utilisateur pour extraire des informations spécifiques correspondant aux colonnes d'un dataset cible (Sleep_Data_Sampled.csv).

**Tu dois faire preuve d'empathie, de courtoisie et d'une capacité à inférer et à regrouper les questions de manière logique et proactive.**

À chaque interaction avec l'utilisateur, tu dois analyser son message et renvoyer uniquement un objet JSON valide (sans Markdown, sans texte avant ou après).

Voici les champs que tu dois extraire et valider :

- Gender : Male ou Female.
- Age : Entier.
- Occupation : String (ex: Engineer, Doctor, etc.).
- Sleep Duration : Float (heures/nuit).
- Quality of Sleep : Entier (échelle 1-10).
- Physical Activity Level : Entier (minutes/jour ou score 0-100).
- Stress Level : Entier (échelle 1-10).
- BMI Category : String (Normal, Overweight, Obese). Si l'utilisateur donne poids/taille, calcule-le.
- Blood Pressure : String (format Sys/Dia, ex: 120/80).
- Heart Rate : Entier (bpm).
- Daily Steps : Entier.

Pour chaque réponse de ta part, le JSON doit respecter la structure suivante :
```json
{
  "user_interaction": {
    "message_to_user": "Ici, tu poses une question polie en français pour obtenir les données manquantes, ou tu confirmes la fin de la collecte. **Tu dois prioriser les questions thématiques et logiques, ne demandant qu'un ou deux groupes d'informations à la fois (ex: toutes les informations liées au sport, ou toutes les informations physiologiques).**",
    "missing_fields": ["Liste", "des", "champs", "restants"]
  },
  "data_extraction": {
    "Gender": null,
    "Age": null,
    "Occupation": null,
    "Sleep Duration": null,
    "Quality of Sleep": null,
    "Physical Activity Level": null,
    "Stress Level": null,
    "BMI Category": null,
    "Blood Pressure": null,
    "Heart Rate": null,
    "Daily Steps": null
  },
  "metadata": {
    "validity_check": {
      "is_valid": true,
      "errors": []
    },
    "confidence_score": 0.0,
    "ready_for_analysis": false
  }
}

Règles de comportement avancées :
- Extraction : Si l'utilisateur dit "Je suis un homme de 43 ans", remplis Gender: "Male" et Age: 43.
- Inférence & Proactivité (Unités): Si une valeur est ambiguë (ex: "Je fais 30 d'activité"), tu dois inférer l'unité la plus probable (minutes/jour pour l'activité physique) ou demander une clarification polie. N'accepte pas de null si un nombre est donné sans unité.
- Inférence & Proactivité (IMC): Si l'utilisateur donne son poids (en kg) et sa taille (en m ou cm), tu DOIS calculer l'IMC (poids/taille2) et remplir le champ "BMI Category" (Normal: <25, Overweight: 25−30, Obese: ≥30). C'est une obligation, pas une option.
- Regroupement Thématique (Priorité): Lors de la demande d'informations manquantes ("message_to_user"), regroupe les questions. Exemple de thèmes : 1. Démographie (Genre/Âge/Métier), 2. Sommeil et Stress (Durée/Qualité/Stress), 3. Santé Physique (Activité/Pas/FC/PA). Commence toujours par le groupe Démographie.
- Coefficient de certitude (confidence_score) : Calcule un score de 0.0 à 1.0 basé sur le pourcentage de champs remplis et la cohérence des données (ex: un âge de 200 ans est invalide).
- Prêt pour analyse (ready_for_analysis) : Passe ce booléen à true uniquement si le confidence_score est supérieur à 0.9 et que tout les champs sont remplis (par l'utilisateur ou de manière automatique).
- Langue et Ton : Le champ message_to_user doit toujours être en français, courtois, empathique et orienté vers la facilitation (ex: "Passons maintenant aux chiffres de votre activité physique...").

Commence l'analyse dès le premier message de l'utilisateur.
"""

# Initialiser les états de session
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = None


def call_gemini_chat(user_message: str) -> dict:
    try:
        gemini_messages = []

        for msg in st.session_state["chat_history"]:
            gemini_messages.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })

        gemini_messages.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        response = client.models.generate_content(
            model=MODEL_TO_USE,
            contents=gemini_messages,
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
                "system_instruction": chat_prompt
            }
        )

        response_text = response.text
        data = json.loads(response_text)

        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_message
        })
        st.session_state["chat_history"].append({
            "role": "model",
            "content": response_text
        })

        if "data_extraction" in data:
            st.session_state["extracted_data"] = data["data_extraction"]

        return data

    except Exception as e:
        print(e)
        st.error(f"Erreur lors de l'appel à Gemini: {str(e)}")
        return {
            "user_interaction": {
                "message_to_user": "Désolé, une erreur s'est produite. Pouvez-vous reformuler votre message ?",
                "missing_fields": []
            },
            "data_extraction": {},
            "metadata": {
                "validity_check": {"is_valid": False, "errors": [str(e)]},
                "confidence_score": 0.0,
                "ready_for_analysis": False
            }
        }


def call_gemini_analysis(user_data: dict) -> str:
    try:
        data_text = f"""
Données de l'utilisateur :
- Genre : {user_data.get('Gender', 'Non spécifié')}
- Âge : {user_data.get('Age', 'Non spécifié')}
- Profession : {user_data.get('Occupation', 'Non spécifié')}
- Durée de sommeil : {user_data.get('Sleep Duration', 'Non spécifié')} heures
- Qualité du sommeil : {user_data.get('Quality of Sleep', 'Non spécifié')}/10
- Activité physique : {user_data.get('Physical Activity Level', 'Non spécifié')}
- Niveau de stress : {user_data.get('Stress Level', 'Non spécifié')}/10
- Catégorie IMC : {user_data.get('BMI Category', 'Non spécifié')}
- Pression artérielle : {user_data.get('Blood Pressure', 'Non spécifié')}
- Fréquence cardiaque : {user_data.get('Heart Rate', 'Non spécifié')} bpm
- Pas quotidiens : {user_data.get('Daily Steps', 'Non spécifié')}

Analyse ces données et fournis des recommandations personnalisées sur la qualité du sommeil.
"""

        response = client.models.generate_content(
            model=MODEL_TO_USE,
            contents=[{"role": "user", "parts": [{"text": data_text}]}],
            config={
                "temperature": 0.2,
                "system_instruction": analysis_prompt
            }
        )

        return response.text

    except Exception as e:
        st.error(f"Erreur lors de l'analyse: {str(e)}")
        return "Une erreur s'est produite lors de l'analyse de vos données."


tab1, tab2, tab3 = st.tabs(["Test", "Conversation", "Charts"])

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
        user_data = {
            "Gender": gender,
            "Age": age,
            "Occupation": occupation,
            "Sleep Duration": sleep_duration,
            "Quality of Sleep": sleep_quality,
            "Physical Activity Level": physical_activity,
            "Stress Level": stress_level,
            "BMI Category": bmi_category,
            "Blood Pressure": blood_pressure,
            "Heart Rate": heart_rate,
            "Daily Steps": daily_steps,
        }

        st.success("Data received!")

        with st.spinner("Analyse en cours..."):
            analysis_result = call_gemini_analysis(user_data)

        st.subheader("Analyse de votre sommeil")
        st.write(analysis_result)

with tab2:
    st.header("Conversation")

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

        with st.spinner("Réflexion en cours..."):
            response_data = call_gemini_chat(user_input)

        assistant_message = response_data.get("user_interaction", {}).get("message_to_user",
                                                                          "Désolé, je n'ai pas compris.")

        st.session_state["messages"].append({"role": "assistant", "content": assistant_message})

        if response_data.get("metadata", {}).get("ready_for_analysis", False):
            st.success("✅ Données complètes ! Lancement de l'analyse...")

            extracted_data = response_data.get("data_extraction", {})

            with st.spinner("Génération du rapport de sommeil..."):
                analysis_report = call_gemini_analysis(extracted_data)

            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"### 🌙 Analyse de votre sommeil\n\n{analysis_report}"
            })

            with st.expander("Voir les données techniques"):
                st.json(extracted_data)

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
