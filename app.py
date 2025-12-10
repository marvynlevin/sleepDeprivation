import streamlit as st
import time
import os
from google import genai
import json
import dotenv
import joblib
import pandas as pd
import numpy as np


@st.cache_resource
def load_brain():
    try:
        return joblib.load('sleep_model_artifacts.pkl')
    except FileNotFoundError:
        return None

artifacts = load_brain()

# Fonction de prédiction nettoyée (sans debug)
def predict_sleep_disorder(user_data):
    if artifacts is None:
        return "Modèle introuvable (fichier .pkl manquant)"

    try:
        # 1. Gestion de la tension (ex: "120/80" -> 120, 80)
        bp = user_data.get('Blood Pressure', '120/80')
        if '/' in bp:
            systolic, diastolic = map(int, bp.split('/'))
        else:
            systolic, diastolic = 120, 80

        # 2. Création du DataFrame (L'ordre et les noms doivent être exacts)
        df_input = pd.DataFrame({
            'Age': [int(user_data.get('Age', 30))],
            'Gender': [user_data.get('Gender', 'Male')],
            'Occupation': [user_data.get('Occupation', 'Engineer')],
            'Sleep Duration': [float(user_data.get('Sleep Duration', 7.0))],
            'Quality of Sleep': [int(user_data.get('Quality of Sleep', 7))],
            'Physical Activity Level': [int(user_data.get('Physical Activity Level', 40))],
            'Stress Level': [int(user_data.get('Stress Level', 5))],
            'BMI Category': [user_data.get('BMI Category', 'Normal')],
            'Heart Rate': [int(user_data.get('Heart Rate', 70))],
            'Daily Steps': [int(user_data.get('Daily Steps', 5000))],
            'Systolic': [systolic],
            'Diastolic': [diastolic]
        })

        # 3. Prédiction
        pipeline = artifacts['model']
        le = artifacts['label_encoder']

        pred_code = pipeline.predict(df_input)
        pred_label = le.inverse_transform(pred_code)[0]

        return pred_label

    except Exception as e:
        return f"Erreur technique : {str(e)}"

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


# Mettez à jour cette fonction pour accepter l'argument 'ai_prediction'
def call_gemini_analysis(user_data, ai_prediction=None):
    try:
        # On intègre la prédiction du modèle dans le prompt
        prediction_text = f"Le modèle prédictif (XGBoost) a diagnostiqué : {ai_prediction}" if ai_prediction else "Le modèle prédictif n'a pas été exécuté."

        data_text = f"""
Données du patient : {user_data}

{prediction_text}

Consigne :
1. Prends en compte le diagnostic du modèle IA ci-dessus.
2. Explique ce résultat en te basant sur les données (Stress, IMC, Tension...).
3. Donne 3 recommandations concrètes.
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
        return f"Erreur lors de l'analyse : {str(e)}"


tab1, tab2, tab3 = st.tabs(["Test", "Conversation", "Charts"])

# TAB 1 : FORMULAIRE
with tab1:
    st.header("Formulaire de Diagnostic")
    with st.form(key='sleep_form'):
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ['Male', 'Female'])
        age = c2.number_input("Age", 20, 90, 40)
        occupation = c3.selectbox("Occupation", ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager'])

        c1, c2 = st.columns(2)
        sleep_duration = c1.number_input("Sleep Duration (h)", 0.0, 24.0, 7.0)
        sleep_quality = c2.slider("Sleep Quality (1-10)", 1, 10, 7)

        c1, c2 = st.columns(2)
        physical_activity = c1.slider("Physical Activity (min/jour)", 0, 120, 40)
        stress_level = c2.slider("Stress Level (1-10)", 1, 10, 5)

        c1, c2 = st.columns(2)
        bmi_category = c1.selectbox("BMI Category", ['Normal', 'Overweight', 'Obese'])
        blood_pressure = c2.text_input("Blood Pressure", "120/80")

        c1, c2 = st.columns(2)
        heart_rate = c1.number_input("Heart Rate (bpm)", 30, 200, 70)
        daily_steps = c2.number_input("Daily Steps", 0, 30000, 5000)

        submit_button = st.form_submit_button("Lancer l'Analyse")

    if submit_button:
        # Création du dictionnaire
        user_data = {
            "Gender": gender, "Age": age, "Occupation": occupation,
            "Sleep Duration": sleep_duration, "Quality of Sleep": sleep_quality,
            "Physical Activity Level": physical_activity, "Stress Level": stress_level,
            "BMI Category": bmi_category, "Blood Pressure": blood_pressure,
            "Heart Rate": heart_rate, "Daily Steps": daily_steps
        }

        with st.spinner("Analyse IA en cours..."):
            # 1. Prédiction Technique
            pred_ia = predict_sleep_disorder(user_data)

            # 2. Analyse Gemini
            report = call_gemini_analysis(user_data, pred_ia)

        st.divider()
        c_res, c_txt = st.columns([1, 2])

        with c_res:
            st.subheader("Diagnostic IA")

            # On vérifie "None" OU "Healthy" (selon ce que votre modèle a appris)
            if pred_ia == "None" or pred_ia == "Healthy":
                st.success("✅ **SAIN** (Healthy)")
            elif pred_ia == "Insomnia":
                st.warning("⚠️ **INSOMNIE**")
            elif pred_ia == "Sleep Apnea":
                st.error("🚨 **APNÉE**")
            else:
                st.info(f"Résultat : {pred_ia}")

        with c_txt:
            st.markdown(report)

with tab2:
    st.header("Conversation avec l'Assistant")
    st.caption("L'IA va vous poser des questions pour établir votre profil, puis utilisera le modèle de Machine Learning pour le diagnostic.")

    chat_container = st.container(height=450, border=True)

    # 1. Affichage de l'historique
    with chat_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 2. Zone de saisie utilisateur
    user_input = st.chat_input(placeholder="Ex: Bonjour, j'ai 30 ans et je dors mal...", key="user_message_chat")

    if user_input:
        # Affiche le message de l'utilisateur
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        # 3. Appel à Gemini pour extraire les données (Conversation)
        with st.spinner("L'assistant réfléchit..."):
            response_data = call_gemini_chat(user_input)

        # Récupération de la réponse JSON
        assistant_message = response_data.get("user_interaction", {}).get("message_to_user", "Je n'ai pas compris.")
        new_extracted_data = response_data.get("data_extraction", {})

        # Mise à jour des données extraites (On fusionne avec ce qu'on avait déjà)
        if st.session_state["extracted_data"] is None:
            st.session_state["extracted_data"] = {}

        # On ne met à jour que les champs non nuls
        for key, value in new_extracted_data.items():
            if value is not None:
                st.session_state["extracted_data"][key] = value

        # Sauvegarde du message assistant dans l'historique
        st.session_state["messages"].append({"role": "assistant", "content": assistant_message})

        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(assistant_message)

        # 4. VÉRIFICATION : Est-ce qu'on a tout pour lancer le modèle .pkl ?
        if response_data.get("metadata", {}).get("ready_for_analysis", False):

            st.success("✅ Toutes les données sont collectées ! Lancement du diagnostic IA...")

            # --- C'EST ICI QU'ON UTILISE VOTRE MODÈLE PKL ---
            final_data = st.session_state["extracted_data"]

            with st.status("Consultation du modèle neuronal...", expanded=True) as status:
                st.write("Formatage des données...")
                time.sleep(0.5)

                # Appel de la fonction qui utilise le .pkl
                prediction_ia = predict_sleep_disorder(final_data)

                st.write(f"Diagnostic du modèle : **{prediction_ia}**")
                status.update(label="Diagnostic terminé", state="complete", expanded=False)

            # --- APPEL FINAL A GEMINI POUR LE RAPPORT ---
            with st.spinner("Rédaction du rapport médical détaillé..."):
                analysis_report = call_gemini_analysis(final_data, prediction_ia)

            # Affichage du rapport final dans le chat
            final_response_text = f"### 🩺 Résultat du Diagnostic\n\n**Le modèle IA a identifié : {prediction_ia}**\n\n{analysis_report}"

            st.session_state["messages"].append({
                "role": "assistant",
                "content": final_response_text
            })

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(final_response_text)

            # (Optionnel) Afficher les données techniques brutes
            with st.expander("Voir les données utilisées par le modèle"):
                st.json(final_data)

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