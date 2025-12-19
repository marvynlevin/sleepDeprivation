import streamlit as st
import time
import os
from google import genai
import json
import dotenv
import joblib
import pandas as pd


@st.cache_resource
def load_brain():
    try:
        return joblib.load('sleep_model_artifacts.pkl')
    except FileNotFoundError:
        return None


artifacts = load_brain()


def predict_sleep_disorder(user_data):
    if artifacts is None:
        return "Modèle introuvable (fichier .pkl manquant)"

    try:
        bp = user_data.get('Blood Pressure', '120/80')
        if '/' in bp:
            systolic, diastolic = map(int, bp.split('/'))
        else:
            systolic, diastolic = 120, 80

        # Préparation du DataFrame pour la prédiction
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

        pipeline = artifacts['model']
        le = artifacts['label_encoder']

        pred_code = pipeline.predict(df_input)
        pred_label = le.inverse_transform(pred_code)[0]

        return pred_label

    except Exception as e:
        return f"Erreur technique : {str(e)}"


dotenv.load_dotenv()

if "GEMINI_API_KEY" in os.environ:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
else:
    st.error("La clé API GEMINI_API_KEY n'est pas configurée dans les variables d'environnement.")
    client = None

st.set_page_config(
    layout="wide",
    page_title="Sleepy - Analyse du Sommeil",
    initial_sidebar_state="collapsed",
)
# --------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@400;600;700&family=Nunito:wght@300;400;600;800;900&display=swap');

    .stApp header {
        display: none !important;
    }

    /* Masquer le footer "Made with Streamlit" */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Polices */
    * {
        font-family: 'Nunito', sans-serif;
        font-size: 1.1rem;
    }

    h1, h2, h3, .choice-title {
        font-family: 'Caveat', cursive;
    }

    .stMainBlockContainer {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* Headers */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
    }

    h2 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }

    h3 {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Titres en majuscules extra bold */
    .extra-bold {
        font-weight: 900 !important;
        letter-spacing: 1px;
    }

    /* Tabs styling (conserve car très personnalisé) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        font-size: 1.15rem;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        color: #667eea;
    }

    .stButton button {
        margin-top: 5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white;
        border: none !important;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton button:hover {
        transform: translateY(2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Les règles pour Sliders, Inputs, Selectbox sont retirées et gérées par primaryColor */

    /* Cards effect */
    .element-container {
        animation: fadeIn 0.6s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Success, warning, error boxes */
    .stSuccess {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #48bb78;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1.15rem;
    }

    .stWarning {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        border-left: 5px solid #ed8936;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1.15rem;
    }

    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #f56565;
        border-radius: 8px;
        padding: 1rem;
        font-size: 1.15rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 1.15rem;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }

    /* Choice cards - Rendu cliquable */
    .choice-card-container {
        cursor: pointer;
        transition: all 0.3s ease;
        padding: 0 !important; /* Retirer padding pour que la carte soit la zone de clic */
    }
    .choice-card {
        background: linear-gradient(135deg, #f0f2f8 0%, #e7e8f0 100%);
        border-radius: 20px;
        padding: 0.5rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 3px solid transparent;
        height: 100%;
        box-sizing: border-box;
    }

    .choice-card-container:hover .choice-card {
        transform: translateY(-3px);
        border-color: #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .choice-card-container:active .choice-card {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    .choice-icon {
        font-size: 5rem;
        margin-bottom: 0.3rem;
    }

    .choice-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
    }

    .choice-description {
        font-size: 1.2rem;
        color: #718096;
        line-height: 1.6;
    }

    /* Labels et Captions (Streamlit les gère, mais on assure la taille) */
    label {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }

    .stCaption {
        font-size: 1.05rem !important;
    }
</style>
""", unsafe_allow_html=True)

MODEL_TO_USE = "gemini-flash-latest"

analysis_prompt = """
Commence IMPÉRATIVEMENT ta réponse par la phrase exacte suivante : 'Bonjour ! Je suis l'assistant Sleepy, et voici le rapport détaillé de votre analyse.'

Agis comme un assistant virtuel expert en hygiène de sommeil. Ta mission est de générer un rapport concis, structuré en exactement trois paragraphes, sans aucune liste à puces, en t'appuyant sur les données patient et le diagnostic prédictif fourni.

**RÈGLE DE TON ET VOCABULAIRE :** Utilise un langage simple, accessible et non-médical. Si tu dois utiliser un mot technique ou complexe (ex: "apnée", "hygiène", "comorbidité"), **tu dois impérativement l'expliquer immédiatement entre parenthèses ( )**.

STRICTE STRUCTURE DE RÉPONSE (3 PARAGRAPHES OBLIGATOIRES) :

1. PARAGRAPHE D'ANALYSE (Diagnostic et Liens Factuels) :
    - Confirme clairement le diagnostic ('Healthy', 'Insomnia', 'Sleep Apnea').
    - Analyse et explique ce résultat en te basant **uniquement** sur les données du patient (Stress Level, Sleep Duration, BMI Category, etc.). Cite explicitement les données clés qui justifient la conclusion. (Ex: "Le diagnostic d'Insomnia est cohérent avec votre niveau de stress élevé (X/10) et votre courte durée de sommeil (Y heures).")

2. PARAGRAPHE DE CONTEXTUALISATION ET D'IMPACT (Signification et Risques) :
    - Décris brièvement ce que signifie le diagnostic pour la santé quotidienne de l'utilisateur.
    - Pour 'Insomnia' ou 'Sleep Apnea', indique clairement les risques potentiels associés ou la nécessité de consultation médicale (surtout pour l'Apnée du Sommeil).

3. PARAGRAPHE DE RECOMMANDATIONS (Trois Actions Clés) :
    - Fournis **exactement trois** recommandations concrètes et spécifiques, adaptées au profil du patient et à son diagnostic. Chaque recommandation doit être courte et directement actionable. (Ex: "Augmenter l'activité physique à [X minutes] par jour.")

Adopte un ton professionnel, concis et serviable. Le rapport final doit contenir l'ouverture, les trois paragraphes, et se terminer IMPÉRATIVEMENT par la phrase exacte suivante : 'En espérant que cela puisse vous aider et à vous revoir d'ici peu pour retester !'

Les paragraphes seront de petite taille.

NE JAMAIS inclure la liste des champs ou des exemples dans la réponse finale. Le corps de la réponse ne doit être que du texte formaté selon ces règles.
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
    "message_to_user": "Ici, tu poses une question polie en français pour obtenir les données manquantes, ou tu confirms la fin de la collecte. **Tu dois prioriser les questions thématiques et logiques, ne demandant qu'un ou deux groupes d'informations à la fois (ex: toutes les informations liées au sport, ou toutes les informations physiologiques).**",
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

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = None

if "app_loaded" not in st.session_state:
    st.session_state["app_loaded"] = False

if "mode_selected" not in st.session_state:
    st.session_state["mode_selected"] = False

if "selected_mode" not in st.session_state:
    st.session_state["selected_mode"] = None

if "show_report" not in st.session_state:
    st.session_state["show_report"] = False

if "report_content" not in st.session_state:
    st.session_state["report_content"] = ""

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = ""


def call_gemini_chat(user_message: str) -> dict:
    if client is None:
        return {
            "user_interaction": {"message_to_user": "La fonction est désactivée car la clé API est manquante.",
                                 "missing_fields": []},
            "data_extraction": {},
            "metadata": {"validity_check": {"is_valid": False, "errors": ["API Key Missing"]}, "confidence_score": 0.0,
                         "ready_for_analysis": False}
        }

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


def call_gemini_analysis(user_data, ai_prediction=None):
    if client is None:
        return "La fonction est désactivée car la clé API est manquante."

    try:
        prediction_text = f"Le modèle prédictif (XGBoost) a diagnostiqué : {ai_prediction}" if ai_prediction else "Le modèle prédictif n'a pas été exécuté."

        data_text = f"""
Données du patient : {user_data}

{prediction_text}

Consigne :
1. Respecte STRICTEMENT la structure de 3 paragraphes définie dans la System Instruction.
2. Explique le résultat en te basant sur les données (Stress, IMC, Tension...).
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


# Page de chargement
if not st.session_state["app_loaded"]:
    loading_container = st.container()
    with loading_container:
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh;">
            <svg width="140" height="140" viewBox="0 0 120 120" xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)">
                <defs>
                    <linearGradient id="sleepyGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
                    </linearGradient>
                </defs>
                <circle cx="60" cy="60" r="50" fill="url(#sleepyGradient)" opacity="0.2"/>
                <path d="M 40 50 Q 40 40, 50 40 Q 60 40, 60 50" stroke="url(#sleepyGradient)" stroke-width="4" fill="none" stroke-linecap="round"/>
                <path d="M 60 50 Q 60 40, 70 40 Q 80 40, 80 50" stroke="url(#sleepyGradient)" stroke-width="4" fill="none" stroke-linecap="round"/>
                <path d="M 45 70 Q 60 80, 75 70" stroke="url(#sleepyGradient)" stroke-width="4" fill="none" stroke-linecap="round"/>
                <circle cx="60" cy="60" r="45" stroke="url(#sleepyGradient)" stroke-width="3" fill="none">
                    <animate attributeName="stroke-dasharray" from="0 283" to="283 283" dur="2s" repeatCount="indefinite"/>
                </circle>
            </svg>
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 4.5rem; font-weight: 800; margin-top: 2rem; letter-spacing: -1px; font-family: 'Caveat', cursive;">
                Sleepy
            </h1>
            <p style="color: #718096; font-size: 1.4rem; margin-top: 0.5rem; font-family: 'Nunito', sans-serif;">analyse intelligente du sommeil</p>
            <div style="margin-top: 2rem;">
                <div style="width: 200px; height: 4px; background: #e2e8f0; border-radius: 10px; overflow: hidden;">
                    <div style="height: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); animation: loading 1.5s ease-in-out infinite;"></div>
                </div>
            </div>
        </div>
        <style>
            @keyframes loading {
                0% { width: 0%; }
                50% { width: 100%; }
                100% { width: 0%; }
            }
        </style>
        """, unsafe_allow_html=True)

    time.sleep(2)
    st.session_state["app_loaded"] = True
    st.rerun()

# Page de sélection du mode
if not st.session_state["mode_selected"]:
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 4rem; font-weight: 800; margin-bottom: 1rem; font-family: 'Caveat', cursive;">
            Sleepy
        </h1>
        <p style="color: #718096; font-size: 1.4rem; font-family: 'Nunito', sans-serif;">comment souhaitez-vous procéder ?</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="choice-card-container">
            <div class="choice-card">
                <div class="choice-icon">💬</div>
                <div class="choice-title">Conversation guidée</div>
                <div class="choice-description">
                    laissez l'assistant vous poser des questions de manière naturelle et fluide
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(" Aller à la conversation guidée ", key="chat_mode", use_container_width=True,
                     help="Conversation guidée"):
            st.session_state["mode_selected"] = True
            st.session_state["selected_mode"] = "conversation"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="choice-card-container">
            <div class="choice-card">
                <div class="choice-icon">📝</div>
                <div class="choice-title">Saisie manuelle</div>
                <div class="choice-description">
                    remplissez directement un formulaire structuré avec toutes vos informations
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(" Aller à la saisie manuelle ", key="form_mode", use_container_width=True, help="Saisie manuelle"):
            st.session_state["mode_selected"] = True
            st.session_state["selected_mode"] = "formulaire"
            st.rerun()

    st.stop()

col_logo, col_back = st.columns([5, 1])

with col_logo:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
        <svg width="70" height="40" viewBox="0 0 120 120" xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)">
            <defs>
                <linearGradient id="headerGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
                </linearGradient>
            </defs>
            <circle cx="60" cy="60" r="50" fill="url(#headerGradient)" opacity="0.2"/>
            <path d="M 40 50 Q 40 40, 50 40 Q 60 40, 60 50" stroke="url(#headerGradient)" stroke-width="4" fill="none" stroke-linecap="round"/>
            <path d="M 60 50 Q 60 40, 70 40 Q 80 40, 80 50" stroke="url(#headerGradient)" stroke-width="4" fill="none" stroke-linecap="round"/>
            <path d="M 45 70 Q 60 80, 75 70" stroke="url(#headerGradient)" stroke-width="4" fill="none" stroke-linecap="round"/>
        </svg>
        <div>
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; font-size: 3.5rem; font-weight: 800; font-family: 'Caveat', cursive;">
                Sleepy
            </h1>
            <p style="color: #718096; margin: 0; font-size: 1.2rem; font-family: 'Nunito', sans-serif;">votre assistant d'analyse du sommeil</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_back:
    if st.button("← Changer de mode", use_container_width=True):
        st.session_state["mode_selected"] = False
        st.session_state["selected_mode"] = None
        st.rerun()


# Modal pour le rapport
@st.dialog("RAPPORT DÉTAILLÉ", width="large")
def show_report_modal():
    st.markdown(f"### Diagnostic : {st.session_state['prediction_result']}")

    if st.session_state['prediction_result'] == "None" or st.session_state['prediction_result'] == "Healthy":
        st.success("Votre profil de sommeil semble équilibré")
    elif st.session_state['prediction_result'] == "Insomnia":
        st.warning("Des troubles du sommeil ont été identifiés")
    elif st.session_state['prediction_result'] == "Sleep Apnea":
        st.error("Attention : consultation médicale recommandée")

    st.divider()
    st.markdown(st.session_state["report_content"])


if st.session_state.get("show_report", False):
    show_report_modal()

# Affichage selon le mode sélectionné
if st.session_state["selected_mode"] == "formulaire":
    st.markdown('<p class="extra-bold" style="font-size: 2rem;">DIAGNOSTIC RAPIDE</p>', unsafe_allow_html=True)
    st.caption("Remplissez vos informations pour obtenir un diagnostic en quelques secondes")

    with st.form(key='sleep_form'):
        st.markdown("### Informations personnelles")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("genre", ['Male', 'Female'])
        age = c2.number_input("âge", 20, 90, 40)
        occupation = c3.selectbox("profession",
                                  ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse',
                                   'Engineer', 'Accountant', 'Scientist', 'Lawyer', 'Salesperson', 'Manager'])

        st.markdown("### Qualité du sommeil")
        c1, c2 = st.columns(2)
        sleep_duration = c1.number_input("durée de sommeil (heures)", 0.0, 24.0, 7.0, step=0.5)
        sleep_quality = c2.slider("qualité du sommeil", 1, 10, 7, help="1 = très mauvais, 10 = excellent")

        st.markdown("### Activité & stress")
        c1, c2 = st.columns(2)
        physical_activity = c1.slider("activité physique (min/jour)", 0, 120, 40)
        stress_level = c2.slider("niveau de stress", 1, 10, 5, help="1 = très calme, 10 = très stressé")

        st.markdown("### Santé physique")
        c1, c2 = st.columns(2)
        bmi_category = c1.selectbox("catégorie IMC", ['Normal', 'Overweight', 'Obese'])
        blood_pressure = c2.text_input("tension artérielle", "120/80", help="format: systolique/diastolique")

        c1, c2 = st.columns(2)
        heart_rate = c1.number_input("fréquence cardiaque (bpm)", 30, 200, 70)
        daily_steps = c2.number_input("pas quotidiens", 0, 30000, 5000, step=500)

        submit_button = st.form_submit_button("Lancer le diagnostic")

    if submit_button:
        user_data = {
            "Gender": gender, "Age": age, "Occupation": occupation,
            "Sleep Duration": sleep_duration, "Quality of Sleep": sleep_quality,
            "Physical Activity Level": physical_activity, "Stress Level": stress_level,
            "BMI Category": bmi_category, "Blood Pressure": blood_pressure,
            "Heart Rate": heart_rate, "Daily Steps": daily_steps
        }

        with st.spinner("analyse en cours..."):
            pred_ia = predict_sleep_disorder(user_data)
            report = call_gemini_analysis(user_data, pred_ia)

        st.session_state["prediction_result"] = pred_ia
        st.session_state["report_content"] = report
        st.session_state["show_report"] = True
        st.rerun()

elif st.session_state["selected_mode"] == "conversation":
    st.markdown('<p class="extra-bold" style="font-size: 2rem;">CONVERSATION AVEC L\'ASSISTANT</p>',
                unsafe_allow_html=True)
    st.caption("Laissez l'assistant vous guider à travers les questions pour établir votre profil de sommeil")

    chat_container = st.container(height=380, border=True)

    with chat_container:
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_input = st.chat_input(placeholder="décrivez votre situation... (ex: j'ai 30 ans et je dors mal)",
                               key="user_message_chat")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        with st.spinner("l'assistant réfléchit..."):
            response_data = call_gemini_chat(user_input)

        assistant_message = response_data.get("user_interaction", {}).get("message_to_user", "je n'ai pas compris")
        new_extracted_data = response_data.get("data_extraction", {})

        if st.session_state["extracted_data"] is None:
            st.session_state["extracted_data"] = {}

        for key, value in new_extracted_data.items():
            if value is not None:
                st.session_state["extracted_data"][key] = value

        st.session_state["messages"].append({"role": "assistant", "content": assistant_message})

        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(assistant_message)

        if response_data.get("metadata", {}).get("ready_for_analysis", False):
            st.success("toutes les données sont collectées ! lancement du diagnostic...")

            final_data = st.session_state["extracted_data"]

            with st.status("analyse du modèle neuronal...", expanded=True) as status:
                st.write("préparation des données...")
                time.sleep(0.5)

                prediction_ia = predict_sleep_disorder(final_data)

                st.write(f"**diagnostic : {prediction_ia}**")
                status.update(label="diagnostic terminé", state="complete", expanded=False)

            with st.spinner("génération du rapport détaillé..."):
                analysis_report = call_gemini_analysis(final_data, prediction_ia)

            st.session_state["prediction_result"] = prediction_ia
            st.session_state["report_content"] = analysis_report
            st.session_state["show_report"] = True

            final_response_text = f"le diagnostic est prêt ! cliquez sur le bouton ci-dessous pour consulter le rapport complet"

            st.session_state["messages"].append({
                "role": "assistant",
                "content": final_response_text
            })

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(final_response_text)

            if st.button("voir le rapport complet", use_container_width=True, type="primary"):
                st.session_state["show_report"] = True
                st.rerun()

            with st.expander("voir les données utilisées"):
                st.json(final_data)
