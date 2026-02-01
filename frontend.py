import streamlit as st
import requests

# Page Config
st.set_page_config(
    page_title="Story Point Estimator",

    layout="centered"
)

st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1E293B;
    }
    .result-card {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-top: 2rem;
        border-left: 5px solid #2563EB;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State for Navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def go_home():
    st.session_state.page = 'home'

def go_about():
    st.session_state.page = 'about'

col1, col2 = st.columns([6, 1])

with col1:
    if st.session_state.page == 'home':
        st.markdown("<h1 class='main-header'>Agile Based User Story Point Estimation</h1>", unsafe_allow_html=True)
    else:
        st.button("← Back to Estimator", on_click=go_home, type="secondary")

with col2:
    if st.session_state.page == 'home':
        st.button("ℹ️ About", on_click=go_about, help="Learn more about this project")


if st.session_state.page == 'home':
    st.markdown("### AI-Powered User Story Estimation")
    st.write("Enter your user story below to receive an instant Story Point estimation based on historical data.")

    story_text = st.text_area("User Story", height=150, placeholder="As a [role], I want [feature] so that [benefit]...")

    if st.button("✨ Estimate Points", type="primary", use_container_width=True):
        if not story_text.strip():
            st.warning("Please enter a valid story.")
        else:
            with st.spinner("Analyzing complexity..."):
                try:
                    response = requests.post("http://localhost:8000/predict", json={"text": story_text})
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        points = data['predicted_story_points']
                        confidence = data['confidence']
                        raw = data['raw_prediction']
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h3 style="margin-top:0; color: #2563EB;">Estimation Result</h3>
                            <div style="font-size: 2.5rem; font-weight: 800; color: #1E293B;">{points} Points</div>
                            <p style="color: #64748B; margin-bottom: 0.5rem;">Confidence Level: <strong>{confidence.title()}</strong></p>
                            <p style="color: #94A3B8; font-size: 0.875rem;">Raw Model Prediction: {raw}</p>
                        </div>
                        """, unsafe_allow_html=True)
                            
                    else:
                        st.error(f"Error from server: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend. Please ensure the backend server is running on port 8000.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

elif st.session_state.page == 'about':
    st.markdown("<h1 class='main-header'>About the Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <p style="font-size: 1.1rem; color: #334155;">
            This tool leverages <strong>Machine Learning</strong> to bring consistency and objectivity to Agile estimation. 
            By analyzing the text of your user stories, it predicts the complexity score (Story Points) using a model trained on historical project data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - **Frontend**: Streamlit
        - **Backend**: FastAPI
        - **ML Model**: Scikit-Learn (Random Forest)
        - **Text Processing**: TF-IDF Vectorization
        """)
        
    with c2:
        st.markdown("### 📊 How it Works")
        st.markdown("""
        1. **Cleaning**: Removes noise and standardization.
        2. **Vectorization**: Converts text to numerical vectors.
        3. **Prediction**: Predicts continuous complexity score.
        4. **Mapping**: Snaps to nearest Fibonacci point (1, 2, 3, 5, 8, 13).
        """)

st.markdown("---")

