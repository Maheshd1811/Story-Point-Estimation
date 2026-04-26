import streamlit as st
import requests
import time

st.set_page_config(
    page_title="Story Point Estimator Pro",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #15B8C5 100%);
        color: white;
    }
    
    /* Typography */
    h1, h2, h3, p, div {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FFD700, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 3rem;
        animation: fadeInDown 1s ease-out;
    }

    .sub-header {
        text-align: center;
        font-weight: 400;
        color: #E2E8F0;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-top: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
    }

    /* Input area styling */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #1E293B !important;
        border-radius: 12px;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .stTextArea textarea:focus {
        border: 2px solid #FF8C00;
        box-shadow: 0 0 15px rgba(255, 140, 0, 0.5);
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #FF8C00, #FF0080);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 0, 128, 0.4);
        width: 100%;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(255, 0, 128, 0.6);
        background: linear-gradient(90deg, #FF0080, #FF8C00);
    }

    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Expanders styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Result styling */
    .result-points {
        font-size: 4rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00FF87, #60EFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 1rem 0;
    }

    .result-label {
        color: #E2E8F0;
        text-align: center;
        font-size: 1.2rem;
    }

</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>✨ Agile Estimation AI ✨</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-header'>Instantly predict story points using Machine Learning</h3>", unsafe_allow_html=True)

# Main input section
st.markdown("""
<div class="glass-card">
    <h3 style="margin-top:0; color: white;">📝 Enter User Story</h3>
</div>
""", unsafe_allow_html=True)

story_text = st.text_area("story_input", height=150, placeholder="As a [role], I want [feature] so that [benefit]...", label_visibility="collapsed")

if st.button("🚀 Estimate Points"):
    if not story_text.strip():
        st.warning("⚠️ Please enter a valid story.")
    else:
        # Fake loading for interactivity feel
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("<p style='color: white; text-align: center;'>Analyzing syntax and semantics...</p>", unsafe_allow_html=True)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        
        status_text.empty()
        progress_bar.empty()
        
        with st.spinner("Connecting to Predictor Model..."):
            try:
                response = requests.post("http://localhost:8000/predict", json={"text": story_text})
                
                if response.status_code == 200:
                    data = response.json()
                    points = data.get('predicted_story_points', 'N/A')
                    confidence = data.get('confidence', 'High')
                    
                    st.markdown(f"""
                    <div class="glass-card" style="text-align: center; animation: fadeInUp 0.5s ease-out;">
                        <h2 style="color: white; margin-top: 0;">Estimation Complete! 🎉</h2>
                        <div class="result-points">{points}</div>
                        <div class="result-label">Recommended Story Points</div>
                        <p style="color: #FFD700; margin-top: 1rem;">Confidence: <strong>{confidence.title()}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.error(f"Server Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Could not connect to backend. Please ensure the backend server is running on port 8000.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

st.markdown("<br><br>", unsafe_allow_html=True)

with st.expander("🛠️ How it Works & Tech Stack"):
    st.markdown("""
    <div style="color: white;">
    <p>This tool replaces guesswork with data-driven objectivity. It uses a trained Machine Learning model to evaluate the complexity of your user stories.</p>
    
    <h4>🚀 Step-by-Step Process:</h4>
    <ol>
        <li><strong>Input:</strong> You write a user story.</li>
        <li><strong>NLP Pipeline:</strong> The text is cleaned, tokenized, and converted to numbers (TF-IDF).</li>
        <li><strong>Prediction:</strong> A robust Random Forest model predicts a continuous complexity score based on historical data.</li>
        <li><strong>Fibonacci Mapping:</strong> The score is snapped to the nearest Agile standard sequence number (1, 2, 3, 5, 8, 13...).</li>
    </ol>
    
    <h4>💻 Tech Stack:</h4>
    <ul>
        <li><strong>Frontend:</strong> Streamlit (with custom CSS)</li>
        <li><strong>Backend:</strong> FastAPI</li>
        <li><strong>ML Model:</strong> Random Forest </li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
