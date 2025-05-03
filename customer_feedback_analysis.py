import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# Load the ML model and vectorizer
model = pickle.load(open('final_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Add Neo-Glass style
def set_glassmorphism_theme():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif;
        background-image: url('https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1650&q=80');
        background-size: cover;
        background-position: center;
    }

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-color: rgba(0,0,0,0.5);
        z-index: -1;
    }

    .glass-box {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: auto;
        width: 100%;
        max-width: 600px;
    }

    h1 {
        color: white;
        text-align: center;
        font-weight: 600;
        font-size: 2.4rem;
        text-shadow: 1px 1px 4px #000;
    }

    .stTextInput>div>div>input {
        background-color: #ffffffdd;
        color: #000;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }

    .stButton>button {
        background-color: #00c9a7;
        color: #fff;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 16px;
        transition: 0.3s ease;
        box-shadow: 0px 4px 15px rgba(0, 201, 167, 0.5);
    }

    .stButton>button:hover {
        background-color: #00b195;
        transform: scale(1.05);
    }

    p, .stMarkdown {
        color: #eeeeee;
        text-align: center;
        font-size: 16px;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Clean text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    words = text.split()
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words if w not in set(stopwords.words('english'))]
    return ' '.join(words)

# Layout
set_glassmorphism_theme()

st.markdown('<div class="glass-box">', unsafe_allow_html=True)

st.markdown("<h1>🔮 Customer Feedback Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p>Instantly identify whether feedback is positive or negative</p>", unsafe_allow_html=True)

review = st.text_input("💬 Enter Customer Review")

if st.button("🚀 Predict"):
    if not review.strip():
        st.warning("Please enter a review.")
    elif review.isdigit():
        st.error("Text review only. Numbers are not valid input.")
    else:
        cleaned = clean_text(review)
        vect = vectorizer.transform([cleaned]).toarray()
        pred = model.predict(vect)
        if pred[0] == 1:
            st.success("✅ Positive Feedback")
        else:
            st.error("❌ Negative Feedback")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<br><hr>
<p style="text-align:center;color:#cccccc;font-size:13px">
    Built by Akash • © 2025 • Streamlit App
</p>
""", unsafe_allow_html=True)
