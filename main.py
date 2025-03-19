# Import all the tools we need
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import spacy
import os
import string

# ------------------- FIX FOR CHATTERBOT/SPACY ISSUE ------------------- #
from chatterbot.tagging import PosLemmaTagger

class CustomPosLemmaTagger(PosLemmaTagger):
    def __init__(self, language):
        # Override the language handling
        self.language = language  # Store the language string directly
        self.punctuation_table = str.maketrans(dict.fromkeys(string.punctuation))
        
        # Load the spaCy model using the full name
        self.nlp = spacy.load("en_core_web_sm")

# Override the default tagger in chatterbot
from chatterbot.storage import StorageAdapter
StorageAdapter.tagger = CustomPosLemmaTagger
# ---------------------------------------------------------------------- #

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Initialize NLTK sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Initialize and train ChatterBot with custom settings
chatbot = ChatBot(
    "EcoSlayBot",
    tagger=CustomPosLemmaTagger  # Use the custom tagger
)
trainer = ListTrainer(chatbot)

# Load product descriptions
try:
    df_desc = pd.read_csv('product_descriptions.csv')
    conversations = []
    for _, row in df_desc.iterrows():
        conversations.append(row['product'])
        conversations.append(row['description'])
    trainer.train(conversations)
except FileNotFoundError:
    st.error("product_descriptions.csv file is missing!")

# Function to save feedback
def save_feedback(feature, feedback):
    try:
        feedback_df = pd.read_csv('feedback.csv') if os.path.exists('feedback.csv') else pd.DataFrame(columns=['feature', 'feedback'])
        new_feedback = pd.DataFrame([[feature, feedback]], columns=['feature', 'feedback'])
        feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
        feedback_df.to_csv('feedback.csv', index=False)
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# Function for Trend Hunt (scraping from Amazon)
def trend_hunt(persona):
    url = "https://www.amazon.com/s?k=eco+friendly+products"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        products = []
        listings = soup.find_all('span', class_='a-size-base-plus a-color-base a-text-normal')[:5]
        
        for item in listings:
            product_name = item.text.strip()
            if product_name:
                products.append(product_name)
                
        if not products:
            products = ["Eco Yoga Mat", "Bamboo Socks", "Eco T-Shirt", "Hemp Backpack", "Recycled Water Bottle"]
            st.warning("Using default product list")
            
        pd.DataFrame({'product': products}).to_csv('products.csv', index=False)
        df = pd.read_csv('products.csv')
        
        # Get trends
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(kw_list=['eco friendly products'], timeframe='now 7-d')
        trend_data = pytrends.interest_over_time()
        df['trend_score'] = trend_data['eco friendly products'].mean() if not trend_data.empty else 0
        
        # Cluster products
        if not df.empty:
            kmeans = KMeans(n_clusters=min(2, len(df)), random_state=0)
            df['cluster'] = kmeans.fit_predict(df[['trend_score']])
            
        return df
        
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

# Function for Pitch Slaps (emotional pitch generation)
def generate_pitch(product):
    try:
        desc_df = pd.read_csv('product_descriptions.csv')
        desc = desc_df[desc_df['product'] == product]['description'].iloc[0]
        sentiment = sid.polarity_scores(desc)
        if sentiment['compound'] >= 0.05:
            return f"🔥 Slay sustainably with our {product}! {desc}"
        else:
            return f"🌱 Go green in style with {product}! {desc}"
    except Exception as e:
        return f"Error generating pitch: {str(e)}"

# Function for Chat Vibes (chatbot response)
def chatbot_response(user_input):
    try:
        return str(chatbot.get_response(user_input))
    except Exception as e:
        return f"Error: {str(e)}"

# Main app function
def run_app():
    # Load CSS
    css_files = ["global.css", "trends.css", "tryon.css", "pitch.css", "chat.css"]
    for css in css_files:
        try:
            with open(os.path.join("styles", css)) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.error(f"Missing CSS file: {css}")

    # App layout
    st.markdown("""
        <div class="genz-title"><span class="eco-icon">🌿</span> EcoSlay Dropshipping AI</div>
        <div class="genz-sub">Gen Z's Sustainable Shopping Sidekick</div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Trend Hunt 🔥", "Fit Check 👕", "Pitch Slaps 💬", "Chat Vibes 🤖"])
    
    # Trend Hunt Tab
    with tab1:
        st.markdown("<h2>Trend Hunt 🔥</h2>", unsafe_allow_html=True)
        persona = st.selectbox("Who you shopping for?", ["Eco-Conscious Gen Z", "Sustainable Millennials"])
        if st.button("Hunt Trends"):
            df = trend_hunt(persona)
            if not df.empty:
                st.dataframe(df)
            else:
                st.warning("No trends found")
        save_feedback("Trend Hunt", st.radio("Helpful?", ["Yes", "No"], key="trend_fb"))
    
    # Fit Check Tab
    with tab2:
        st.markdown("<h2>Fit Check 👕</h2>", unsafe_allow_html=True)
        st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        st.number_input("Weight (kg)", min_value=30, max_value=150, value=70)
        st.selectbox("Pick gear", ["Eco T-Shirt", "Bamboo Shorts"])
        if st.button("Check Fit"):
            st.success("Fit check coming soon! 🚀")
        save_feedback("Fit Check", st.radio("Helpful?", ["Yes", "No"], key="fit_fb"))
    
    # Pitch Slaps Tab
    with tab3:
        st.markdown("<h2>Pitch Slaps 💬</h2>", unsafe_allow_html=True)
        try:
            products = pd.read_csv('product_descriptions.csv')['product'].tolist()
            product = st.selectbox("Pick a product", products)
            if st.button("Generate Pitch"):
                st.markdown(f"<div class='pitch-card'>{generate_pitch(product)}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
        save_feedback("Pitch Slaps", st.radio("Helpful?", ["Yes", "No"], key="pitch_fb"))
    
    # Chat Vibes Tab
    with tab4:
        st.markdown("<h2>Chat Vibes 🤖</h2>", unsafe_allow_html=True)
        user_input = st.text_input("Ask me anything eco-slay!")
        if user_input:
            response = chatbot_response(user_input)
            st.markdown(f"<div class='chat-response'>{response}</div>", unsafe_allow_html=True)
        save_feedback("Chat Vibes", st.radio("Helpful?", ["Yes", "No"], key="chat_fb"))

if __name__ == "__main__":
    run_app()