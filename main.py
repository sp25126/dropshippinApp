import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from sklearn.cluster import KMeans
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
# Download NLTK data (if not already downloaded)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('vader_lexicon')

# Initialize NLTK sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Initialize and train ChatterBot
chatbot = ChatBot("EcoSlayBot")
trainer = ListTrainer(chatbot)
df_desc = pd.read_csv('product_descriptions.csv')
conversations = []
for _, row in df_desc.iterrows():
    conversations.append(row['product'])
    conversations.append(row['description'])
trainer.train(conversations)

# Function to save feedback
def save_feedback(feature, feedback):
    feedback_df = pd.read_csv('feedback.csv') if pd.io.common.file_exists('feedback.csv') else pd.DataFrame(columns=['feature', 'feedback'])
    new_feedback = pd.DataFrame([[feature, feedback]], columns=['feature', 'feedback'])
    feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    feedback_df.to_csv('feedback.csv', index=False)

# Function for Trend Hunt (scraping from Amazon)
def trend_hunt(persona):
    # Scrape product data from Amazon
    url = "https://www.amazon.com/s?k=eco+friendly+products"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        products = []
        for item in soup.find_all('span', class_='a-size-base-plus a-color-base a-text-normal')[:5]:
            product_name = item.text.strip()
            if product_name:
                products.append(product_name)
    except requests.RequestException as e:
        st.error(f"Error scraping products: {e}")
        products = []

    # If no products were scraped, use a default dataset
    if not products:
        products = ["Eco Yoga Mat", "Bamboo Socks", "Eco T-Shirt", "Hemp Backpack", "Recycled Water Bottle"]
        st.warning("No products found online. Using default product list.")

    # Write to products.csv
    with open('products.csv', 'w') as f:
        f.write("product,price,category\n")
        for i, prod in enumerate(products):
            f.write(f"{prod},{20 + i*5},Eco-Friendly\n")

    # Load products
    df = pd.read_csv('products.csv')
    
    # Use pytrends to get trending data
    pytrends = TrendReq(hl='en-US', tz=360)
    try:
        pytrends.build_payload(kw_list=['eco friendly products'], timeframe='now 7-d')
        trend_data = pytrends.interest_over_time()
        trend_score = trend_data['eco friendly products'].mean() if not trend_data.empty else 0
    except Exception as e:
        st.error(f"Error fetching trends: {e}")
        trend_score = 0

    # Add trend score to DataFrame
    df['trend_score'] = trend_score

    # Cluster products (only if DataFrame has data)
    if len(df) > 0:
        kmeans = KMeans(n_clusters=min(2, len(df)), random_state=0).fit(df[['price']])
        df['cluster'] = kmeans.labels_
    else:
        st.error("No data available for clustering.")
        df['cluster'] = 0  # Default cluster value

    return df

    # If no products were scraped, use a default dataset
    if not products:
        products = ["Eco Yoga Mat", "Bamboo Socks", "Eco T-Shirt", "Hemp Backpack", "Recycled Water Bottle"]
        st.warning("No products found online. Using default product list.")

    # Write to products.csv
    with open('products.csv', 'w') as f:
        f.write("product,price,category\n")
        for i, prod in enumerate(products):
            f.write(f"{prod},{20 + i*5},Eco-Friendly\n")

    # Load products
    df = pd.read_csv('products.csv')
    
    # Use pytrends to get trending data
    pytrends = TrendReq(hl='en-US', tz=360)
    try:
        pytrends.build_payload(kw_list=['eco friendly products'], timeframe='now 7-d')
        trend_data = pytrends.interest_over_time()
        trend_score = trend_data['eco friendly products'].mean() if not trend_data.empty else 0
    except Exception as e:
        st.error(f"Error fetching trends: {e}")
        trend_score = 0

    # Add trend score to DataFrame
    df['trend_score'] = trend_score

    # Cluster products (only if DataFrame has data)
    if len(df) > 0:
        kmeans = KMeans(n_clusters=min(2, len(df)), random_state=0).fit(df[['price']])
        df['cluster'] = kmeans.labels_
    else:
        st.error("No data available for clustering.")
        df['cluster'] = 0  # Default cluster value

    return df

    # Load products
    df = pd.read_csv('products.csv')
    
    # Use pytrends to get trending data
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload(kw_list=['eco friendly products'], timeframe='now 7-d')
    trend_data = pytrends.interest_over_time()
    trend_score = trend_data['eco friendly products'].mean() if not trend_data.empty else 0

    # Cluster products (simplified KMeans for demo)
    df['trend_score'] = trend_score
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df[['price']])
    df['cluster'] = kmeans.labels_

    return df

# Function for Pitch Slaps (emotional pitch generation)
def generate_pitch(product):
    df_desc = pd.read_csv('product_descriptions.csv')
    desc = df_desc[df_desc['product'] == product]['description'].iloc[0] if product in df_desc['product'].values else "No description available."
    sentiment = sid.polarity_scores(desc)
    if sentiment['compound'] >= 0.05:
        pitch = f"Feel the vibe with this {product}! 🌟 {desc}"
    else:
        pitch = f"Get cozy with this {product}! 🧘 {desc}"
    return pitch

# Function for Chat Vibes (chatbot response)
def chatbot_response(user_input):
    return str(chatbot.get_response(user_input))

# Main app function
def run_app():
    # Load CSS
    with open('styles/global.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Trend Hunt 📈", "Fit Check 👕", "Pitch Slaps 💬", "Chat Vibes 🤖"])

    # Tab 1: Trend Hunt
    with tab1:
        st.markdown("<h2>Trend Hunt 📈</h2>", unsafe_allow_html=True)
        st.write("Pick your vibe, fam!")
        persona = st.selectbox("Who you shopping for?", ["Eco-Conscious Gen Z", "Sustainable Millennials", "Green Boomers"])
        if st.button("Hunt Trends"):
            df = trend_hunt(persona)
            st.write(f"Trending for {persona}:")
            st.dataframe(df)
        feedback = st.radio("Was this helpful? Yes/No", ["Yes", "No"], key="trend_feedback")
        if feedback:
            save_feedback("Trend Hunt", feedback)
            st.write("Thanks, fam!")

    # Tab 2: Fit Check
    with tab2:
        st.markdown("<h2>Fit Check 👕</h2>", unsafe_allow_html=True)
        with open('styles/tryon.css') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        st.write("Drop your stats, let’s see if it fits!")
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=150, value=70)
        clothing_item = st.selectbox("What you tryna fit?", ["Eco T-Shirt", "Bamboo Socks", "Eco Yoga Mat"])
        if st.button("Check Fit"):
            st.write("Fit Check on hold—need more space vibes! 🚀")
        feedback = st.radio("Was this helpful? Yes/No", ["Yes", "No"], key="fit_feedback")
        if feedback:
            save_feedback("Fit Check", feedback)
            st.write("Thanks, fam!")

    # Tab 3: Pitch Slaps
    with tab3:
        st.markdown("<h2>Pitch Slaps 💬</h2>", unsafe_allow_html=True)
        with open('styles/pitch.css') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        st.write("Let’s make it emotional, fam!")
        df_desc = pd.read_csv('product_descriptions.csv')
        product = st.selectbox("Pick a product to pitch:", df_desc['product'].tolist())
        if st.button("Slap a Pitch"):
            pitch = generate_pitch(product)
            st.markdown(f"<div class='pitch-card'>{pitch}</div>", unsafe_allow_html=True)
        feedback = st.radio("Was this helpful? Yes/No", ["Yes", "No"], key="pitch_feedback")
        if feedback:
            save_feedback("Pitch Slaps", feedback)
            st.write("Thanks, fam!")

    # Tab 4: Chat Vibes
    with tab4:
        st.markdown("<h2>Chat Vibes 🤖</h2>", unsafe_allow_html=True)
        with open('styles/chat.css') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        st.write("Hit me up—what you wanna know?")
        user_input = st.text_input("Ask away, fam!", key="chat_input")
        if user_input:
            response = chatbot_response(user_input)
            st.markdown(f"<div class='chat-response'>{response}</div>", unsafe_allow_html=True)
        feedback = st.radio("Was this helpful? Yes/No", ["Yes", "No"], key="chat_feedback")
        if feedback:
            save_feedback("Chat Vibes", feedback)
            st.write("Thanks, fam!")

if __name__ == "__main__":
    run_app()