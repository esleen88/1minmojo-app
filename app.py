import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. Load Emotion Data ==========
emotion_data = pd.DataFrame({
    'text': [
        "I'm feeling great today!", "This is so frustrating", "I'm really sad about this",
        "I'm excited for the future", "This situation makes me anxious", "I feel proud of myself",
        "I'm completely burned out", "My heart is broken after what happened",
        "We won!", "This is terrifying", "I miss them deeply", "You're everything to me"
    ],
    'emotion_name': [
        'joy', 'anger', 'sadness', 'excitement', 'fear', 'pride',
        'sadness', 'grief', 'pride', 'fear', 'grief', 'love'
    ]
})

# ========== 2. Define Personas ==========
personas = {
    "Sales Hustler": {
        "emotions": ['joy', 'excitement', 'pride'],
        "responses": [
            "💼 Hustler says: {} Now close that deal!",
            "💼 Hustler says: {} You’re on fire today!",
            "💼 Hustler says: {} Keep the momentum going!"
        ]
    },
    "The Burnout": {
        "emotions": ['sadness', 'anger', 'fear'],
        "responses": [
            "🛌 Burnout buddy says: {} Even volcanoes rest.",
            "🛌 Burnout buddy says: {} Breathe. One thing at a time.",
            "🛌 Burnout buddy says: {} Rest is productive too."
        ]
    },
    "Broken Hearted": {
        "emotions": ['grief', 'sadness', 'love'],
        "responses": [
            "💔 Heart friend says: {} The cracks let the light in.",
            "💔 Heart friend says: {} You're not alone.",
            "💔 Heart friend says: {} Healing takes time."
        ]
    },
    "Warrior": {
        "emotions": ['joy', 'fear', 'pride'],
        "responses": [
            "🛡️ Warrior says: {} You’ve survived 100% of your bad days.",
            "🛡️ Warrior says: {} Every scar tells a strength story.",
            "🛡️ Warrior says: {} You’re built for this!"
        ]
    }
}

# ========== 3. Load the Model ==========
model = SentenceTransformer('all-MiniLM-L6-v2')

# ========== 4. Create Persona Datasets ==========
persona_datasets = {}
for name, config in personas.items():
    filtered = emotion_data[emotion_data['emotion_name'].isin(config['emotions'])]
    embeddings = model.encode(filtered['text'].tolist())
    persona_datasets[name] = {
        'texts': filtered['text'].tolist(),
        'embeddings': embeddings,
        'responses': config['responses']
    }

# ========== 5. Streamlit UI ==========
st.title("✨ 1MinMojo — Get a Pep Talk in 1 Minute ✨")
st.markdown("Choose a persona and share what you're feeling. Get a short emotional boost 🫶")

persona_choice = st.selectbox("Who's your emotional buddy today?", list(personas.keys()))
user_input = st.text_area("What’s on your mind?")

if st.button("Send me a pep talk!"):
    if user_input.strip() == "":
        st.warning("Please type something first.")
    else:
        selected = persona_datasets[persona_choice]
        input_emb = model.encode([user_input])
        sims = cosine_similarity(input_emb, selected['embeddings'])
        best_idx = np.argmax(sims)
        response_template = np.random.choice(selected['responses'])
        pep = response_template.format(selected['texts'][best_idx])
        st.success(pep)
