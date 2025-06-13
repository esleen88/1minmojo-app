!pip install datasets transformers torch pandas numpy scikit-learn
!pip install sentence-transformers
!pip install streamlit

import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. DATA SETUP ==========
# Expanded emotional dataset with 32 examples 
expanded_data = pd.DataFrame({
    'text': [
        # Original 8 examples
        "I'm feeling great today!", 
        "This is so frustrating", 
        "I'm really sad about this",
        "I'm excited for the future",
        "This situation makes me anxious",
        "I feel so proud of my accomplishments",
        "I'm feeling completely burned out",
        "My heart is broken after what happened",
        
        # Additional positive emotions (joy, excitement, pride)
        "What a wonderful day!",
        "I can't wait to see what happens next!",
        "We won the competition!",
        "This is the best news I've heard all year",
        "I'm thrilled with my progress",
        "My team did an amazing job",
        "Everything is working out perfectly",
        "I feel on top of the world",
        
        # Negative emotions (anger, sadness, fear)
        "I can't believe they did that to me",
        "Why does this always happen to me?",
        "I'm so disappointed by this outcome",
        "This makes my blood boil",
        "I don't know how to handle this loss",
        "The thought of that terrifies me",
        "I'm dreading what comes next",
        "This is absolutely unacceptable",
        
        # Complex emotions (grief, love, confusion)
        "I miss them so much it hurts",
        "My feelings for you grow every day",
        "I don't understand what's happening",
        "This doesn't make any sense to me",
        "My chest aches with longing",
        "You mean everything to me",
        "I'm completely lost right now",
        "Why can't things go back to how they were?"
    ],
    'emotion': [
        # Original emotions
        17, 2, 25, 13, 14, 21, 25, 16,
        
        # Additional positive
        17, 13, 21, 17, 13, 21, 17, 13,
        
        # Negative
        2, 25, 9, 2, 25, 14, 14, 2,
        
        # Complex
        16, 18, 6, 6, 16, 18, 6, 25
    ],
    'emotion_name': [
        # Original
        'joy', 'anger', 'sadness', 'excitement', 'fear', 'pride', 'sadness', 'grief',
        
        # Positive
        'joy', 'excitement', 'pride', 'joy', 'excitement', 'pride', 'joy', 'excitement',
        
        # Negative
        'anger', 'sadness', 'disappointment', 'anger', 'sadness', 'fear', 'fear', 'anger',
        
        # Complex
        'grief', 'love', 'confusion', 'confusion', 'grief', 'love', 'confusion', 'sadness'
    ]
})

# Update our emotion_data
emotion_data = expanded_data
print(f"Now using expanded dataset with {len(emotion_data)} examples")
print("Emotion distribution:")
print(emotion_data['emotion_name'].value_counts())

# Rebuild persona datasets with the expanded data
persona_datasets = {}
for name, config in personas.items():
    persona_data = emotion_data[emotion_data['emotion'].isin(config['emotions'])]
    embeddings = model.encode(persona_data['text'].tolist())
    persona_datasets[name] = {
        'texts': persona_data['text'].tolist(),
        'embeddings': embeddings,
        'config': config
    }
# ========== 2. PERSONA SETUP ==========
personas = {
    "Sales Hustler": {
        "traits": "Energetic, persuasive, optimistic",
        "emotions": [17, 13, 21],
        "responses": [
            "💼 Hustler says: {} Remember - every 'no' brings you closer to a 'yes'!",
            "💼 Hustler says: {} That's the spirit! Now go crush your goals!",
            "💼 Hustler says: {} Pro tip: The fortune is in the follow-up!",
            "💼 Hustler says: {} Winners focus on winning, losers focus on winners!",
            "💼 Hustler says: {} Opportunity dances with those already on the dance floor!"
        ],
        "color": "🟡",
        "examples": ["motivation", "sales rejection", "goal setting"]
    },
    "The Burnout": {
        "traits": "Empathetic, understanding, been through stress",
        "emotions": [2, 25, 14],
        "responses": [
            "🛌 Burnout buddy says: {} It's okay to rest - even volcanoes sleep sometimes.",
            "🛌 Burnout buddy says: {} Remember: You can't pour from an empty cup.",
            "🛌 Burnout buddy says: {} Small steps still move you forward.",
            "🛌 Burnout buddy says: {} Your worth isn't measured by productivity.",
            "🛌 Burnout buddy says: {} Healing isn't linear - be gentle with yourself."
        ],
        "color": "🔵",
        "examples": ["exhaustion", "work stress", "overwhelm"]
    },
    "Broken Hearted": {
        "traits": "Compassionate, emotional, healing",
        "emotions": [25, 16, 9],
        "responses": [
            "💔 Heart friend says: {} The cracks let the light in, my dear.",
            "💔 Heart friend says: {} This pain is part of your love story.",
            "💔 Heart friend says: {} Hearts grow back stronger where they break.",
            "💔 Heart friend says: {} Some souls just take longer to heal.",
            "💔 Heart friend says: {} Your heart knows its own healing timeline."
        ],
        "color": "🟣",
        "examples": ["breakups", "grief", "loneliness"]
    },
    "Warrior": {
        "traits": "Resilient, strong, battling challenges",
        "emotions": [17, 20, 4],
        "responses": [
            "🛡️ Warrior says: {} You're stronger than you think - keep fighting!",
            "🛡️ Warrior says: {} Battles shape warriors - you're being forged!",
            "🛡️ Warrior says: {} The strongest steel goes through the hottest fire.",
            "🛡️ Warrior says: {} Your struggle is part of your strength story.",
            "🛡️ Warrior says: {} Every scar is proof you're still standing."
        ],
        "color": "🟢",
        "examples": ["illness", "recovery", "life challenges"]
    }
}



# ========== 3. MODEL SETUP ==========
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully")

# Prepare persona datasets
persona_datasets = {}
for name, config in personas.items():
    persona_data = emotion_data[emotion_data['emotion'].isin(config['emotions'])]
    embeddings = model.encode(persona_data['text'].tolist())
    persona_datasets[name] = {
        'texts': persona_data['text'].tolist(),
        'embeddings': embeddings,
        'config': config
    }

# ========== 4. RESPONSE GENERATION ==========
def generate_response(user_input, persona_name):
    persona = persona_datasets[persona_name]
    input_embedding = model.encode([user_input])
    
    # Find most similar text
    similarities = cosine_similarity(input_embedding, persona['embeddings'])
    best_match_idx = np.argmax(similarities)
    best_response = persona['texts'][best_match_idx]
    
    # Get the persona's response templates
    response_templates = personas[persona_name]['responses']  # Changed from 'response_template' to 'responses'
    
    # Select a random template and format it

    chosen_template = np.random.choice(response_templates)
    return chosen_template.format("")  # Empty string instead of the matched text

# ========== 5. CHAT INTERFACE ==========
def chat_interface():
    print("✨ Welcome to 1MinMojo! Your quick emotional support system ✨\n")
    print("Choose a persona to talk to:\n")
    
    for i, (name, config) in enumerate(personas.items(), 1):
        # Safe printing with default values
        color = config.get('color', '⚪')
        traits = config.get('traits', 'No description available')
        
        print(f"{i}. {color} {name}: {traits}")
        
        # Only print examples if they exist
        if 'examples' in config:
            print(f"   Best for: {', '.join(config['examples'])}")
        print()  # Empty line between personas
    
    while True:
        try:
            choice = input("Enter persona number (1-4) or 'q' to quit: ")
            if choice.lower() == 'q':
                break
                
            persona_name = list(personas.keys())[int(choice)-1]
            persona_color = personas[persona_name].get('color', '⚪')
            print(f"\n{persona_color} You're now talking to: {persona_name}")
            print("Type your message below (or 'back' to choose another persona):\n")
            
            while True:
                user_input = input("You: ")
                if user_input.lower() in ['back', 'quit', 'exit']:
                    break
                
                response = generate_response(user_input, persona_name)
                print(f"\n{persona_name}: {response}\n")
                
        except (ValueError, IndexError):
            print("Please enter a number between 1-4")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

# Start the chat
chat_interface()
