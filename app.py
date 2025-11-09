import streamlit as st
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import os
from preprocess import CONFIG, GENRES
from train import MusicGenreNet

# Page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1DB954 0%, #191414 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1ed760;
    }
    .genre-card {
        padding: 1rem;
        border-radius: 10px;
        background: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MusicGenreNet(num_classes=len(GENRES)).to(device)
    model.load_state_dict(torch.load('models/genre_model.pth', map_location=device))
    model.eval()
    return model, device

def extract_features(audio_file):
    """Extract mel-spectrogram from audio file"""
    try:
        y, sr = librosa.load(audio_file, sr=CONFIG['sample_rate'], duration=CONFIG['duration'])
        
        # Pad if too short
        if len(y) < sr * CONFIG['duration']:
            y = np.pad(y, (0, sr * CONFIG['duration'] - len(y)))
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels']
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        mel_spec_db = 2 * (mel_spec_db - min_val) / (max_val - min_val) - 1
        
        return mel_spec_db, y, sr
    
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

def predict_genre(model, device, mel_spec):
    """Predict genre from mel-spectrogram"""
    mel_spec_tensor = torch.FloatTensor(mel_spec[np.newaxis, np.newaxis, ...]).to(device)
    
    with torch.no_grad():
        output = model(mel_spec_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
    
    return probabilities

def plot_spectrogram(mel_spec):
    """Plot mel-spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        mel_spec, 
        sr=CONFIG['sample_rate'],
        hop_length=CONFIG['hop_length'],
        x_axis='time', 
        y_axis='mel',
        ax=ax,
        cmap='viridis'
    )
    ax.set_title('Mel-Spectrogram', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    return fig

def plot_predictions(probabilities):
    """Plot prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by probability
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_genres = [GENRES[i] for i in sorted_indices]
    sorted_probs = probabilities[sorted_indices]
    
    colors = ['#1DB954' if i == 0 else '#191414' for i in range(len(GENRES))]
    
    bars = ax.barh(sorted_genres, sorted_probs * 100, color=colors)
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Genre Prediction Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Main UI
st.markdown('<h1 class="main-header">🎵 Music Genre Classifier</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/musical-notes.png", width=80)
    st.title("About")
    st.info("""
    This app uses Deep Learning (CNN) to classify music into 10 genres:
    
    🎼 Blues | Classical | Country  
    🕺 Disco | Hip-hop | Jazz  
    🤘 Metal | Pop | Reggae | Rock
    
    **Model Accuracy:** 86%  
    **Training Time:** 99 epochs  
    **Framework:** PyTorch
    """)
    
    st.markdown("---")
    st.subheader("Supported Formats")
    st.write("- WAV (.wav)")
    st.write("- MP3 (.mp3)")
    st.write("- FLAC (.flac)")
    
    st.markdown("---")
    st.caption("Built with Streamlit 🎈 & PyTorch 🔥")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'flac'],
        help="Upload a 30-second audio clip for best results"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Add predict button
        predict_button = st.button("🎯 Predict Genre", use_container_width=True)

with col2:
    st.subheader("📊 Results")
    
    if uploaded_file is not None and predict_button:
        with st.spinner('🎵 Analyzing audio...'):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load model
            model, device = load_model()
            
            # Extract features
            mel_spec, audio, sr = extract_features(temp_path)
            
            if mel_spec is not None:
                # Predict
                probabilities = predict_genre(model, device, mel_spec)
                predicted_idx = np.argmax(probabilities)
                predicted_genre = GENRES[predicted_idx]
                confidence = probabilities[predicted_idx]
                
                # Display main prediction
                st.success(f"🎼 **Predicted Genre:** {predicted_genre.upper()}")
                st.metric("Confidence", f"{confidence*100:.2f}%")
                
                # Display top 3 predictions
                st.markdown("#### Top 3 Predictions:")
                top3_indices = np.argsort(probabilities)[::-1][:3]
                
                for i, idx in enumerate(top3_indices):
                    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                    st.markdown(f"""
                    <div class="genre-card">
                        {emoji} <b>{GENRES[idx].capitalize()}</b>: {probabilities[idx]*100:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
            
            # Clean up
            os.remove(temp_path)
    
    elif uploaded_file is None:
        st.info("👆 Upload an audio file to get started!")

# Visualization section
if uploaded_file is not None and predict_button:
    st.markdown("---")
    st.subheader("📈 Detailed Analysis")
    
    tab1, tab2 = st.tabs(["📊 Prediction Chart", "🎨 Spectrogram"])
    
    with tab1:
        st.pyplot(plot_predictions(probabilities))
        
        # Add explanation
        with st.expander("ℹ️ How to interpret this chart"):
            st.write("""
            - **Green bar**: Most likely genre (highest confidence)
            - **Black bars**: Other possible genres
            - The longer the bar, the more confident the model is
            - A confidence > 80% indicates high certainty
            """)
    
    with tab2:
        st.pyplot(plot_spectrogram(mel_spec))
        
        with st.expander("ℹ️ What is a Mel-Spectrogram?"):
            st.write("""
            A mel-spectrogram is a visual representation of audio:
            - **X-axis**: Time (seconds)
            - **Y-axis**: Frequency (Hz)
            - **Color**: Intensity (dB) - brighter = louder
            
            The model analyzes this image to determine the genre!
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Made using Deep Learning | Model accuracy: 86% on test set</p>
    <p><small>Trained on GTZAN Dataset (1000 songs, 10 genres)</small></p>
</div>
""", unsafe_allow_html=True)
