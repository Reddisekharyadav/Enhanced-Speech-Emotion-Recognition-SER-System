"""
Enhanced Speech Emotion Recognition (SER) - Streamlit Web App
Exact replica of the PyQt5 GUI with all features
Deploy to Streamlit Cloud for a live demo link
"""

import streamlit as st
import numpy as np
import tempfile
import os
import librosa
import soundfile as sf
from pathlib import Path
import time
import traceback
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

# Import your SER modules
try:
    from ser.models.enhanced_emotion_model import predict_emotion_realtime, train_enhanced_model
    from ser.config import Config
    from ser.utils import get_logger
    SER_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ SER modules import warning: {e}")
    SER_AVAILABLE = False

# Initialize logger
try:
    logger = get_logger(__name__)
except:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Constants from GUI
SAMPLE_RATE = 22050
CHUNK_DURATION = 2.0
CHANNELS = 1

# Page configuration
st.set_page_config(
    page_title="Enhanced Speech Emotion Recognition (SER) System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching PyQt5 GUI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .emotion-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: white;
        border: 2px solid #2E86AB;
        color: #2E86AB;
        text-align: center;
        font-size: 1.8rem;
        margin: 1rem 0;
        font-weight: bold;
    }
    .confidence-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1F5F79;
    }
    .chat-message {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid;
    }
    .transcript-box {
        background: #FFFFFF;
        border: 1px solid #CCC;
        border-radius: 6px;
        padding: 10px;
        max-height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Emotion emoji mapping
EMOTION_EMOJIS = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "surprised": "😲",
    "neutral": "😐",
    "calm": "😌",
    "disgust": "🤢"
}

def get_ai_response(emotion: str) -> str:
    """Return AI response based on detected emotion - same as PyQt5 GUI."""
    responses = {
        "happy": "I'm glad to hear you're feeling happy! 😊",
        "sad": "I'm here for you. If you want to talk, I'm listening. 😢",
        "angry": "It's okay to feel angry sometimes. Take a deep breath. 😠",
        "fearful": "If something is worrying you, remember you're not alone. 😨",
        "surprised": "Wow, that sounds surprising! 😲",
        "neutral": "Let me know if you want to share more. 😐",
        "calm": "It's great to feel calm and relaxed. 😌",
        "disgust": "If something bothers you, I'm here to listen. 🤢"
    }
    return responses.get(emotion, "I'm here to support you, whatever you're feeling.")

def update_emotion_visualization(emotion: str, confidence: float):
    """Update emotion display and history."""
    # Add to history
    st.session_state.emotion_history.append(emotion)
    st.session_state.confidence_history.append(confidence)
    st.session_state.current_emotion = emotion
    st.session_state.current_confidence = confidence
    
    # Keep only recent history (last 50)
    if len(st.session_state.emotion_history) > 50:
        st.session_state.emotion_history.pop(0)
        st.session_state.confidence_history.pop(0)

def plot_emotion_history():
    """Create emotion distribution chart - same as PyQt5 GUI."""
    if len(st.session_state.emotion_history) == 0:
        return None
    
    # Count emotions in recent history (last 20)
    recent_emotions = st.session_state.emotion_history[-20:]
    emotion_counts = {}
    for emotion in recent_emotions:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    if not emotion_counts:
        return None
    
    # Color map matching PyQt5 GUI
    color_map = {
        "happy": "#FFD700", "sad": "#4169E1", "angry": "#DC143C",
        "fearful": "#FF8C00", "surprised": "#FF1493", "neutral": "#808080",
        "calm": "#20B2AA", "disgust": "#9932CC"
    }
    
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    colors = [color_map.get(emotion, "#808080") for emotion in emotions]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(emotions, counts, color=colors, alpha=0.7)
    ax.set_title("Recent Emotion Distribution", fontsize=14, fontweight='bold')
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def predict_emotion_from_file(audio_path: str):
    """Predict emotion from audio file - same logic as PyQt5 GUI."""
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file missing: {audio_path}")
        
        logger.info(f"Predicting emotion for: {audio_path}")
        result = predict_emotion_realtime(audio_path)
        
        # Handle result format
        if isinstance(result, dict):
            if "error" in result:
                logger.error(f"Prediction error: {result['error']}")
                return None, None
            elif "label" in result and "confidence" in result:
                return result["label"], result["confidence"]
        elif isinstance(result, (list, tuple)) and len(result) >= 2:
            return str(result[0]), float(result[1])
        else:
            return str(result), 0.75
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        st.error(f"Error: {str(e)}")
        return None, None

def create_sample_audio(emotion: str) -> str:
    """Create sample audio file - same as PyQt5 GUI."""
    try:
        samples_dir = "samples"
        os.makedirs(samples_dir, exist_ok=True)
        
        duration, sr = 3.0, 16000
        freq = {"happy": 440, "sad": 220, "angry": 660, "neutral": 330}.get(emotion, 330)
        t = np.linspace(0, duration, int(sr * duration), False)
        rng = np.random.default_rng(seed=42)
        audio_data = np.sin(2 * np.pi * freq * t) * 0.3 + rng.normal(0, 0.05, int(sr * duration))
        
        filename = os.path.join(samples_dir, f"{emotion}_sample.wav")
        wav.write(filename, sr, (audio_data * 32767).astype(np.int16))
        
        return filename
    except Exception as e:
        logger.error(f"Failed creating sample {emotion}: {e}")
        return None

def predict_emotion_from_file(audio_path: str):
    """Predict emotion from uploaded audio file."""
    
    if not SER_AVAILABLE:
        st.warning("⚠️ Running in DEMO mode. Install dependencies for real predictions.")
        # Demo mode: analyze audio properties for basic emotion guess
        try:
            y, sr = librosa.load(audio_path, duration=10)
            # Use audio features for demo prediction
            energy = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            tempo = librosa.beat.tempo(y=y, sr=sr)[0] if len(y) > 0 else 100
            
            # Simple heuristic for demo
            if energy > 0.1 and tempo > 120:
                return "happy", 0.65
            elif energy < 0.05:
                return "calm", 0.70
            elif zcr > 0.1:
                return "angry", 0.60
            else:
                return "neutral", 0.75
        except Exception as e:
            logger.error(f"Demo mode error: {e}")
            return "neutral", 0.50
    
    try:
        # Import here to catch any import errors
        import pickle
        import os
        from ser.config import Config
        
        # Try to load model and predict
        result = predict_emotion_realtime(audio_path)
        
        # Handle different return types
        if isinstance(result, dict):
            if "error" in result:
                error_msg = result["error"]
                st.error(f"Model error: {error_msg}")
                logger.error(f"Prediction error: {error_msg}")
                # Try demo mode as fallback
                st.info("🔄 Falling back to demo mode...")
                return predict_emotion_from_file.__wrapped__(audio_path) if hasattr(predict_emotion_from_file, '__wrapped__') else ("neutral", 0.5)
            elif "label" in result and "confidence" in result:
                return result["label"], result["confidence"]
        
        # Fallback: treat as string label
        return str(result), 0.75
        
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found. Please ensure trained models are in 'ser/models/' directory.")
        logger.error(f"Model file error: {e}")
        st.info("💡 Tip: Make sure your .keras and .pkl model files are committed to GitHub")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    # Initialize session state variables at the very beginning
    if "current_emotion" not in st.session_state:
        st.session_state.current_emotion = "neutral"
    
    if "current_confidence" not in st.session_state:
        st.session_state.current_confidence = 0.0
    
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []
    
    if "confidence_history" not in st.session_state:
        st.session_state.confidence_history = []
    
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []
    
    if "transcript_history" not in st.session_state:
        st.session_state.transcript_history = []
    
    if "training_logs" not in st.session_state:
        st.session_state.training_logs = []
    
    # Header - matching PyQt5 GUI
    st.markdown('<h1 class="main-header">🎙️ Enhanced Speech Emotion Recognition (SER) System</h1>', unsafe_allow_html=True)
    
    # Sidebar - matching PyQt5 sensitivity control
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/voice-recognition.png", width=80)
        st.title("⚙️ Settings")
        
        sensitivity = st.slider(
            "Sensitivity",
            min_value=1,
            max_value=10,
            value=6,
            key="sensitivity_slider_main",
            help="Adjust emotion detection threshold (higher = more sensitive)"
        )
        
        confidence_threshold = (11 - sensitivity) / 10.0 * 0.6
        
        st.markdown("---")
        st.markdown("### 📊 Supported Emotions")
        emotions_text = """
        - 😊 Happy
        - 😢 Sad
        - 😠 Angry
        - 😨 Fearful
        - 😲 Surprised
        - 😐 Neutral
        - 😌 Calm
        - 🤢 Disgust
        """
        st.info(emotions_text)
    
    # Main tabs - exactly like PyQt5 GUI
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎙️ Real-time",
        "📁 File Analysis", 
        "🎵 Sample Audio",
        "🔧 Model Training"
    ])
    
    # Tab 1: Real-time Analysis (matching PyQt5 create_realtime_tab)
    with tab1:
        st.header("🎙️ Real-time Emotion Detection")
        
        # Add audio recorder
        st.markdown("### 🎤 Record Your Voice")
        st.info("💡 Click the microphone button below to record your voice and analyze emotions in real-time!")
        
        # Try to import audio recorder
        try:
            from audio_recorder_streamlit import audio_recorder
            
            col_record1, col_record2 = st.columns([2, 1])
            
            with col_record1:
                # Audio recorder component
                audio_bytes = audio_recorder(
                    text="Click to Record",
                    recording_color="#e74c3c",
                    neutral_color="#2E86AB",
                    icon_name="microphone",
                    icon_size="3x",
                    key="audio_recorder"
                )
                
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Analyze recorded audio
                    with st.spinner("🔄 Analyzing your voice..."):
                        # Save audio bytes to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        try:
                            # Predict emotion
                            emotion, confidence = predict_emotion_from_file(tmp_path)
                            
                            if emotion and confidence:
                                # Update visualization
                                update_emotion_visualization(emotion, confidence)
                                
                                st.success(f"✅ Detected: **{emotion.upper()}** ({confidence:.1%} confidence)")
                                
                                # AI Response
                                ai_response = get_ai_response(emotion)
                                st.info(f"🤖 {ai_response}")
                                
                                # Add to chat history
                                timestamp = time.strftime("%H:%M:%S")
                                emotion_colors = {
                                    "happy": "#4CAF50", "sad": "#2196F3", "angry": "#F44336",
                                    "fearful": "#FF9800", "surprised": "#E91E63", "neutral": "#9E9E9E",
                                    "calm": "#00BCD4", "disgust": "#9C27B0"
                                }
                                color = emotion_colors.get(emotion, "#9E9E9E")
                                
                                chat_msg = f"""
                                <div class="chat-message" style="border-left-color: {color}; background-color: {color}20;">
                                    <b style="color: {color};">🎤 Live Recording:</b> {emotion.title()} ({confidence:.0%} confidence)<br>
                                    <b>🤖 AI Response:</b> {ai_response}<br>
                                    <small style="color: #666;">Time: {timestamp}</small>
                                </div>
                                """
                                st.session_state.ai_chat_history.append(chat_msg)
                            else:
                                st.warning("⚠️ Could not detect emotion. Try speaking more clearly.")
                        
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                        
                        finally:
                            # Clean up
                            if os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
            
            with col_record2:
                st.markdown("**📝 Tips:**")
                st.markdown("""
                - Speak clearly for 2-5 seconds
                - Express your emotion naturally
                - Ensure quiet environment
                - Allow microphone access
                """)
        
        except ImportError:
            st.warning("⚠️ Audio recorder not available. Install with: `pip install audio-recorder-streamlit`")
            st.info("💡 **Alternative:** Use the **File Analysis** tab to upload pre-recorded audio files.")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🎭 Emotion Visualization")
            
            # Current emotion display
            emoji_map = {
                "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😨",
                "surprised": "😲", "neutral": "😐", "calm": "😌", "disgust": "🤢"
            }
            emoji = emoji_map.get(st.session_state.current_emotion, "🎭")
            
            st.markdown(f"""
            <div class="emotion-box">
                {emoji} Current Emotion: {st.session_state.current_emotion.title()}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence display
            st.markdown("**Confidence:**")
            confidence_percent = int(st.session_state.current_confidence * 100)
            st.progress(st.session_state.current_confidence)
            st.write(f"**{confidence_percent}%**")
            
            # Emotion history chart
            if len(st.session_state.emotion_history) > 0:
                fig = plot_emotion_history()
                if fig:
                    st.pyplot(fig)
        
        with col2:
            st.markdown("### 💬 AI Emotional Chat")
            
            # Chat display area
            chat_container = st.container()
            with chat_container:
                if not st.session_state.ai_chat_history:
                    st.info("🤖 **AI Assistant:** Hello! I'm here to respond to your emotions in real-time. Start recording or load an audio file to begin our conversation!")
                else:
                    for msg in st.session_state.ai_chat_history[-10:]:  # Show last 10 messages
                        st.markdown(msg, unsafe_allow_html=True)
        
        # Transcription section
        st.markdown("---")
        st.markdown("### 📝 Transcribed Text (per chunk)")
        transcript_container = st.container()
        with transcript_container:
            if st.session_state.transcript_history:
                st.markdown('<div class="transcript-box">', unsafe_allow_html=True)
                for transcript in st.session_state.transcript_history[-20:]:
                    st.text(transcript)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.text("[No transcription yet]")
    
    # Tab 2: File Analysis (matching PyQt5 create_file_tab)
    with tab2:
        st.header("📁 File Analysis")
        st.write("Upload an audio file (.wav, .mp3, .flac) to analyze the emotion")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="Supported formats: WAV, MP3, FLAC, OGG, M4A",
            key="file_upload"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.audio(uploaded_file, format='audio/wav')
                
                st.markdown("**File Details:**")
                st.write(f"- **Filename:** {uploaded_file.name}")
                st.write(f"- **Size:** {uploaded_file.size / 1024:.2f} KB")
            
            with col2:
                if st.button("🔍 Analyze Emotion", key="analyze_btn", use_container_width=True):
                    with st.spinner("🔄 Analyzing emotion... Please wait..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Predict emotion
                            emotion, confidence = predict_emotion_from_file(tmp_path)
                            
                            if emotion and confidence:
                                # Update visualization
                                update_emotion_visualization(emotion, confidence)
                                
                                # Display results
                                emoji = emoji_map.get(emotion.lower(), "🎭")
                                
                                st.markdown(f"""
                                <div class="emotion-box">
                                    {emoji} <b>{emotion.upper()}</b>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Confidence meter
                                st.markdown("**Confidence Level:**")
                                st.progress(confidence)
                                st.write(f"**{confidence*100:.1f}%** confidence")
                                
                                if confidence >= confidence_threshold:
                                    st.success("✅ High confidence prediction!")
                                    
                                    # AI Response
                                    ai_response = get_ai_response(emotion)
                                    st.markdown("### 🤖 AI Response")
                                    st.info(ai_response)
                                    
                                    # Add to chat history
                                    timestamp = time.strftime("%H:%M:%S")
                                    emotion_colors = {
                                        "happy": "#4CAF50", "sad": "#2196F3", "angry": "#F44336",
                                        "fearful": "#FF9800", "surprised": "#E91E63", "neutral": "#9E9E9E",
                                        "calm": "#00BCD4", "disgust": "#9C27B0"
                                    }
                                    color = emotion_colors.get(emotion, "#9E9E9E")
                                    
                                    chat_msg = f"""
                                    <div class="chat-message" style="border-left-color: {color}; background-color: {color}20;">
                                        <b style="color: {color};">🎭 Detected Emotion:</b> {emotion.title()} ({confidence:.0%} confidence)<br>
                                        <b>🤖 AI Response:</b> {ai_response}<br>
                                        <small style="color: #666;">Time: {timestamp}</small>
                                    </div>
                                    """
                                    st.session_state.ai_chat_history.append(chat_msg)
                                else:
                                    st.warning(f"⚠️ Confidence below threshold ({confidence_threshold*100:.0f}%). Results may be uncertain.")
                                
                                # Emotion analysis
                                st.markdown("---")
                                st.markdown("### 📊 Emotion Analysis")
                                st.write(f"**Detected Emotion:** {emotion.title()}")
                                st.write(f"**Confidence Score:** {confidence:.3f}")
                                
                                # Show chart
                                fig = plot_emotion_history()
                                if fig:
                                    st.pyplot(fig)
                                
                            else:
                                st.error("❌ Failed to analyze emotion.")
                                with st.expander("💡 Troubleshooting Tips"):
                                    st.markdown("""
                                    **Possible reasons:**
                                    - Model files not found in `ser/models/` directory
                                    - Audio file format not supported
                                    - Audio quality too poor
                                    
                                    **Solutions:**
                                    - Ensure `.keras` and `.pkl` model files are in `ser/models/`
                                    - Try a different audio file (clear speech, < 10MB)
                                    - Check deployment logs for detailed errors
                                    """)
                        
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            with st.expander("🔍 View Error Details"):
                                st.code(traceback.format_exc())
                        
                        finally:
                            # Clean up temporary file
                            if os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
    
    # Tab 3: Sample Audio (matching PyQt5 create_samples_tab)
    with tab3:
        st.header("🎵 Sample Audio Generation")
        st.info("🎵 Download sample audio files to test different emotions.")
        
        st.markdown("### Generate Samples")
        col1, col2, col3, col4 = st.columns(4)
        
        emotions_to_generate = ["happy", "sad", "angry", "neutral"]
        cols = [col1, col2, col3, col4]
        
        for emotion, col in zip(emotions_to_generate, cols):
            with col:
                if st.button(f"📥 {emotion.title()}", key=f"gen_{emotion}", use_container_width=True):
                    with st.spinner(f"Creating {emotion} sample..."):
                        filename = create_sample_audio(emotion)
                        if filename:
                            st.success(f"✅ Created {emotion} sample!")
                            
                            # Provide download link
                            with open(filename, "rb") as f:
                                st.download_button(
                                    label=f"⬇️ Download {emotion}_sample.wav",
                                    data=f,
                                    file_name=f"{emotion}_sample.wav",
                                    mime="audio/wav",
                                    key=f"download_{emotion}"
                                )
                        else:
                            st.error(f"❌ Failed to create {emotion} sample")
        
        st.markdown("---")
        st.markdown("### Sample Files Status")
        
        samples_dir = "samples"
        if os.path.exists(samples_dir):
            sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.wav')]
            if sample_files:
                st.success(f"✅ {len(sample_files)} sample(s) created")
                for sample_file in sample_files:
                    st.text(f"✅ {sample_file}")
            else:
                st.info("No samples created yet. Click buttons above to generate.")
        else:
            st.info("Ready to create samples. Click buttons above.")
    
    # Tab 4: Model Training (matching PyQt5 create_training_tab)
    with tab4:
        st.header("🔧 Model Training")
        st.info("🔧 **Instructions:** Ensure RAVDESS dataset is in `ser/features/dataset/ravdess/`")
        
        # Model status check
        try:
            if SER_AVAILABLE:
                model_path = Config.MODELS_CONFIG.get("enhanced_model_path", "")
                if model_path and os.path.exists(model_path):
                    st.success("✅ Enhanced model found!")
                    train_button_text = "🔄 Retrain Model"
                else:
                    st.warning("⚠️ No trained model found. Please train a model.")
                    train_button_text = "🚀 Start Training"
            else:
                st.error("⚠️ SER modules not available. Cannot train model.")
                train_button_text = "❌ Training Unavailable"
        except:
            st.warning("⚠️ Could not check model status.")
            train_button_text = "🚀 Start Training"
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button(train_button_text, key="train_btn", use_container_width=True, disabled=not SER_AVAILABLE):
                st.session_state.training_logs = []
                st.session_state.training_logs.append("🚀 Starting model training...")
                
                try:
                    with st.spinner("📊 Training in progress... This may take several minutes..."):
                        st.session_state.training_logs.append("📊 Loading dataset...")
                        
                        # Check for dataset
                        try:
                            from glob import glob
                            base = Config.DATASET["folder"]
                            pattern = os.path.join(base, Config.DATASET["subfolder_prefix"], Config.DATASET["extension"])
                            files = glob(pattern)
                            
                            if not files:
                                st.error(f"❌ Dataset not found at {os.path.abspath(base)}")
                                st.session_state.training_logs.append(f"❌ No dataset files found")
                            else:
                                st.session_state.training_logs.append(f"✅ Found {len(files)} audio files")
                                st.session_state.training_logs.append("🔄 Starting training process...")
                                
                                # Train model
                                results = train_enhanced_model()
                                
                                st.session_state.training_logs.append(f"✅ Training completed!")
                                st.session_state.training_logs.append(f"⏱️ Training time: {results.get('training_time', 0):.2f}s")
                                st.session_state.training_logs.append(f"🎯 Model accuracy: {results.get('accuracy', 0):.3f}")
                                
                                st.success("✅ Model training completed successfully!")
                                st.balloons()
                        except Exception as dataset_error:
                            st.error(f"❌ Dataset error: {str(dataset_error)}")
                            st.session_state.training_logs.append(f"❌ Error: {str(dataset_error)}")
                            
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.session_state.training_logs.append(f"❌ Training failed: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
        
        with col2:
            if st.button("🔍 Check Model Status", key="check_model_btn", use_container_width=True):
                st.session_state.training_logs.append("🔍 Checking model status...")
                
                try:
                    models_dir = "ser/models"
                    if os.path.exists(models_dir):
                        model_files = [f for f in os.listdir(models_dir) if f.endswith(('.keras', '.pkl'))]
                        if model_files:
                            st.session_state.training_logs.append(f"✅ Found {len(model_files)} model file(s):")
                            for mf in model_files:
                                size = os.path.getsize(os.path.join(models_dir, mf)) / (1024 * 1024)
                                st.session_state.training_logs.append(f"  - {mf} ({size:.2f} MB)")
                        else:
                            st.session_state.training_logs.append("⚠️ No model files found")
                    else:
                        st.session_state.training_logs.append("❌ Models directory not found")
                except Exception as e:
                    st.session_state.training_logs.append(f"❌ Error checking models: {str(e)}")
        
        # Training log display
        st.markdown("---")
        st.markdown("### 📋 Training Log")
        
        log_container = st.container()
        with log_container:
            if st.session_state.training_logs:
                log_text = "\n".join(st.session_state.training_logs)
                st.text_area("Logs", value=log_text, height=300, key="training_log_display")
            else:
                st.info("No training logs yet. Start training to see progress.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎭 Enhanced Speech Emotion Recognition System | Built with ❤️ using Streamlit</p>
        <p><small>GitHub: <a href="https://github.com/Reddisekharyadav/Enhanced-Speech-Emotion-Recognition-SER-System">Enhanced-SER-System</a></small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
