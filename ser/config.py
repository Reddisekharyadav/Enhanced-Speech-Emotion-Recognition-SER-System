"""
Configuration Module for Speech Emotion Recognition (SER) Tool

This module provides a central configuration class for the SER application. It defines
various settings and parameters used throughout the tool, including emotions, feature
extraction configuration, neural network parameters, audio file read parameters, dataset
configuration, model configuration, transcript configuration, and default language settings.

Classes:
    - Config: Contains all configuration settings for the SER application.

  
"""

import os
from typing import Any
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


class Config:
    """
    Central configuration class for the SER application.
    """

    # Emotions supported by the dataset
    EMOTIONS: dict[str, str] = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    # Temporary folder for processing
    TMP_FOLDER: str = "./tmp"

    # Default feature extraction configuration (tuned for speed by default)
    DEFAULT_FEATURE_CONFIG: dict[str, bool] = {
        "mfcc": True,
        "chroma": True,
        "mel": True,
        # Heavy features disabled by default for faster dataset preprocessing
        "contrast": False,
        "tonnetz": False,
    }

    # Feature extraction parameters (speed vs quality knobs)
    FEATURE_PARAMS: dict[str, int] = {
        # Smaller FFT and fewer MFCCs speed up preprocessing significantly
        "n_fft": 1024,
        "hop_length": 512,  # optional; librosa default if omitted
        "n_mfcc": 24,
    }

    # Neural network parameters for MLP Classifier
    NN_PARAMS: dict[str, Any] = {
        "alpha": 0.01,
        "batch_size": 256,
        "epsilon": 1e-08,
        "hidden_layer_sizes": (300,),
        "learning_rate": "adaptive",
        "max_iter": 500,
    }

    # Audio file read parameters
    AUDIO_READ_CONFIG: dict[str, int] = {
        "max_retries": 3,
        "retry_delay": 1,  # in seconds
    }

    # Dataset configuration
    # Resolve dataset folder to an absolute path by default (relative to this file's directory)
    _DEFAULT_DATASET_FOLDER = os.getenv("DATASET_FOLDER")
    if not _DEFAULT_DATASET_FOLDER:
        _SER_DIR = os.path.dirname(__file__)
        _DEFAULT_DATASET_FOLDER = os.path.abspath(os.path.join(_SER_DIR, "features", "dataset", "ravdess"))

    DATASET: dict[str, str] = {
        "folder": _DEFAULT_DATASET_FOLDER,
        "subfolder_prefix": "Actor_*",
        "extension": "*.wav",
    }

        # Model configuration
    MODELS_CONFIG: dict[str, Any] = {
    "models_folder": "./speech-emotion-recognition/ser/models",
        "whisper_model": {"name": "tiny", "path": "OpenAI/whisper/"},
        # Cap workers to avoid Windows I/O thrash
        "num_cores": min(8, os.cpu_count() or 4),
        # Optional: sample subset for faster first training (0 disables)
        "max_train_files": 0,
        "enhanced_model_path": "ser/models/enhanced_ser_model.pkl",
        "scaler_path": "ser/models/feature_scaler.pkl",
        "fast_model_path": "enhanced_emotion_model_fast.pkl",
        "fast_scaler_path": "enhanced_scaler_fast.pkl",
    }

    # Feature cache directory (speeds up repeated trainings)
    FEATURES_CACHE_DIR: str = os.path.join("ser", "features", ".cache")

    # Transcript configuration
    TIMELINE_CONFIG: dict[str, str] = {"folder": "./transcripts"}

    # Real-time audio processing configuration
    REALTIME_CONFIG: dict[str, Any] = {
        "chunk_duration": 3.0,  # seconds
        "sample_rate": 16000,
        "channels": 1,
        "confidence_threshold": 0.6,
        "emotion_smoothing_window": 5,  # number of chunks for smoothing
    }

    # AI Chat responses based on emotions
    AI_RESPONSES: dict[str, list[str]] = {
        "happy": [
            "That's wonderful! I can hear the joy in your voice! 😊",
            "You sound really cheerful! What's making you so happy?",
            "I love your positive energy! Keep that smile going! ✨",
        ],
        "sad": [
            "I can sense you're feeling down. Would you like to talk about it? 💙",
            "Your voice sounds a bit sad. I'm here to listen if you need support.",
            "Sometimes it's okay to feel sad. Take your time, I'm here for you. 🤗",
        ],
        "angry": [
            "I can hear some frustration in your voice. Take a deep breath with me. 😌",
            "It sounds like something is bothering you. Want to share what's wrong?",
            "Let's try to work through this together. What's causing this anger?",
        ],
        "fearful": [
            "You sound a bit anxious. Everything will be okay. 🌟",
            "I can sense some worry in your voice. What's concerning you?",
            "Take a moment to breathe. I'm here to help you feel more at ease.",
        ],
        "surprised": [
            "Wow! You sound surprised! What happened? 😮",
            "That sounds like quite a surprise! Tell me more!",
            "I can hear the amazement in your voice! What's got you so surprised?",
        ],
        "neutral": [
            "I'm listening. How are you feeling today?",
            "What's on your mind?",
            "I'm here for you. What would you like to talk about?",
        ],
        "calm": [
            "You sound very peaceful. That's nice to hear. ☮️",
            "Your calm energy is wonderful. How are you doing?",
            "I can sense the tranquility in your voice. What's bringing you peace?",
        ],
        "disgust": [
            "Something seems to be bothering you. What's wrong?",
            "I can sense displeasure in your voice. Want to talk about it?",
            "What's causing that reaction? I'm here to listen.",
        ],
    }

    # Sample audio files for testing different emotions
    SAMPLE_AUDIO_URLS: dict[str, str] = {
        "happy": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav",
        "sad": "https://www.soundjay.com/misc/sounds/rain-02.wav",
        "angry": "https://www.soundjay.com/misc/sounds/thunder-01.wav",
        "neutral": "https://www.soundjay.com/misc/sounds/chimes-01.wav",
    }

    # GUI Configuration
    GUI_CONFIG: dict[str, Any] = {
        "window_title": "🎙️ Advanced Speech Emotion Recognition",
        "window_size": (1200, 800),
        "theme_colors": {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#F18F01",
            "warning": "#C73E1D",
            "background": "#F5F5F5",
        }
    }

    # Language settings
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")
    FILE_SETTING: None | str = None
    TRAIN_MODE: bool = False
