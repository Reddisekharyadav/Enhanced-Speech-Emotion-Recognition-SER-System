#!/usr/bin/env python3
"""
Sample Audio Generator for Enhanced Speech Emotion Recognition

This script creates sample audio files with different emotional characteristics
for testing the emotion recognition system.

Usage:
    python create_samples.py
"""

import os
import numpy as np
import scipy.io.wavfile as wav
from ser.utils import get_logger
from ser.config import Config

logger = get_logger(__name__)

def create_emotion_sample(emotion: str, filename: str, duration: float = 3.0):
    """
    Create a sample audio file with characteristics that might suggest an emotion.
    
    Note: These are simple synthetic audio samples for testing purposes.
    Real emotion recognition should be trained on actual human speech data.
    """
    sample_rate = Config.REALTIME_CONFIG["sample_rate"]
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Base parameters for different emotions
    emotion_params = {
        "happy": {
            "base_freq": 440,  # A4 note
            "freq_variation": 0.2,
            "amplitude": 0.7,
            "vibrato_rate": 6.0,
            "vibrato_depth": 0.1
        },
        "sad": {
            "base_freq": 220,  # A3 note (lower)
            "freq_variation": 0.05,
            "amplitude": 0.4,
            "vibrato_rate": 2.0,
            "vibrato_depth": 0.05
        },
        "angry": {
            "base_freq": 660,  # E5 note (higher)
            "freq_variation": 0.3,
            "amplitude": 0.9,
            "vibrato_rate": 8.0,
            "vibrato_depth": 0.2
        },
        "neutral": {
            "base_freq": 330,  # E4 note
            "freq_variation": 0.1,
            "amplitude": 0.5,
            "vibrato_rate": 4.0,
            "vibrato_depth": 0.05
        },
        "fearful": {
            "base_freq": 550,  # C#5 note
            "freq_variation": 0.4,
            "amplitude": 0.3,
            "vibrato_rate": 10.0,
            "vibrato_depth": 0.3
        },
        "surprised": {
            "base_freq": 880,  # A5 note (high)
            "freq_variation": 0.5,
            "amplitude": 0.8,
            "vibrato_rate": 12.0,
            "vibrato_depth": 0.4
        },
        "calm": {
            "base_freq": 200,  # G3 note (low and steady)
            "freq_variation": 0.02,
            "amplitude": 0.4,
            "vibrato_rate": 1.0,
            "vibrato_depth": 0.02
        },
        "disgust": {
            "base_freq": 150,  # Very low
            "freq_variation": 0.1,
            "amplitude": 0.6,
            "vibrato_rate": 3.0,
            "vibrato_depth": 0.1
        }
    }
    
    params = emotion_params.get(emotion, emotion_params["neutral"])
    
    # Generate base frequency with variation
    base_freq = params["base_freq"]
    freq_variation = params["freq_variation"]
    
    # Add frequency modulation (vibrato)
    vibrato = params["vibrato_depth"] * np.sin(2 * np.pi * params["vibrato_rate"] * t)
    frequency = base_freq * (1 + freq_variation * np.sin(2 * np.pi * 0.5 * t) + vibrato)
    
    # Generate the main tone
    audio_signal = params["amplitude"] * np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics for more realistic sound
    audio_signal += 0.3 * params["amplitude"] * np.sin(2 * np.pi * frequency * 2 * t)  # Second harmonic
    audio_signal += 0.1 * params["amplitude"] * np.sin(2 * np.pi * frequency * 3 * t)  # Third harmonic
    
    # Add some noise for realism
    noise_level = 0.02
    rng = np.random.default_rng(42)  # Use seed for reproducibility
    audio_signal += rng.normal(0, noise_level, audio_signal.shape)
    
    # Apply envelope (fade in/out)
    fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
    envelope = np.ones_like(audio_signal)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    audio_signal *= envelope
    
    # Normalize to prevent clipping
    audio_signal = audio_signal / np.max(np.abs(audio_signal)) * 0.8
    
    # Convert to 16-bit integers
    audio_int = (audio_signal * 32767).astype(np.int16)
    
    # Save the file
    wav.write(filename, sample_rate, audio_int)
    logger.info(f"Created sample audio: {filename}")

def main():
    """Create sample audio files for all emotions."""
    logger.info("Creating sample audio files for emotion testing...")
    
    # Create samples directory
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)
    
    # List of emotions to create samples for
    emotions = list(Config.EMOTIONS.values())
    
    for emotion in emotions:
        filename = os.path.join(samples_dir, f"{emotion}_sample.wav")
        create_emotion_sample(emotion, filename)
    
    logger.info(f"Successfully created {len(emotions)} sample audio files in '{samples_dir}' directory")
    logger.info("Sample files created:")
    
    for emotion in emotions:
        filename = f"{emotion}_sample.wav"
        logger.info(f"  - {filename}")
    
    logger.info("\\nThese sample files can be used to test the emotion recognition system.")
    logger.info("Note: These are synthetic samples. For best results, use real human speech data.")

if __name__ == "__main__":
    main()
