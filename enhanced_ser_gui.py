from ser.models.enhanced_emotion_model import load_ser_model, train_enhanced_model

def load_trained_model(self, best=True):
    """Load the trained SER model (best or final)."""
    try:
        self.model = load_ser_model(best=best)
        logger.info("Trained model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load trained model: {e}")
        self.model = None

def train_and_save_model(self):
    """Train the enhanced model and save it."""
    try:
        logger.info("Starting model training...")
        train_enhanced_model()
        logger.info("Model training and saving complete.")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
import sys
import time
import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import threading
import requests
from urllib.parse import urlparse
import random
import traceback
import logging
from glob import glob

from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                           QHBoxLayout, QTextEdit, QLabel, QFileDialog, 
                           QTabWidget, QProgressBar, QComboBox, QSlider,
                           QFrame, QScrollArea, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches

from ser.models.enhanced_emotion_model import predict_emotion_realtime, train_enhanced_model
from ser.config import Config
from ser.utils import get_logger

logger = get_logger(__name__)

# Raise verbosity to debug for diagnostics
try:
    logger.setLevel(logging.DEBUG)
    logger.debug("Logger level set to DEBUG")
except Exception:
    pass

# Constants for real-time processing
SAMPLE_RATE = Config.REALTIME_CONFIG["sample_rate"]
CHUNK_DURATION = Config.REALTIME_CONFIG["chunk_duration"]
CHANNELS = Config.REALTIME_CONFIG["channels"]

# UI Text Constants
START_RECORDING_TEXT = "🎙️ Start Recording"
STOP_RECORDING_TEXT = "⏹️ Stop Recording"
ANALYZE_EMOTION_TEXT = "🔍 Analyze Emotion"
ANALYZING_TEXT = "🔄 Analyzing..."
START_TRAINING_TEXT = "🚀 Start Training"
RETRAIN_MODEL_TEXT = "🔄 Retrain Model"



class EmotionVisualizationWidget(QWidget):
    """Widget for visualizing emotion detection results."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.emotion_history = []
        self.confidence_history = []
        self.max_history = 50
        # Initialize default state
        self.reset()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Current emotion display
        self.emotion_label = QLabel("🎭 Current Emotion: Neutral")
        self.emotion_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2E86AB;
                background-color: white;
                border: 2px solid #2E86AB;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
            }
        """)
        layout.addWidget(self.emotion_label)
        
        # Confidence bar
        self.confidence_label = QLabel("Confidence: 0%")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #A23B72;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #F18F01;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.confidence_bar)
        
        # Emotion history chart
        self.figure, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def reset(self):
        """Reset the visualization to the default state for a new analysis."""
        try:
            # Clear histories
            self.emotion_history = []
            self.confidence_history = []
            # Reset labels and progress bar
            self.emotion_label.setText("🎭 Current Emotion: Neutral")
            self.confidence_label.setText("Confidence: 0%")
            self.confidence_bar.setValue(0)
            # Clear chart
            if hasattr(self, 'ax'):
                self.ax.clear()
                self.ax.set_title("Recent Emotion Distribution", fontsize=14, fontweight='bold')
                self.ax.grid(True, alpha=0.3)
                self.figure.tight_layout()
                self.canvas.draw()
        except Exception as e:
            logger.warning(f"Failed to reset visualization: {e}")
        
    @pyqtSlot()
    def update_chart(self):
        """Update the emotion history chart."""
        if len(self.emotion_history) == 0:
            return
            
        self.ax.clear()
        
        # Count emotions in recent history
        emotion_counts = {}
        for emotion in self.emotion_history[-20:]:  # Last 20 predictions
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if emotion_counts:
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            
            # Color map for emotions
            color_map = {
                "happy": "#FFD700", "sad": "#4169E1", "angry": "#DC143C",
                "fearful": "#FF8C00", "surprised": "#FF1493", "neutral": "#808080",
                "calm": "#20B2AA", "disgust": "#9932CC"
            }
            colors = [color_map.get(emotion, "#808080") for emotion in emotions]
            
            bars = self.ax.bar(emotions, counts, color=colors, alpha=0.7)
            self.ax.set_title("Recent Emotion Distribution", fontsize=14, fontweight='bold')
            self.ax.set_ylabel("Count")
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom')
        
        self.ax.grid(True, alpha=0.3)
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        self.figure.tight_layout()
        self.canvas.draw()

    @pyqtSlot(str, float)
    def update_emotion(self, emotion, confidence):
        """Update the emotion display and history."""
        # Update current emotion display
        emoji_map = {
            "happy": "😊", "sad": "😢", "angry": "😠", "fearful": "😨",
            "surprised": "😲", "neutral": "😐", "calm": "😌", "disgust": "🤢"
        }
        emoji = emoji_map.get(emotion, "🎭")
        
        self.emotion_label.setText(f"{emoji} Current Emotion: {emotion.title()}")
        
        # Update confidence
        confidence_percent = int(confidence * 100)
        self.confidence_label.setText(f"Confidence: {confidence_percent}%")
        self.confidence_bar.setValue(confidence_percent)
        
        # Add to history
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        
        # Keep only recent history
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
        
        # Update chart
        self.update_chart()

def get_ai_response(emotion: str) -> str:
    """Return a basic AI response based on the detected emotion."""
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

class AIResponseWidget(QWidget):
    """Widget for AI chat responses based on emotions."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #F5F5F5;
                border: 2px solid #A23B72;
                border-radius: 10px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        
        # Add welcome message
        welcome_msg = """
        <div style='background-color: #E3F2FD; padding: 10px; border-radius: 5px; margin: 5px;'>
            <b>🤖 AI Assistant:</b> Hello! I'm here to respond to your emotions in real-time. 
            Start recording or load an audio file to begin our conversation!
        </div>
        """
        self.chat_display.setHtml(welcome_msg)
        
        layout.addWidget(QLabel("💬 AI Emotional Chat"))
        layout.addWidget(self.chat_display)
        
        self.setLayout(layout)
        
    @pyqtSlot(str, float)
    def add_emotion_response(self, emotion, confidence):
        """Add AI response based on detected emotion."""
        if confidence < Config.REALTIME_CONFIG["confidence_threshold"]:
            return  # Don't respond to low-confidence predictions
            
        ai_response = get_ai_response(emotion)
        
        # Create formatted message
        timestamp = time.strftime("%H:%M:%S")
        emotion_color = {
            "happy": "#4CAF50", "sad": "#2196F3", "angry": "#F44336",
            "fearful": "#FF9800", "surprised": "#E91E63", "neutral": "#9E9E9E",
            "calm": "#00BCD4", "disgust": "#9C27B0"
        }.get(emotion, "#9E9E9E");
        
        message = f"""
        <div style='background-color: {emotion_color}20; padding: 10px; border-radius: 5px; margin: 5px; border-left: 4px solid {emotion_color};'>
            <b style='color: {emotion_color};'>🎭 Detected Emotion:</b> {emotion.title()} ({confidence:.0%} confidence)<br>
            <b>🤖 AI Response:</b> {ai_response}<br>
            <small style='color: #666;'>Time: {timestamp}</small>
        </div>
        """
        
        # Append to chat
        current_html = self.chat_display.toHtml()
        self.chat_display.setHtml(current_html + message)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

class AdvancedSERApp(QWidget):
    """Main application window for Speech Emotion Recognition."""
    
    log_message = pyqtSignal(str)
    training_finished = pyqtSignal(dict)
    file_emotion_result = pyqtSignal(str, float)
    file_analysis_finished = pyqtSignal()
    realtime_emotion_update = pyqtSignal(str, float)
    ai_response_update = pyqtSignal(str, float)
    transcript_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.is_recording = False
        self.current_audio_file = None
        self.stt_model = None  # lazy-loaded speech-to-text model
        self.install_global_exception_hook()
        self.log_message.connect(self.training_log.append)
        # connect other signals after UI ready
        self.file_emotion_result.connect(self.file_emotion_viz.update_emotion)
        self.file_analysis_finished.connect(self.on_file_analysis_finished)
        self.realtime_emotion_update.connect(self.emotion_viz.update_emotion)
        self.ai_response_update.connect(self.ai_chat.add_emotion_response)
        self.transcript_update.connect(self.update_transcript)
        self.training_finished.connect(self.handle_training_results)
        logger.debug("GUI initialized; attempting initial model load")
        self.attempt_initial_model_load()

    def install_global_exception_hook(self):
        def handle_exception(exc_type, exc_value, exc_tb):
            msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            logger.error(f"UNCAUGHT EXCEPTION:\n{msg}")
        sys.excepthook = handle_exception

    def init_ui(self):
        self.setWindowTitle(Config.GUI_CONFIG["window_title"])
        self.setGeometry(100, 100, *Config.GUI_CONFIG["window_size"])
        self.setStyleSheet("""
            QWidget { background-color: #F5F5F5; font-family: 'Segoe UI'; }
            QPushButton { background-color: #2E86AB; color: white; border: none; padding: 10px 20px; border-radius: 5px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #1F5F79; }
            QTabWidget::pane { border: 2px solid #A23B72; border-radius: 5px; }
            QTabBar::tab { background-color: #A23B72; color: white; padding: 10px 20px; border-radius: 5px 5px 0 0; }
            QTabBar::tab:selected { background-color: #2E86AB; }
        """)
        
        main_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        
        self.tab_widget.addTab(self.create_realtime_tab(), "🎙️ Real-time")
        self.tab_widget.addTab(self.create_file_tab(), "📁 File Analysis")
        self.tab_widget.addTab(self.create_samples_tab(), "🎵 Sample Audio")
        self.tab_widget.addTab(self.create_training_tab(), "🔧 Model Training")
        
        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_audio_chunk)
        
    def create_realtime_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        controls_layout = QHBoxLayout()

        self.start_btn = QPushButton(START_RECORDING_TEXT)
        self.start_btn.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.start_btn)

        controls_layout.addWidget(QLabel("Sensitivity:"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(6)
        controls_layout.addWidget(self.sensitivity_slider)

        layout.addLayout(controls_layout)

        splitter = QSplitter(Qt.Horizontal)
        self.emotion_viz = EmotionVisualizationWidget()
        splitter.addWidget(self.emotion_viz)
        self.ai_chat = AIResponseWidget()
        splitter.addWidget(self.ai_chat)
        splitter.setSizes([600, 600])

        layout.addWidget(splitter)

        # Transcription panel
        trans_layout = QVBoxLayout()
        trans_label = QLabel("📝 Transcribed Text (per chunk)")
        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        self.transcript_display.setStyleSheet("QTextEdit { background: #FFFFFF; border: 1px solid #CCC; border-radius: 6px; }")
        trans_layout.addWidget(trans_label)
        trans_layout.addWidget(self.transcript_display)
        layout.addLayout(trans_layout)

        tab.setLayout(layout)
        return tab
        
    def create_file_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        file_layout = QHBoxLayout()
        
        self.file_path_label = QLabel("No file selected")
        self.browse_btn = QPushButton("📁 Browse Audio File")
        self.browse_btn.clicked.connect(self.browse_file)
        self.analyze_btn = QPushButton(ANALYZE_EMOTION_TEXT)
        self.analyze_btn.clicked.connect(self.analyze_file)
        self.analyze_btn.setEnabled(False)
        
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(self.browse_btn)
        file_layout.addWidget(self.analyze_btn)
        layout.addLayout(file_layout)
        
        self.file_emotion_viz = EmotionVisualizationWidget()
        layout.addWidget(self.file_emotion_viz)
        
        tab.setLayout(layout)
        return tab
        
    def create_samples_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("🎵 Download sample audio files to test different emotions."))
        
        samples_layout = QHBoxLayout()
        for emotion in ["happy", "sad", "angry", "neutral"]:
            btn = QPushButton(f"📥 Download {emotion.title()} Sample")
            btn.clicked.connect(lambda checked, e=emotion: self.download_sample(e))
            samples_layout.addWidget(btn)
        layout.addLayout(samples_layout)
        
        self.download_progress = QLabel("Ready to download samples")
        layout.addWidget(self.download_progress)
        self.samples_list = QTextEdit()
        self.samples_list.setReadOnly(True)
        layout.addWidget(self.samples_list)
        
        tab.setLayout(layout)
        return tab
        
    def create_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("🔧 Model Training Instructions: Ensure RAVDESS dataset is in ser/features/dataset/ravdess/"))

        self.model_status_label = QLabel("Checking model status...")
        layout.addWidget(self.model_status_label)

        training_controls = QHBoxLayout()
        self.train_btn = QPushButton(START_TRAINING_TEXT)
        self.train_btn.clicked.connect(self.start_training)
        training_controls.addWidget(self.train_btn)

        self.check_model_btn = QPushButton("🔍 Check Model Status")
        self.check_model_btn.clicked.connect(self.check_model_status)
        training_controls.addWidget(self.check_model_btn)
        layout.addLayout(training_controls)

        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        layout.addWidget(self.training_progress)

        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        layout.addWidget(self.training_log)

        self.check_model_status()
        tab.setLayout(layout)
        return tab
        
    @pyqtSlot()
    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.start_btn.setText(STOP_RECORDING_TEXT)
            self.timer.start(int(CHUNK_DURATION * 1000))
        else:
            self.start_btn.setText(START_RECORDING_TEXT)
            self.timer.stop()
            
    @pyqtSlot()
    def process_audio_chunk(self):
            try:
                audio = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
                sd.wait()
                # Apply noise reduction
                try:
                    import noisereduce as nr
                    # If stereo, use first channel for noise profile
                    if audio.ndim > 1:
                        noise_profile = audio[:, 0]
                    else:
                        noise_profile = audio
                    reduced_audio = nr.reduce_noise(y=audio.flatten(), sr=SAMPLE_RATE, y_noise=noise_profile)
                    logger.debug("Noise reduction applied to audio chunk.")
                except Exception as ne:
                    logger.warning(f"Noise reduction failed or not available: {ne}")
                    reduced_audio = audio.flatten()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    wav.write(tmp.name, SAMPLE_RATE, (reduced_audio * 32767).astype(np.int16))
                    threading.Thread(target=self.analyze_realtime_audio, args=(tmp.name,), daemon=True).start()
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
            
    def browse_file(self):
        logger.info("Browse file dialog opened")
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
        if path:
            logger.info(f"File selected: {path}")
            # Reset previous file analysis UI so we start fresh
            self.file_emotion_viz.reset()
            self.current_audio_file = path
            self.file_path_label.setText(f"Selected: {os.path.basename(path)}")
            self.analyze_btn.setEnabled(True)
        else:
            logger.info("No file selected")

    def analyze_file(self):
        if not self.current_audio_file:
            logger.warning("Analyze requested but no file selected")
            return
        logger.info(f"Starting file analysis for: {self.current_audio_file}")
        # Clear previous results for a clean start
        self.file_emotion_viz.reset()
        self.analyze_btn.setText(ANALYZING_TEXT)
        self.analyze_btn.setEnabled(False)
        t = threading.Thread(target=self.analyze_file_emotion_thread, daemon=True)
        t.start()
        logger.info(f"Spawned analysis thread id={t.ident}")

    def analyze_file_emotion_thread(self):
        thread_id = threading.get_ident()
        logger.info(f"[File Analysis] Thread {thread_id} started")
        try:
            if not os.path.exists(self.current_audio_file):
                raise FileNotFoundError(f"Audio file missing: {self.current_audio_file}")
            logger.info(f"[File Analysis] Loading & predicting: {self.current_audio_file}")
            result = predict_emotion_realtime(self.current_audio_file)
            if isinstance(result, dict) and "label" in result and "confidence" in result:
                emotion = result["label"]
                confidence = result["confidence"]
            else:
                emotion = str(result)
                confidence = 1.0
            logger.info(f"[File Analysis] Thread {thread_id} prediction -> emotion={emotion} confidence={confidence}")
            self.file_emotion_result.emit(emotion, confidence)
        except Exception as e:
            logger.error(f"[File Analysis] Thread {thread_id} error: {e}", exc_info=True)
        finally:
            self.file_analysis_finished.emit()
            logger.info(f"[File Analysis] Thread {thread_id} finished (signal emitted)")
            logger.debug("File analysis thread finished")

    def on_file_analysis_finished(self):
        logger.debug("File analysis finished UI update")
        self.analyze_btn.setText(ANALYZE_EMOTION_TEXT)
        self.analyze_btn.setEnabled(True)

    def analyze_realtime_audio(self, audio_path):
            logger.debug(f"Realtime chunk analysis started: {audio_path}")
            try:
                # Optional transcription (non-blocking UI, we're in a thread)
                text = self.transcribe_chunk(audio_path)
                if text:
                    self.transcript_update.emit(text)
                else:
                    self.transcript_update.emit("[no speech recognized]")
                result = predict_emotion_realtime(audio_path)
                # Robustly unpack emotion/confidence from dict or tuple/list
                if isinstance(result, dict) and "label" in result and "confidence" in result:
                    emotion = result["label"]
                    confidence = result["confidence"]
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    first = result[0]
                    if isinstance(first, (list, tuple)):
                        if len(first) >= 2:
                            emotion = first[0]
                            confidence = first[1] if isinstance(first[1], float) else 1.0
                        else:
                            emotion = first[0]
                            confidence = 1.0
                    else:
                        emotion = str(first)
                        confidence = 1.0
                else:
                    emotion = str(result)
                    confidence = 1.0
                logger.debug(f"[Realtime] Raw prediction {emotion} ({confidence:.2f})")
                # Dynamic threshold: higher sensitivity lowers the bar
                slider_val = self.sensitivity_slider.value()  # 1..10
                effective_threshold = Config.REALTIME_CONFIG["confidence_threshold"] * ((11 - slider_val) / 10.0)
                if confidence >= effective_threshold:
                    self.realtime_emotion_update.emit(emotion, confidence)
                    self.ai_response_update.emit(emotion, confidence)
                    logger.debug("Realtime update emitted to UI")
                else:
                    logger.debug(f"Realtime prediction below threshold ({confidence:.2f} < {effective_threshold:.2f}); ignored")
            except Exception as e:
                logger.error(f"[Real-time Analysis] Error: {e}", exc_info=True)
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    logger.debug(f"Temp file removed: {audio_path}")

    @pyqtSlot(str)
    def update_transcript(self, text: str):
        try:
            timestamp = time.strftime('%H:%M:%S')
            self.transcript_display.append(f"[{timestamp}] {text}")
            # Limit transcript to last ~100 lines
            if self.transcript_display.document().blockCount() > 100:
                cursor = self.transcript_display.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.select(cursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deletePreviousChar()
        except Exception as e:
            logger.error(f"Transcript UI update failed: {e}")

    def transcribe_chunk(self, audio_path: str) -> str:
        """Transcribe a short audio chunk. Tries Whisper if available; returns empty string on failure."""
        try:
            # Lazy import and model load
            if self.stt_model is None:
                try:
                    import whisper  # type: ignore
                except Exception as ie:
                    logger.info(f"Whisper not available for transcription: {ie}")
                    return ""
                model_name = (Config.MODELS_CONFIG.get("whisper_model") or {}).get("name", "tiny")
                logger.info(f"Loading Whisper model '{model_name}' for transcription (first time may be slow)...")
                self.stt_model = whisper.load_model(model_name)
            # Run transcription (short chunk)
            result = self.stt_model.transcribe(audio_path, fp16=False, language=Config.DEFAULT_LANGUAGE)
            text = (result or {}).get("text", "").strip()
            logger.debug(f"Transcription result: '{text}'")
            return text
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            return ""

    def attempt_initial_model_load(self):
        try:
            logger.info("Existing model loaded on startup.")
            self.log_message.emit("✅ Existing model loaded.")
        except Exception as e:
            logger.warning("No existing model found on startup.")
            logger.error(f"Startup model load failed: {e}")

    def download_sample(self, emotion):
        try:
            self.download_progress.setText(f"Creating {emotion} sample...")
            samples_dir = "samples"
            os.makedirs(samples_dir, exist_ok=True)
            duration, sr = 3.0, 16000
            freq = {"happy": 440, "sad": 220, "angry": 660, "neutral": 330}.get(emotion, 330)
            t = np.linspace(0, duration, int(sr * duration), False)
            rng = np.random.default_rng(seed=42)
            audio_data = np.sin(2 * np.pi * freq * t) * 0.3 + rng.normal(0, 0.05, int(sr * duration))
            filename = os.path.join(samples_dir, f"{emotion}_sample.wav")
            wav.write(filename, sr, (audio_data * 32767).astype(np.int16))
            self.download_progress.setText(f"✅ Created {emotion} sample!")
            self.samples_list.append(f"✅ {emotion}_sample.wav - Created")
        except Exception as e:
            logger.error(f"[Samples] Failed creating sample {emotion}: {e}")
            self.download_progress.setText(f"❌ Failed to create {emotion} sample")

    def check_model_status(self):
        model_path = Config.MODELS_CONFIG.get("enhanced_model_path")
        if model_path and os.path.exists(model_path):
            self.model_status_label.setText("✅ Enhanced model found!")
            self.train_btn.setText(RETRAIN_MODEL_TEXT)
        else:
            self.model_status_label.setText("⚠️ No trained model found. Please train a model.")
            self.train_btn.setText(START_TRAINING_TEXT)

    def check_dataset_available(self) -> bool:
        base = Config.DATASET["folder"]
        pattern = os.path.join(base, Config.DATASET["subfolder_prefix"], Config.DATASET["extension"]) 
        files = glob(pattern)
        if not files:
            self.log_message.emit(f"⚠️ Dataset not found at {os.path.abspath(base)}. Expected WAVs under {Config.DATASET['subfolder_prefix']}")
            return False
        return True

    def start_training(self):
        # Skip if model already present unless user confirms retrain
        model_path = Config.MODELS_CONFIG.get("enhanced_model_path")
        scaler_path = Config.MODELS_CONFIG.get("scaler_path")
        if (model_path and scaler_path and os.path.exists(model_path) and os.path.exists(scaler_path)):
            reply = QMessageBox.question(
                self,
                "Model Already Trained",
                "A trained model and scaler already exist. Do you want to retrain?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                self.log_message.emit("✅ Using existing trained model (skipped retraining).")
                self.check_model_status()
                return
        # Dataset check
        if not self.check_dataset_available():
            QMessageBox.warning(self, "Dataset Missing", f"No dataset files found at {Config.DATASET['folder']}. Please ensure RAVDESS WAVs exist under Actor_* folders.")
            return
        # Proceed with training
        self.train_btn.setEnabled(False)
        self.training_progress.setVisible(True)
        self.training_progress.setRange(0, 0)
        self.training_log.clear()
        self.log_message.emit("🚀 Starting model training...")
        threading.Thread(target=self.train_model_background, daemon=True).start()
        
    def train_model_background(self):
        try:
            self.log_message.emit("📊 Starting optimized training...")
            results = train_enhanced_model()
            self.training_finished.emit(results)
        except Exception as e:
            logger.error(f"[Training] Error during model training: {e}", exc_info=True)
            self.log_message.emit(f"❌ Training failed: {e}")
            self.train_btn.setEnabled(True)
            self.training_progress.setVisible(False)

    @pyqtSlot(dict)
    def handle_training_results(self, results):
        self.log_message.emit(f"✅ Training completed in {results.get('training_time', 0):.2f}s")
        self.log_message.emit(f"🎯 Model accuracy: {results.get('accuracy', 0):.3f}")
        self.check_model_status()
        self.train_btn.setEnabled(True)
        self.training_progress.setVisible(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    main_window = AdvancedSERApp()
    main_window.show()
    sys.exit(app.exec_())
