"""
Emotion Classification Model for Speech Emotion Recognition (SER) System

This module provides functions for training and using the emotion classification model
in the SER system. It includes functions to train the model, load the trained model, and
predict emotions from audio files.

Functions:
    - train_model: Trains the emotion classification model.
    - load_model: Loads the trained emotion classification model.
    - predict_emotions: Predicts emotions from an audio file.

  
"""

import os
import warnings
import logging
import pickle
from typing import Optional, Tuple, List
import tempfile
import time
import scipy.io.wavfile as wav

import numpy as np
import librosa
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from halo import Halo

from ser.utils import get_logger, read_audio_file
from ser.transcript import extract_transcript
from ser.config import Config
from ser.data import load_data
from ser.features import extended_extract_feature


logger: logging.Logger = get_logger(__name__)


def train_model() -> None:
    """
    Train the emotion classification model.

    This function loads the dataset, trains an MLPClassifier on the training data,
    measures the model's accuracy on the test data, and saves the trained model to a file.

    Raises:
        Exception: If the dataset is not loaded successfully.
    """
    with Halo(text="Loading dataset... ", spinner="dots", text_color="green"):
        if data := load_data(test_size=0.25):
            x_train, x_test, y_train, y_test = data
            # Ensure numpy arrays and downcast to float32 for speed/memory
            x_train = np.asarray(x_train, dtype=np.float32)
            x_test = np.asarray(x_test, dtype=np.float32)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)
            n_classes = len(set(y_train.tolist())) if y_train.size else 0
            logger.info(
                msg=f"Dataset loaded: train={x_train.shape}, test={x_test.shape}, classes={n_classes}"
            )
            # Build a fast, stable pipeline: StandardScaler -> MLPClassifier
            model_params = {
                "hidden_layer_sizes": (128, 64),
                "activation": "relu",
                "solver": "adam",
                "batch_size": min(256, max(32, x_train.shape[0] // 20)) if x_train.shape[0] > 0 else 64,
                "learning_rate": "adaptive",
                "learning_rate_init": 1e-3,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 8,
                "max_iter": 200,
                "tol": 1e-3,
                "shuffle": True,
                "random_state": 42,
                "verbose": True,
            }
            # Merge overrides from Config if provided
            try:
                if hasattr(Config, "NN_PARAMS") and isinstance(Config.NN_PARAMS, dict):
                    model_params.update(Config.NN_PARAMS)
            except Exception:
                pass
            clf = MLPClassifier(**model_params)
            model: BaseEstimator = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                clf,
                memory=None,
            )
            logger.info(msg="Model pipeline constructed (StandardScaler + MLP).")
        else:
            logger.error("Dataset not loaded. Please load the dataset first.")
            raise RuntimeError(
                "Dataset not loaded. Please load the dataset first."
            )

    start = time.perf_counter()
    with Halo(
        text="Training the model (with early stopping)... ", spinner="dots", text_color="green"
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model.fit(x_train, y_train)
    elapsed = time.perf_counter() - start
    logger.info(msg=f"Model trained with {len(x_train)} samples in {elapsed:.2f}s")

    with Halo(
        text="Measuring accuracy... ", spinner="dots", text_color="green"
    ):
        y_pred: np.ndarray = model.predict(x_test)
        accuracy: float = float(accuracy_score(y_true=y_test, y_pred=y_pred))
        model_file: str = (
            f"{Config.MODELS_CONFIG['models_folder']}/ser_model.pkl"
        )
    logger.info(msg=f"Accuracy: {accuracy * 100:.2f}%")

    with Halo(text="Saving the model... ", spinner="dots", text_color="green"):
        # Use joblib for pipelines; keep same file name for backward compatibility
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        joblib.dump(model, model_file)
    logger.info(msg=f"Model saved to {model_file}")


def load_model() -> BaseEstimator:
    """
    Load the trained emotion classification model.

    This function loads the trained model pipeline from a file.

    Returns:
        BaseEstimator: The trained emotion classification model pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
                os.makedirs(os.path.dirname(model_file), exist_ok=True)
                joblib.dump(model, model_file)
    """
    model_path: str = f"{Config.MODELS_CONFIG['models_folder']}/ser_model.pkl"
    model: Optional[BaseEstimator] = None
    try:
        with Halo(
            text=f"Loading SER model from {model_path}... ",
            spinner="dots",
            text_color="green",
        ):
            if not os.path.exists(model_path):
                raise FileNotFoundError(model_path)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    model = joblib.load(model_path)
                except Exception:
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)

        if model is not None:
            logger.info(msg=f"Model loaded from {model_path}")
            return model
    except FileNotFoundError as err:
        logger.error(
            msg=(
                "Model not found. Please train the model first. If you already trained the model, "
                f"ensure that {model_path} exists and is a valid .pkl file."
            )
        )
        raise err

    logger.error(
        msg=(
            "Failed to load the model. Ensure MODELS_CONFIG['models_folder'] is configured and the file exists."
        )
    )
    raise ValueError("Failed to load the model.")


def predict_emotions(file: str) -> List[Tuple[str, float, float]]:
    """
    Predict emotions from an audio file.

    This function loads a trained model, extracts features from the audio file,
    predicts emotions at each timestamp, and returns a list of predicted emotions
    with their start and end timestamps.

    Args:
        file (str): Path to the audio file.

    Returns:
        List[Tuple[str, float, float]]: A list of tuples where each tuple contains
        the predicted emotion, start time, and end time.

    Raises:
        Exception: If the model is not loaded.
    """
    model: BaseEstimator = load_model()
    if model is None:
        raise RuntimeError("Model not loaded.")

    with Halo(
        text="Inferring Emotions from Audio File... ",
        spinner="dots",
        text_color="green",
    ):
        feature: List[np.ndarray] = extended_extract_feature(file)
        predicted_emotions: np.ndarray = model.predict(feature)
    logger.info(msg="Emotion inference completed.")

    audio_duration: float = librosa.get_duration(y=read_audio_file(file)[0])
    emotion_timestamps: List[Tuple[str, float, float]] = []
    prev_emotion: Optional[str] = None
    start_time: float = 0

    for timestamp, emotion in enumerate(predicted_emotions):
        if emotion != prev_emotion:
            if prev_emotion is not None:
                end_time: float = timestamp * audio_duration / len(predicted_emotions)
                emotion_timestamps.append((prev_emotion, start_time, end_time))
            (
                prev_emotion,
                start_time,
            ) = emotion, timestamp * audio_duration / len(predicted_emotions)

    if prev_emotion is not None:
        emotion_timestamps.append((prev_emotion, start_time, audio_duration))

    logger.info("Emotion prediction and timestamp extraction completed.")
    return emotion_timestamps


def _majority_emotion_weighted(spans: List[Tuple[str, float, float]]) -> str:
    """
    Pick one overall emotion weighted by span duration.
    spans: list of (emotion, start, end)
    """
    if not spans:
        return "neutral"
    weights = {}
    for emo, s, e in spans:
        dur = max(0.0, float(e) - float(s))
        weights[emo] = weights.get(emo, 0.0) + dur
    # fallback if zero durations
    if sum(weights.values()) <= 0:
        for emo, _, _ in spans:
            weights[emo] = weights.get(emo, 0.0) + 1.0
    return max(weights, key=weights.get)


def predict_overall_emotion(file: str) -> str:
    """
    Predict a single emotion for the whole audio.
    Uses the frame/timestep predictions from predict_emotions and returns
    a duration-weighted majority label.
    """
    spans = predict_emotions(file)
    overall = _majority_emotion_weighted(spans)
    logger.info(msg=f"Overall emotion (duration-weighted): {overall}")
    return overall


def _write_segment_to_tmp(y: np.ndarray, sr: int) -> str:
    # Write float audio to a temp WAV (int16) for existing feature pipeline
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, sr, (y * 32767).astype(np.int16))
    return tmp.name


def _segments_from_transcript(words: List[Tuple[str, float, float]], gap_threshold: float = 0.8) -> List[Tuple[float, float]]:
    """
    Convert word-level transcript into sentence-like time ranges using pauses.
    Returns list of (start, end) boundaries.
    """
    if not words:
        return []
    # words as (text, start, end)
    segments: List[Tuple[float, float]] = []
    cur_start = float(words[0][1])
    prev_end = float(words[0][2])
    for _, w_start, w_end in words[1:]:
        if float(w_start) - float(prev_end) >= gap_threshold:
            segments.append((cur_start, prev_end))
            cur_start = float(w_start)
        prev_end = float(w_end)
    segments.append((cur_start, prev_end))
    return segments


def predict_emotions_by_sentences(file: str, gap_threshold: float = 0.8) -> List[Tuple[str, float, float]]:
    """
    Predict one emotion per sentence-like segment (derived from transcript pauses).
    Returns a list of (emotion, start, end) per segment. If transcript fails or
    is empty, falls back to a single overall emotion for the entire file.
    """
    # 1) extract transcript words
    try:
        words = extract_transcript(file, language=Config.DEFAULT_LANGUAGE) or []
    except Exception as e:
        logger.warning(f"Transcript unavailable: {e}")
        words = []

    # 2) build segments from pauses
    segs = _segments_from_transcript(words, gap_threshold=gap_threshold)
    if not segs:
        # fallback: single overall emotion
        overall = predict_overall_emotion(file)
        dur = librosa.get_duration(y=read_audio_file(file)[0])
        return [(overall, 0.0, float(dur))]

    # 3) predict per segment using the existing model and feature pipeline
    model: BaseEstimator = load_model()
    y, sr = read_audio_file(file)
    results: List[Tuple[str, float, float]] = []
    for s, e in segs:
        s_idx = max(0, int(float(s) * sr))
        e_idx = min(len(y), int(float(e) * sr))
        if e_idx <= s_idx:
            continue
        y_seg = y[s_idx:e_idx]
        try:
            tmp_path = _write_segment_to_tmp(y_seg, sr)
            feats = extended_extract_feature(tmp_path)
            preds = model.predict(feats)
            # majority vote within sentence segment
            votes = {}
            for p in preds:
                votes[p] = votes.get(p, 0) + 1
            seg_emotion = max(votes, key=votes.get)
            results.append((seg_emotion, float(s), float(e)))
        except Exception as err:
            logger.warning(f"Segment prediction failed for {s:.2f}-{e:.2f}s: {err}")
        finally:
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
    return results


def predict_overall_emotion_from_sentences(file: str, gap_threshold: float = 0.8) -> str:
    """
    Compute a single emotion by aggregating sentence-level predictions using duration weighting.
    """
    seg_spans = predict_emotions_by_sentences(file, gap_threshold=gap_threshold)
    return _majority_emotion_weighted(seg_spans)
