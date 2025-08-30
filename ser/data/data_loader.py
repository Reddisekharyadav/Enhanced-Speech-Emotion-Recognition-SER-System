"""
Data Loader for Speech Emotion Recognition (SER) Tool

This module provides functions to load and process audio data for the SER tool. It
handles feature extraction from audio files, splitting the dataset into training and
testing sets, and parallel processing of audio files.

Functions:
    - process_file: Processes an audio file to extract features and associated emotion label.
    - load_data: Loads data from the dataset directory and splits it into training and testing sets.

  
"""

import os
import glob
import logging
import random
from typing import List, Tuple, Optional, Callable
import multiprocessing as mp
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split

from ser.utils import get_logger
from ser.features import extract_feature
from ser.config import Config


logger: logging.Logger = get_logger(__name__)


def process_file(
    file: str, observed_emotions: List[str]
) -> Tuple[np.ndarray, str]:
    """
    Process an audio file to extract features and the associated emotion label.

    Arguments:
        file (str): Path to the audio file.
        observed_emotions (List[str]): List of observed emotions.

    Returns:
        Optional[Tuple[np.ndarray, str]]: Extracted features and associated
            emotion label for the audio file.
        Returns None if the emotion is not in observed_emotions.
    """
    try:
        logger.info(f"Processing file: {file}")
        file_name: str = os.path.basename(file)
        logger.info(f"Extracting emotion from filename: {file_name}")
        split_parts = file_name.split("-")
        if len(split_parts) < 3:
            logger.warning(f"Filename format unexpected: {file_name}")
        emotion: Optional[str] = Config.EMOTIONS.get(split_parts[2] if len(split_parts) > 2 else "")
        logger.info(f"Extracted emotion: {emotion}")
        if not emotion or emotion not in observed_emotions:
            logger.warning(f"Emotion not found or not observed for file: {file}")
            return (np.array([]), "")
        logger.info(f"Starting feature extraction for file: {file}")
        try:
            features: np.ndarray = extract_feature(file)
            logger.info(f"Feature extraction complete for file: {file}, shape: {features.shape if isinstance(features, np.ndarray) else 'N/A'}")
        except Exception as fe:
            logger.error(f"Feature extraction failed for file: {file}: {fe}")
            raise fe
        return (features, emotion)
    except Exception as e:
        logger.error(msg=f"Failed to process file {file}: {e}", exc_info=True)
        raise e


def load_data(test_size: float = 0.2, use_multiprocessing: bool = True, max_files: Optional[int] = None, progress_cb: Optional[Callable[[str], None]] = None) -> Optional[List]:
    """
    Load data from the dataset directory and split into training and testing sets.

    Arguments:
        test_size (float): Fraction of the dataset to be used as test set.

    Returns:
        Tuple containing training features, training labels, test features,
        and test labels.
    """
    observed_emotions: List[str] = list(Config.EMOTIONS.values())
    data: List[Tuple[np.ndarray, str]]
    data_path_pattern: str = (
        f"{Config.DATASET['folder']}/"
        f"{Config.DATASET['subfolder_prefix']}/"
        f"{Config.DATASET['extension']}"
    )
    files: List[str] = glob.glob(data_path_pattern)
    if max_files is not None and max_files > 0:
        # Randomly sample a subset for faster training to avoid class/order bias
        files = random.sample(files, k=min(max_files, len(files)))
    logger.info(f"Dataset glob '{data_path_pattern}' -> {len(files)} files; multiprocessing={use_multiprocessing} workers={Config.MODELS_CONFIG['num_cores']}")
    if progress_cb:
        progress_cb(f"Found {len(files)} files. Starting feature extraction...")

    if use_multiprocessing and int(Config.MODELS_CONFIG["num_cores"]) > 1 and len(files) > 1:
        workers = int(Config.MODELS_CONFIG["num_cores"]) or 1
        workers = min(workers, max(2, len(files) // 50)) if len(files) < 200 else workers
        chunksize = max(1, len(files) // (workers * 4) or 1)
        logger.info(f"Starting multiprocessing pool: workers={workers}, chunksize={chunksize}")
        with mp.Pool(workers) as pool:
            # Use imap_unordered to allow progress updates as results arrive
            results = []
            for idx, res in enumerate(pool.imap_unordered(partial(process_file, observed_emotions=observed_emotions), files, chunksize=chunksize), start=1):
                results.append(res)
                if idx % 100 == 0 or idx == len(files):
                    msg = f"Extracted features for {idx}/{len(files)} files..."
                    logger.info(msg)
                    if progress_cb:
                        progress_cb(msg)
            data = results
    else:
        # Sequential fallback with simple progress logs
        data = []
        last_log = 0
        for idx, f in enumerate(files, start=1):
            try:
                feats, emo = process_file(f, observed_emotions)
                # Defensive check: skip obviously broken feature vectors
                if isinstance(feats, np.ndarray) and feats.size > 0:
                    data.append((feats, emo))
                else:
                    logger.warning(f"Empty feature vector for {f}; skipping")
            except Exception as e:
                logger.warning(f"Skipping file due to error: {f} -> {e}")
            # Log more frequently early on to show liveness
            if idx % 25 == 0 or idx == 1 or idx == len(files) or (idx - last_log) >= 60:
                msg = f"Processed {idx}/{len(files)} files..."
                logger.info(msg)
                if progress_cb:
                    progress_cb(msg)
                last_log = idx

    # Remove invalid entries (None, empty feature arrays, or empty labels)
    cleaned = []
    for item in data:
        if item is None:
            continue
        feats, label = item
        try:
            from ser.features.feature_extractor import pad_or_truncate
            if isinstance(feats, np.ndarray) and feats.size > 0 and isinstance(label, str) and label:
                # Always pad/truncate to (128, 128)
                feats = pad_or_truncate(feats, (128, 128))
                cleaned.append((feats, label))
        except Exception:
            continue
    data = cleaned
    logger.info(f"Retained {len(data)}/{len(files)} samples after cleaning")
    if not data:
        logger.warning("No data found or processed.")
        return None

    features: Tuple[np.ndarray, ...]
    labels: Tuple[str, ...]
    features, labels = zip(*data)
    # Debug: print shape of each feature
    for i, feat in enumerate(features):
        logger.info(f"Feature {i} shape: {np.shape(feat)}")
    # Map string labels to integer indices
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels_idx = [label_to_idx[label] for label in labels]
    return train_test_split(
        np.array(features), labels_idx, test_size=test_size, random_state=42, stratify=labels_idx
    )
