"""
Speech Emotion Recognition (SER) Tool

This module serves as the entry point for the Speech Emotion Recognition (SER)
tool. It provides command-line interface (CLI) options for training the 
emotion classification model or predicting emotions and generating transcripts
from audio files.

Usage:
    The tool can be operated in two primary modes:
    1. Training mode: Trains the model using labeled audio data.
    2. Prediction mode: Predicts emotions in a given audio file 
        and extracts the transcript.

Author: Juan Sugg (juanpedrosugg@gmail.com)
Version: 1.0
License: MIT
"""

import argparse
import sys
import time
import logging
from typing import List, Tuple

from ser.utils import get_logger
from ser.config import Config


logger: logging.Logger = get_logger("ser")


def main() -> None:
    """
    Main function to handle the command line interface logic.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Enhanced Speech Emotion Recognition Tool"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the standard emotion classification model",
    )
    parser.add_argument(
        "--train-enhanced",
        action="store_true",
        help="Train the enhanced emotion classification model (recommended)",
    )
    parser.add_argument(
        "--fast-enhanced",
        action="store_true",
        help="Use fast mode for enhanced training (MFCC-only, lower accuracy but quicker)",
    )
    parser.add_argument(
        "--full-enhanced",
        action="store_true",
        help="Use full/accuracy mode for enhanced training (richer features; slower but higher accuracy)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["hgb", "svm", "rf"],
        default="hgb",
        help="Enhanced model type: hgb (default), svm, or rf",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run a small hyperparameter search to squeeze extra accuracy (slower)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the audio file for emotion prediction",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Use enhanced real-time emotion prediction",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=Config.DEFAULT_LANGUAGE,
        help="Language of the audio file",
    )
    parser.add_argument(
        "--save_transcript",
        action="store_true",
        help="Save the transcript to a CSV file",
    )
    parser.add_argument(
        "--ai-response",
        action="store_true",
        help="Generate AI response based on detected emotion",
    )
    parser.add_argument(
        "--overall",
        action="store_true",
        help="Return a single overall emotion for the whole file (standard model)",
    )
    parser.add_argument(
        "--by-sentences",
        action="store_true",
        help="Return one emotion per sentence-like segment (uses pauses)",
    )
    parser.add_argument(
        "--overall-from-sentences",
        action="store_true",
        help="Return a single emotion aggregated from sentence segments",
    )
    args: argparse.Namespace = parser.parse_args()

    # Training modes
    if args.train:
        from ser.models.emotion_model import train_model
        logger.info("Starting standard model training...")
        start_time: float = time.time()
        train_model()
        logger.info(
            msg=f"Training completed in {time.time() - start_time:.2f} seconds"
        )
        sys.exit(0)
        
    if args.train_enhanced:
        from ser.models.enhanced_emotion_model import train_enhanced_model
        logger.info("Starting enhanced model training...")
        start_time: float = time.time()
        train_enhanced_model()
        logger.info(
            msg=f"Enhanced training completed in {time.time() - start_time:.2f} seconds"
        )
        sys.exit(0)

    if not args.file:
        logger.error(msg="No audio file provided for prediction.")
        sys.exit(1)

    logger.info(msg="Starting emotion prediction...")
    start_time = time.time()
    
    # Use enhanced model for real-time prediction
    if args.realtime:
        from ser.models.enhanced_emotion_model import predict_emotion_realtime, get_ai_response
        emotion, confidence = predict_emotion_realtime(args.file)
        logger.info(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
        
        if args.ai_response:
            ai_response = get_ai_response(emotion)
            logger.info(f"AI Response: {ai_response}")
            
        sys.exit(0)
    
    # Standard prediction variants
    if args.overall:
        from ser.models.emotion_model import predict_overall_emotion
        label = predict_overall_emotion(args.file)
        logger.info(f"Overall emotion: {label}")
    elif args.by_sentences:
        from ser.models.emotion_model import predict_emotions_by_sentences
        segs = predict_emotions_by_sentences(args.file)
        for emo, s, e in segs:
            logger.info(f"{emo}: {s:.2f}s → {e:.2f}s")
    elif args.overall_from_sentences:
        from ser.models.emotion_model import predict_overall_emotion_from_sentences
        label = predict_overall_emotion_from_sentences(args.file)
        logger.info(f"Overall (from sentences): {label}")
    else:
        # Standard prediction with transcript and timeline
        from ser.models.emotion_model import predict_emotions
        from ser.transcript import extract_transcript
        from ser.utils import (
            build_timeline,
            print_timeline,
            save_timeline_to_csv,
        )
        emotions: List[Tuple[str, float, float]] = predict_emotions(args.file)
        transcript: List[Tuple[str, float, float]] = extract_transcript(args.file, args.language)
        timeline: list = build_timeline(transcript, emotions)
        print_timeline(timeline)

        if args.save_transcript:
            csv_file_name: str = save_timeline_to_csv(timeline, args.file)
            logger.info(msg=f"Timeline saved to {csv_file_name}")

    logger.info(
        msg=f"Emotion prediction completed in {time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
