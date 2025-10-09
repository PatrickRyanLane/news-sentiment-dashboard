#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Sentiment Classifier: VADER + DistilBERT

Strategy:
1. Use VADER (fast, rule-based) as the primary classifier
2. Calculate confidence based on compound score distance from thresholds
3. For low-confidence predictions, use DistilBERT (transformer-based) for refinement
4. Track which method was used for transparency and debugging

Confidence Thresholds:
- High confidence: VADER compound score is far from decision boundaries
- Low confidence: VADER compound score is near decision boundaries
  (e.g., between -0.15 and 0.35)
"""

from __future__ import annotations
import os
from typing import Literal, TypedDict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Lazy import for transformers (only loaded when needed)
_distilbert_pipeline = None


class SentimentResult(TypedDict):
    """Result from sentiment analysis"""
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float
    method: Literal["vader", "distilbert"]
    vader_compound: float


# VADER thresholds (matching current implementation)
VADER_POSITIVE_THRESHOLD = 0.25
VADER_NEGATIVE_THRESHOLD = -0.05

# Confidence thresholds - if VADER score is in these ranges, use distilBERT
# These create a "uncertainty zone" around the decision boundaries
LOW_CONFIDENCE_LOWER = -0.15  # Below this but above NEGATIVE_THRESHOLD = uncertain negative
LOW_CONFIDENCE_UPPER = 0.35   # Above NEGATIVE_THRESHOLD but below this = uncertain neutral/positive


def _load_distilbert():
    """Lazy-load distilBERT model only when needed"""
    global _distilbert_pipeline
    if _distilbert_pipeline is None:
        try:
            from transformers import pipeline
            print("[INFO] Loading distilBERT sentiment model (first use only)...")
            _distilbert_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU; set to 0 for GPU if available
            )
            print("[INFO] DistilBERT loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load distilBERT: {e}")
            print("[INFO] Will fall back to VADER-only classification")
            _distilbert_pipeline = False  # Mark as failed to avoid repeated attempts
    return _distilbert_pipeline if _distilbert_pipeline is not False else None


def _vader_classify(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[str, float]:
    """
    Classify using VADER, return (sentiment, compound_score)
    """
    scores = analyzer.polarity_scores(text or "")
    compound = scores["compound"]
    
    if compound >= VADER_POSITIVE_THRESHOLD:
        sentiment = "positive"
    elif compound <= VADER_NEGATIVE_THRESHOLD:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return sentiment, compound


def _is_low_confidence(compound: float) -> bool:
    """
    Determine if VADER score indicates low confidence
    
    Low confidence zones:
    - Near the negative/neutral boundary: -0.15 to -0.05
    - Near the neutral/positive boundary: -0.05 to 0.35
    """
    return LOW_CONFIDENCE_LOWER <= compound <= LOW_CONFIDENCE_UPPER


def _distilbert_classify(text: str) -> str | None:
    """
    Classify using distilBERT
    Returns: "positive", "negative", or None if model unavailable
    
    Note: DistilBERT SST-2 model outputs POSITIVE/NEGATIVE (no neutral),
    so we map to our 3-class system.
    """
    pipeline = _load_distilbert()
    if pipeline is None:
        return None
    
    try:
        result = pipeline(text[:512])[0]  # DistilBERT max length is 512 tokens
        label = result["label"].upper()
        
        # Map to our sentiment labels
        if label == "POSITIVE":
            return "positive"
        elif label == "NEGATIVE":
            return "negative"
        else:
            # Fallback for unexpected labels
            return "neutral"
    except Exception as e:
        print(f"[WARN] DistilBERT classification failed: {e}")
        return None


def classify_sentiment(
    text: str,
    analyzer: SentimentIntensityAnalyzer,
    use_distilbert: bool = True
) -> SentimentResult:
    """
    Classify sentiment using hybrid VADER + DistilBERT approach
    
    Args:
        text: Text to classify (typically a headline)
        analyzer: Pre-initialized VADER analyzer
        use_distilbert: If True, use distilBERT for low-confidence cases
        
    Returns:
        SentimentResult with sentiment, confidence, method used, and VADER compound score
    """
    # Step 1: Always run VADER first (fast baseline)
    vader_sentiment, vader_compound = _vader_classify(text, analyzer)
    
    # Step 2: Calculate confidence
    # High confidence = far from decision boundaries
    # Low confidence = near decision boundaries
    is_low_conf = _is_low_confidence(vader_compound)
    
    # Step 3: If low confidence and distilBERT is enabled, use it
    final_sentiment = vader_sentiment
    method = "vader"
    
    if use_distilbert and is_low_conf:
        distilbert_result = _distilbert_classify(text)
        if distilbert_result is not None:
            final_sentiment = distilbert_result
            method = "distilbert"
    
    # Calculate confidence score (0-1)
    # For VADER: based on distance from nearest threshold
    if method == "vader":
        # Distance from nearest decision boundary
        distances = [
            abs(vader_compound - VADER_NEGATIVE_THRESHOLD),
            abs(vader_compound - VADER_POSITIVE_THRESHOLD)
        ]
        min_distance = min(distances)
        # Map to 0-1 scale (0.4 distance ≈ high confidence)
        confidence = min(1.0, min_distance / 0.4)
    else:
        # DistilBERT was used, so we consider it high confidence
        confidence = 0.85  # Fixed high confidence for distilBERT
    
    return {
        "sentiment": final_sentiment,
        "confidence": confidence,
        "method": method,
        "vader_compound": vader_compound
    }


def get_simple_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> str:
    """
    Simple interface that returns just the sentiment label
    Useful for backward compatibility with existing code
    """
    result = classify_sentiment(text, analyzer)
    return result["sentiment"]


# Example usage and testing
if __name__ == "__main__":
    print("Testing Hybrid Sentiment Classifier\n")
    analyzer = SentimentIntensityAnalyzer()
    
    test_cases = [
        # Clear cases
        "This is absolutely amazing and wonderful!",  # Clear positive
        "This is terrible, horrible, and awful.",     # Clear negative
        
        # Ambiguous cases (should trigger distilBERT)
        "The company reported mixed results.",        # Ambiguous
        "Stock prices remained steady.",              # Neutral-ish
        "CEO announces changes.",                     # Very neutral
        
        # Edge cases
        "The plan seems okay but has concerns.",      # Mixed sentiment
    ]
    
    print("=" * 80)
    for text in test_cases:
        result = classify_sentiment(text, analyzer, use_distilbert=True)
        print(f"Text: {text}")
        print(f"  → Sentiment: {result['sentiment']}")
        print(f"  → Confidence: {result['confidence']:.3f}")
        print(f"  → Method: {result['method']}")
        print(f"  → VADER compound: {result['vader_compound']:.3f}")
        print()
