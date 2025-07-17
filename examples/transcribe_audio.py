#!/usr/bin/env python
"""
Transcribe Audio Script

This script transcribes audio files using the Google Cloud Speech-to-Text API
and generates subtitle files in the format required by our video processing pipeline.
"""

import os
import json
import argparse
from pathlib import Path
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Google Cloud Speech-to-Text API
    from google.cloud import speech
    from google.cloud.speech import RecognitionConfig, RecognitionAudio
    GOOGLE_API_AVAILABLE = True
except ImportError:
    logger.warning("Google Cloud Speech-to-Text API not available. Please install with: pip install google-cloud-speech")
    GOOGLE_API_AVAILABLE = False

def transcribe_audio_google(audio_path, output_path, language_code="en-US"):
    """
    Transcribe audio using Google Cloud Speech-to-Text API.
    
    Args:
        audio_path: Path to the audio file
        output_path: Path to save the transcription JSON
        language_code: Language code for transcription
        
    Returns:
        True if transcription was successful, False otherwise
    """
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google Cloud Speech-to-Text API not available")
        return False
    
    try:
        # Check for GOOGLE_APPLICATION_CREDENTIALS environment variable
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
            logger.error("Please set it to the path of your Google Cloud service account key file")
            return False
        
        # Initialize the client
        client = speech.SpeechClient()
        
        # Load the audio file
        with io.open(audio_path, "rb") as audio_file:
            content = audio_file.read()
        
        audio = RecognitionAudio(content=content)
        
        # Configure the request
        config = RecognitionConfig(
            encoding=RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,  # Adjust based on your audio file
            language_code=language_code,
            enable_word_time_offsets=True,  # Important for getting word-level timestamps
            enable_automatic_punctuation=True
        )
        
        # Send the request
        logger.info(f"Sending audio to Google Cloud Speech-to-Text API...")
        response = client.recognize(config=config, audio=audio)
        
        # Process the response to extract word-level timestamps
        words = []
        for result in response.results:
            for alternative in result.alternatives:
                for word_info in alternative.words:
                    word = {
                        "word": word_info.word,
                        "start_time": word_info.start_time.total_seconds(),
                        "end_time": word_info.end_time.total_seconds()
                    }
                    words.append(word)
        
        # Group words into subtitle lines
        subtitles = group_words_into_subtitles(words)
        
        # Save the subtitles to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subtitles, f, indent=4)
        
        logger.info(f"Transcription saved to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return False

def group_words_into_subtitles(words, max_words_per_line=7, min_line_duration=1.0, max_line_duration=5.0):
    """
    Group words into subtitle lines.
    
    Args:
        words: List of words with start_time and end_time
        max_words_per_line: Maximum number of words per subtitle line
        min_line_duration: Minimum duration for a subtitle line in seconds
        max_line_duration: Maximum duration for a subtitle line in seconds
        
    Returns:
        List of subtitle objects with text, start_time, and end_time
    """
    if not words:
        return []
    
    subtitles = []
    current_line = []
    current_start = words[0]["start_time"]
    
    for word in words:
        # If adding this word would exceed max_words_per_line or max_line_duration
        current_duration = word["end_time"] - current_start
        if (len(current_line) >= max_words_per_line or 
            current_duration > max_line_duration) and current_line:
            
            # Create a subtitle from the current line
            text = " ".join([w["word"] for w in current_line])
            end_time = current_line[-1]["end_time"]
            
            # Ensure minimum duration
            if end_time - current_start < min_line_duration:
                end_time = current_start + min_line_duration
            
            subtitles.append({
                "text": text,
                "start_time": current_start,
                "end_time": end_time
            })
            
            # Start a new line
            current_line = [word]
            current_start = word["start_time"]
        else:
            current_line.append(word)
    
    # Don't forget the last line
    if current_line:
        text = " ".join([w["word"] for w in current_line])
        end_time = current_line[-1]["end_time"]
        
        # Ensure minimum duration
        if end_time - current_start < min_line_duration:
            end_time = current_start + min_line_duration
        
        subtitles.append({
            "text": text,
            "start_time": current_start,
            "end_time": end_time
        })
    
    return subtitles

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio and generate subtitles")
    parser.add_argument("--audio", "-a", required=True, help="Path to the audio file")
    parser.add_argument("--output", "-o", default=None, help="Path to save the subtitle JSON file")
    parser.add_argument("--language", "-l", default="en-US", help="Language code for transcription")
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        audio_path = Path(args.audio)
        args.output = str(audio_path.parent / f"{audio_path.stem}_subtitles.json")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Transcribe the audio
    success = transcribe_audio_google(args.audio, args.output, args.language)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
