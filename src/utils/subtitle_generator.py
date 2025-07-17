"""
Subtitle Generator Module

This module provides utilities to convert speech-to-text API output to subtitle JSON format
compatible with the video processing pipeline.

Supported APIs:
- Google Cloud Speech-to-Text
- Azure Speech Service
- AWS Transcribe
- Generic word-level timestamp format
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def convert_google_speech_to_subtitles(
    transcript_path: Union[str, Path], 
    output_path: Union[str, Path],
    max_words_per_line: int = 7,
    min_line_duration: float = 1.0,
    max_line_duration: float = 5.0
) -> bool:
    """
    Convert Google Cloud Speech-to-Text API output to subtitle JSON format.
    
    Args:
        transcript_path: Path to Google Speech-to-Text JSON output
        output_path: Path to save the subtitle JSON file
        max_words_per_line: Maximum number of words per subtitle line
        min_line_duration: Minimum duration for a subtitle line in seconds
        max_line_duration: Maximum duration for a subtitle line in seconds
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Extract word-level timestamps from Google format
        words = []
        
        # Handle different Google Speech API response formats
        if 'results' in transcript_data:
            # Standard response format
            for result in transcript_data['results']:
                if 'alternatives' in result and len(result['alternatives']) > 0:
                    alt = result['alternatives'][0]
                    if 'words' in alt:
                        words.extend(alt['words'])
        else:
            logger.error("Unsupported Google Speech-to-Text format")
            return False
        
        # Group words into subtitle lines
        subtitles = []
        current_line = []
        line_start_time = None
        
        for word_info in words:
            word = word_info.get('word', '')
            start_time = float(word_info.get('startTime', '0').rstrip('s'))
            end_time = float(word_info.get('endTime', '0').rstrip('s'))
            
            if not line_start_time:
                line_start_time = start_time
                
            current_line.append(word)
            
            # Create a new line if we've reached max words or max duration
            if len(current_line) >= max_words_per_line or (end_time - line_start_time) >= max_line_duration:
                subtitles.append({
                    "text": " ".join(current_line),
                    "start_time": line_start_time,
                    "end_time": end_time
                })
                current_line = []
                line_start_time = None
        
        # Add any remaining words as the last line
        if current_line:
            subtitles.append({
                "text": " ".join(current_line),
                "start_time": line_start_time,
                "end_time": end_time
            })
        
        # Ensure minimum duration for each subtitle
        for subtitle in subtitles:
            duration = subtitle["end_time"] - subtitle["start_time"]
            if duration < min_line_duration:
                subtitle["end_time"] = subtitle["start_time"] + min_line_duration
        
        # Write subtitles to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subtitles, f, indent=2)
            
        logger.info(f"Successfully converted {len(subtitles)} subtitle lines to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting Google Speech-to-Text to subtitles: {e}")
        return False

def convert_azure_speech_to_subtitles(
    transcript_path: Union[str, Path], 
    output_path: Union[str, Path],
    max_words_per_line: int = 7,
    min_line_duration: float = 1.0,
    max_line_duration: float = 5.0
) -> bool:
    """
    Convert Azure Speech Service API output to subtitle JSON format.
    
    Args:
        transcript_path: Path to Azure Speech Service JSON output
        output_path: Path to save the subtitle JSON file
        max_words_per_line: Maximum number of words per subtitle line
        min_line_duration: Minimum duration for a subtitle line in seconds
        max_line_duration: Maximum duration for a subtitle line in seconds
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Extract word-level timestamps from Azure format
        words = []
        
        # Handle Azure Speech Service format
        if 'recognizedPhrases' in transcript_data:
            for phrase in transcript_data['recognizedPhrases']:
                if 'nBest' in phrase and len(phrase['nBest']) > 0:
                    best_result = phrase['nBest'][0]
                    if 'words' in best_result:
                        words.extend(best_result['words'])
        else:
            logger.error("Unsupported Azure Speech Service format")
            return False
        
        # Group words into subtitle lines
        subtitles = []
        current_line = []
        line_start_time = None
        
        for word_info in words:
            word = word_info.get('word', '')
            start_time = word_info.get('offsetInSeconds', 0)
            end_time = start_time + word_info.get('durationInSeconds', 0)
            
            if not line_start_time:
                line_start_time = start_time
                
            current_line.append(word)
            
            # Create a new line if we've reached max words or max duration
            if len(current_line) >= max_words_per_line or (end_time - line_start_time) >= max_line_duration:
                subtitles.append({
                    "text": " ".join(current_line),
                    "start_time": line_start_time,
                    "end_time": end_time
                })
                current_line = []
                line_start_time = None
        
        # Add any remaining words as the last line
        if current_line:
            subtitles.append({
                "text": " ".join(current_line),
                "start_time": line_start_time,
                "end_time": end_time
            })
        
        # Ensure minimum duration for each subtitle
        for subtitle in subtitles:
            duration = subtitle["end_time"] - subtitle["start_time"]
            if duration < min_line_duration:
                subtitle["end_time"] = subtitle["start_time"] + min_line_duration
        
        # Write subtitles to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subtitles, f, indent=2)
            
        logger.info(f"Successfully converted {len(subtitles)} subtitle lines to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting Azure Speech Service to subtitles: {e}")
        return False

def convert_aws_transcribe_to_subtitles(
    transcript_path: Union[str, Path], 
    output_path: Union[str, Path],
    max_words_per_line: int = 7,
    min_line_duration: float = 1.0,
    max_line_duration: float = 5.0
) -> bool:
    """
    Convert AWS Transcribe API output to subtitle JSON format.
    
    Args:
        transcript_path: Path to AWS Transcribe JSON output
        output_path: Path to save the subtitle JSON file
        max_words_per_line: Maximum number of words per subtitle line
        min_line_duration: Minimum duration for a subtitle line in seconds
        max_line_duration: Maximum duration for a subtitle line in seconds
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Extract word-level timestamps from AWS format
        words = []
        
        # Handle AWS Transcribe format
        if 'results' in transcript_data and 'items' in transcript_data['results']:
            items = transcript_data['results']['items']
            
            for item in items:
                if item['type'] == 'pronunciation':
                    word = item['alternatives'][0]['content']
                    start_time = float(item['start_time'])
                    end_time = float(item['end_time'])
                    
                    words.append({
                        'word': word,
                        'start_time': start_time,
                        'end_time': end_time
                    })
        else:
            logger.error("Unsupported AWS Transcribe format")
            return False
        
        # Group words into subtitle lines
        subtitles = []
        current_line = []
        line_start_time = None
        
        for word_info in words:
            word = word_info['word']
            start_time = word_info['start_time']
            end_time = word_info['end_time']
            
            if not line_start_time:
                line_start_time = start_time
                
            current_line.append(word)
            
            # Create a new line if we've reached max words or max duration
            if len(current_line) >= max_words_per_line or (end_time - line_start_time) >= max_line_duration:
                subtitles.append({
                    "text": " ".join(current_line),
                    "start_time": line_start_time,
                    "end_time": end_time
                })
                current_line = []
                line_start_time = None
        
        # Add any remaining words as the last line
        if current_line:
            subtitles.append({
                "text": " ".join(current_line),
                "start_time": line_start_time,
                "end_time": end_time
            })
        
        # Ensure minimum duration for each subtitle
        for subtitle in subtitles:
            duration = subtitle["end_time"] - subtitle["start_time"]
            if duration < min_line_duration:
                subtitle["end_time"] = subtitle["start_time"] + min_line_duration
        
        # Write subtitles to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subtitles, f, indent=2)
            
        logger.info(f"Successfully converted {len(subtitles)} subtitle lines to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting AWS Transcribe to subtitles: {e}")
        return False

def convert_generic_timestamps_to_subtitles(
    transcript_path: Union[str, Path], 
    output_path: Union[str, Path],
    max_words_per_line: int = 7,
    min_line_duration: float = 1.0,
    max_line_duration: float = 5.0
) -> bool:
    """
    Convert generic word-level timestamps to subtitle JSON format.
    
    Expected format:
    [
        {"word": "Hello", "start_time": 0.0, "end_time": 0.5},
        {"word": "world", "start_time": 0.6, "end_time": 1.0},
        ...
    ]
    
    Args:
        transcript_path: Path to generic word-level timestamps JSON
        output_path: Path to save the subtitle JSON file
        max_words_per_line: Maximum number of words per subtitle line
        min_line_duration: Minimum duration for a subtitle line in seconds
        max_line_duration: Maximum duration for a subtitle line in seconds
        
    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            words = json.load(f)
        
        # Group words into subtitle lines
        subtitles = []
        current_line = []
        line_start_time = None
        
        for word_info in words:
            word = word_info.get('word', '')
            start_time = word_info.get('start_time', 0)
            end_time = word_info.get('end_time', 0)
            
            if not line_start_time:
                line_start_time = start_time
                
            current_line.append(word)
            
            # Create a new line if we've reached max words or max duration
            if len(current_line) >= max_words_per_line or (end_time - line_start_time) >= max_line_duration:
                subtitles.append({
                    "text": " ".join(current_line),
                    "start_time": line_start_time,
                    "end_time": end_time
                })
                current_line = []
                line_start_time = None
        
        # Add any remaining words as the last line
        if current_line:
            subtitles.append({
                "text": " ".join(current_line),
                "start_time": line_start_time,
                "end_time": end_time
            })
        
        # Ensure minimum duration for each subtitle
        for subtitle in subtitles:
            duration = subtitle["end_time"] - subtitle["start_time"]
            if duration < min_line_duration:
                subtitle["end_time"] = subtitle["start_time"] + min_line_duration
        
        # Write subtitles to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subtitles, f, indent=2)
            
        logger.info(f"Successfully converted {len(subtitles)} subtitle lines to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting generic timestamps to subtitles: {e}")
        return False

def main():
    """Command-line interface for subtitle generation."""
    parser = argparse.ArgumentParser(description='Convert speech-to-text API output to subtitle JSON format')
    parser.add_argument('--input', required=True, help='Path to speech-to-text API output JSON file')
    parser.add_argument('--output', required=True, help='Path to save the subtitle JSON file')
    parser.add_argument('--format', required=True, choices=['google', 'azure', 'aws', 'generic'],
                        help='Format of the input file')
    parser.add_argument('--max-words', type=int, default=7, 
                        help='Maximum number of words per subtitle line')
    parser.add_argument('--min-duration', type=float, default=1.0,
                        help='Minimum duration for a subtitle line in seconds')
    parser.add_argument('--max-duration', type=float, default=5.0,
                        help='Maximum duration for a subtitle line in seconds')
    
    args = parser.parse_args()
    
    # Ensure input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert based on format
    success = False
    if args.format == 'google':
        success = convert_google_speech_to_subtitles(
            args.input, args.output, args.max_words, args.min_duration, args.max_duration
        )
    elif args.format == 'azure':
        success = convert_azure_speech_to_subtitles(
            args.input, args.output, args.max_words, args.min_duration, args.max_duration
        )
    elif args.format == 'aws':
        success = convert_aws_transcribe_to_subtitles(
            args.input, args.output, args.max_words, args.min_duration, args.max_duration
        )
    elif args.format == 'generic':
        success = convert_generic_timestamps_to_subtitles(
            args.input, args.output, args.max_words, args.min_duration, args.max_duration
        )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
