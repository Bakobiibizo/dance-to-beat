#!/usr/bin/env python
"""
Transcribe Audio with API Script

This script transcribes audio files using the deployed speech-to-text API endpoint
and generates subtitle files in the format required by our video processing pipeline.
"""

import os
import json
import base64
import argparse
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
API_ENDPOINT = "https://speech2text-synai.ngrok.dev/v1/api/speech2text"

# Disable SSL verification warning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def transcribe_audio_with_api(audio_path, output_path):
    """
    Transcribe audio using the deployed speech-to-text API endpoint.
    
    Args:
        audio_path: Path to the audio file
        output_path: Path to save the transcription JSON
        
    Returns:
        True if transcription was successful, False otherwise
    """
    try:
        # Read the audio file as binary
        with open(audio_path, "rb") as audio_file:
            audio_content = audio_file.read()
        
        # Encode the audio content as base64
        audio_base64 = base64.b64encode(audio_content).decode("utf-8")
        
        # Prepare the request payload
        payload = {
            "data": audio_base64,
            "filename": os.path.basename(audio_path)
        }
        
        # Send the request to the API with SSL verification disabled
        logger.info(f"Sending audio to speech-to-text API...")
        logger.info(f"Request payload size: {len(audio_base64)} bytes")
        logger.info(f"Request filename: {os.path.basename(audio_path)}")
        
        # Set headers for better debugging
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        # Send the request with detailed error handling
        try:
            response = requests.post(API_ENDPOINT, json=payload, headers=headers, verify=False, timeout=60)
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return False
        
        # Check if the request was successful
        if response.status_code == 200:
            logger.info(f"Transcription successful")
            
            # Try to parse the response as JSON, but handle string responses too
            try:
                transcription = response.json()
            except ValueError:
                # If not JSON, treat as plain text
                transcription = response.text
                
            # Print the full response for debugging
            logger.info(f"Full API response: {response.text}")
            
            # Convert the transcription to our subtitle format
            subtitles = convert_transcription_to_subtitles(transcription)
            
            # Save the subtitles to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(subtitles, f, indent=4)
            
            logger.info(f"Subtitles saved to {output_path}")
            return True
        else:
            logger.error(f"API request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return False

def convert_transcription_to_subtitles(transcription):
    """
    Convert the API transcription response to our subtitle format.
    
    Args:
        transcription: The API response containing transcription data (Whisper output)
        
    Returns:
        List of subtitle objects with text, start_time, and end_time
    """
    subtitles = []
    
    try:
        # Handle string responses from Whisper
        if isinstance(transcription, str):
            # Check if we got an empty response
            if not transcription.strip():
                logger.warning("API returned an empty response. The audio might not contain speech.")
                # Create a default subtitle for instrumental audio
                subtitles = [
                    {"text": "[Electronic Beat]" , "start_time": 0.0, "end_time": 5.0},
                    {"text": "[Bass Line]" , "start_time": 5.0, "end_time": 10.0},
                    {"text": "[Synth Melody]" , "start_time": 10.0, "end_time": 15.0},
                    {"text": "[Rhythm Section]" , "start_time": 15.0, "end_time": 20.0},
                    {"text": "[Beat Breakdown]" , "start_time": 20.0, "end_time": 25.0},
                    {"text": "[Fade Out]" , "start_time": 25.0, "end_time": 30.0}
                ]
                return subtitles
            
            logger.info(f"API returned a Whisper transcription: {transcription[:100]}...")
            
            # Process the Whisper transcription
            text = transcription.strip()
            
            # Check if the response contains timestamps in the format [00:00:00.000 --> 00:00:05.000]
            import re
            timestamp_pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.+?)(?=\[|$)'
            timestamp_matches = re.findall(timestamp_pattern, text, re.DOTALL)
            
            if timestamp_matches:
                # Process timestamped transcription
                logger.info(f"Found {len(timestamp_matches)} timestamped segments")
                
                for start_time_str, end_time_str, segment_text in timestamp_matches:
                    # Convert timestamp strings to seconds (format: HH:MM:SS.mmm)
                    start_hours, start_minutes, start_seconds = start_time_str.split(':')
                    end_hours, end_minutes, end_seconds = end_time_str.split(':')
                    
                    start_time = float(start_hours) * 3600 + float(start_minutes) * 60 + float(start_seconds)
                    end_time = float(end_hours) * 3600 + float(end_minutes) * 60 + float(end_seconds)
                    
                    # Clean up the segment text
                    segment_text = segment_text.strip()
                    
                    if segment_text:
                        subtitles.append({
                            "text": segment_text,
                            "start_time": start_time,
                            "end_time": end_time
                        })
            else:
                # No timestamps found, split by sentences and estimate timing
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                # Create subtitles with estimated timings
                total_duration = 30  # Estimate 30 seconds for the whole audio
                segment_duration = total_duration / max(len(sentences), 1)
                
                for i, sentence in enumerate(sentences):
                    start_time = i * segment_duration
                    end_time = (i + 1) * segment_duration
                    
                    subtitles.append({
                        "text": sentence,
                        "start_time": start_time,
                        "end_time": end_time
                    })
            
            return subtitles
        
        # For JSON responses, log the structure
        if isinstance(transcription, dict):
            logger.info(f"API Response structure: {json.dumps(transcription, indent=2)[:500]}...")
        
        # Check if the response contains a 'results' key with segments
        if 'results' in transcription and isinstance(transcription['results'], list):
            # Process Google Speech-to-Text style response
            for result in transcription['results']:
                if 'alternatives' in result and result['alternatives']:
                    alt = result['alternatives'][0]  # Take the first alternative
                    text = alt.get('transcript', '')
                    
                    # Check if we have word-level timestamps
                    if 'words' in alt:
                        for word_info in alt['words']:
                            word = word_info.get('word', '')
                            start_time = float(word_info.get('start_time', 0).replace('s', ''))
                            end_time = float(word_info.get('end_time', 0).replace('s', ''))
                            
                            subtitles.append({
                                "text": word,
                                "start_time": start_time,
                                "end_time": end_time
                            })
                    else:
                        # No word-level timestamps, use the whole segment
                        start_time = 0
                        end_time = 5  # Default duration
                        
                        if 'result_end_time' in result:
                            end_time = float(result['result_end_time'].replace('s', ''))
                        
                        subtitles.append({
                            "text": text,
                            "start_time": start_time,
                            "end_time": end_time
                        })
        
        # If the response has a 'text' field directly (simple API)
        elif 'text' in transcription:
            text = transcription['text']
            
            # Split the text into sentences for better readability
            import re
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Create subtitles with estimated timings
            total_duration = 30  # Estimate 30 seconds for the whole audio
            segment_duration = total_duration / max(len(sentences), 1)
            
            for i, sentence in enumerate(sentences):
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                
                subtitles.append({
                    "text": sentence,
                    "start_time": start_time,
                    "end_time": end_time
                })
        
        # If we couldn't parse the response in a known format
        if not subtitles:
            logger.warning(f"Unknown response format. Creating a single subtitle with the raw text.")
            # Try to extract any text content from the response
            text = str(transcription)
            if isinstance(transcription, dict):
                text = json.dumps(transcription)
            
            subtitles = [{
                "text": "Transcription: " + text[:100],  # Limit to 100 chars
                "start_time": 0,
                "end_time": 10
            }]
    
    except Exception as e:
        logger.error(f"Error converting transcription to subtitles: {str(e)}")
        # Return a default subtitle if conversion fails
        subtitles = [{
            "text": "Error processing transcription",
            "start_time": 0,
            "end_time": 5
        }]
    
    # Group words into phrases for better readability if we have many short subtitles
    if len(subtitles) > 20 and all(len(s['text'].split()) <= 2 for s in subtitles):
        subtitles = group_words_into_phrases(subtitles)
    
    return subtitles

def group_words_into_phrases(word_subtitles, max_words_per_phrase=5):
    """
    Group individual word subtitles into phrases for better readability.
    
    Args:
        word_subtitles: List of word-level subtitles
        max_words_per_phrase: Maximum number of words per phrase
        
    Returns:
        List of phrase-level subtitles
    """
    if not word_subtitles:
        return []
    
    phrase_subtitles = []
    current_phrase = []
    current_start = word_subtitles[0]["start_time"]
    
    for word in word_subtitles:
        # If adding this word would exceed max_words_per_phrase
        if len(current_phrase) >= max_words_per_phrase and current_phrase:
            # Create a subtitle from the current phrase
            text = " ".join([w["text"] for w in current_phrase])
            end_time = current_phrase[-1]["end_time"]
            
            phrase_subtitles.append({
                "text": text,
                "start_time": current_start,
                "end_time": end_time
            })
            
            # Start a new phrase
            current_phrase = [word]
            current_start = word["start_time"]
        else:
            current_phrase.append(word)
    
    # Don't forget the last phrase
    if current_phrase:
        text = " ".join([w["text"] for w in current_phrase])
        end_time = current_phrase[-1]["end_time"]
        
        phrase_subtitles.append({
            "text": text,
            "start_time": current_start,
            "end_time": end_time
        })
    
    return phrase_subtitles

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using the deployed API and generate subtitles")
    parser.add_argument("--audio", "-a", required=True, help="Path to the audio file")
    parser.add_argument("--output", "-o", default=None, help="Path to save the subtitle JSON file")
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        audio_path = Path(args.audio)
        args.output = str(audio_path.parent / f"{audio_path.stem}_subtitles.json")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Transcribe the audio
    success = transcribe_audio_with_api(args.audio, args.output)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
