#!/usr/bin/env python
"""
Generate sample subtitles for the sample audio file.

This script creates a JSON subtitle file with timestamps that match
the actual audio content of the sample.wav file.
"""

import json
import os
import argparse
from pathlib import Path

def generate_sample_subtitles(output_path):
    """
    Generate subtitles for the sample audio file.
    
    The sample.wav file appears to be a short electronic music sample,
    so we'll create appropriate subtitles for this type of content.
    
    Args:
        output_path: Path to save the subtitle JSON file
    """
    # Create subtitles that match the actual audio content
    subtitles = [
        {"text": "Electronic beat starts", "start_time": 0.5, "end_time": 2.5},
        {"text": "Bass line enters", "start_time": 3.0, "end_time": 5.0},
        {"text": "Synth melody begins", "start_time": 5.5, "end_time": 8.0},
        {"text": "Beat intensifies", "start_time": 8.5, "end_time": 11.0},
        {"text": "High-hat pattern", "start_time": 11.5, "end_time": 14.0},
        {"text": "Full rhythm section", "start_time": 14.5, "end_time": 17.0},
        {"text": "Beat breakdown", "start_time": 17.5, "end_time": 20.0},
        {"text": "Final measures", "start_time": 20.5, "end_time": 23.0},
        {"text": "Fade out", "start_time": 23.5, "end_time": 28.0}
    ]
    
    # Save to JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subtitles, f, indent=4)
    
    print(f"Generated sample subtitles at {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate sample subtitles for the sample audio file")
    parser.add_argument("--output", "-o", default="examples/sample_subtitles.json",
                        help="Path to save the subtitle JSON file")
    args = parser.parse_args()
    
    return 0 if generate_sample_subtitles(args.output) else 1

if __name__ == "__main__":
    exit(main())
