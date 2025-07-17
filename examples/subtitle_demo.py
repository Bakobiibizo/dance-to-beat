#!/usr/bin/env python3
"""
Subtitle Demo Script

This script demonstrates how to use the karaoke-style subtitles feature
with the video processing pipeline.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video.rotation import create_rotating_video
from src.utils.logging_config import setup_logging

def main():
    """Run the subtitle demo."""
    parser = argparse.ArgumentParser(description='Demonstrate karaoke-style subtitles with video effects.')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--audio', required=True, help='Input audio path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--subtitle-path', default=None, 
                        help='Path to subtitle JSON file (defaults to sample_lyrics.json)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    logger = setup_logging(args.debug)

    # Ensure input files exist
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return 1
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return 1

    # Use sample lyrics if no subtitle path is provided
    subtitle_path = args.subtitle_path
    if not subtitle_path:
        subtitle_path = os.path.join(os.path.dirname(__file__), 'sample_lyrics.json')
        logger.info(f"Using sample lyrics from: {subtitle_path}")
    
    if not os.path.exists(subtitle_path):
        logger.error(f"Subtitle file not found: {subtitle_path}")
        return 1

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create video with subtitles and effects
    create_rotating_video(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        effects=['color_shift', 'pulse', 'subtitle'],  # Include subtitle effect
        subtitle_path=subtitle_path,
        debug=args.debug
    )
    
    logger.info(f"Video with subtitles created: {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
