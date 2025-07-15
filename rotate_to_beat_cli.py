#!/usr/bin/env python3
import os
import sys
import argparse
import traceback
import logging

from src.utils.logging_config import setup_logging
from src.video.rotation import create_rotating_video

def main():
    """
    Main entry point for the rotate_to_beat application.
    Parses command line arguments and calls the appropriate functions.
    """
    parser = argparse.ArgumentParser(description='Rotate an image to beats in audio.')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--audio', required=True, help='Input audio path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--effects', nargs='+', choices=['color_shift', 'pulse', 'beat_marker'], 
                        help='Visual effects')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)

    # Validate input files
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    # Create the rotating video
    create_rotating_video(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        effects=args.effects,
        debug=args.debug
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        logging.error(f"Fatal error: {error}")
        traceback.print_exc()
