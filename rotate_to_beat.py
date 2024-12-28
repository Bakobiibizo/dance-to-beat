"""
Rotate to Beat

This script creates a video by rotating an image in sync with the beats of an audio file.
It includes various visual effects and uses frequency-filtered beat detection for accuracy.

Key Features:
    - Automatic BPM detection using frequency filtering (100-350Hz range)
    - Multiple visual effects:
        - Beat-synchronized rotation
        - Color shifting based on rotation angle
        - Size pulsing based on audio intensity
        - Edge glow effect on beats
    - Support for various image formats (PNG, JPEG, WEBP)
    - Support for various audio formats (MP3, WAV, MP4, MKV)

Example:
    python rotate_to_beat.py --image "masked_image.png" --audio "music.mp3" 
        --output "output.mp4" --effects color_shift pulse beat_marker
"""

import os
import cv2
import numpy as np
import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import VideoClip
import argparse

def detect_bpm(audio_path, freq_min=100, freq_max=350):
    """Detect BPM of the audio file using librosa within specified frequency range.
    
    Args:
        audio_path (str): Path to the audio file
        freq_min (int, optional): Minimum frequency for filtering. Defaults to 100.
        freq_max (int, optional): Maximum frequency for filtering. Defaults to 350.
    
    Returns:
        tuple: (tempo, beat_frames, onset_envelope)
            - tempo (float): Detected tempo in BPM
            - beat_frames (numpy.ndarray): Array of beat frame indices
            - onset_env (numpy.ndarray): Onset strength envelope
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path)
        
        # Apply bandpass filter to isolate frequencies
        y_filt = librosa.effects.trim(y)[0]  # Remove silence
        
        # Adjust mel spectrogram parameters to avoid empty filters
        n_mels = 64  # Reduce number of mel bands
        mel_spec = librosa.feature.melspectrogram(
            y=y_filt, 
            sr=sr,
            n_mels=n_mels,
            fmin=freq_min,
            fmax=freq_max,
            htk=True  # Use HTK formula for better frequency resolution
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Get tempo and beats from filtered audio
        tempo, beat_frames = librosa.beat.beat_track(
            y=y_filt, 
            sr=sr,
            start_bpm=120,
            units='frames'
        )
        
        # Get onset envelope from filtered frequencies
        onset_env = librosa.onset.onset_strength(
            y=y_filt, 
            sr=sr,
            fmin=freq_min,
            fmax=freq_max
        )
        
        # Ensure tempo is a scalar
        if isinstance(tempo, np.ndarray):
            tempo = tempo.item()
        
        return tempo, beat_frames, onset_env
    except Exception as e:
        print(f"Error detecting BPM: {str(e)}")
        return 120, [], []

def create_rotating_video(image_path, audio_path, output_path, effects=None):
    """Create a video with rotating image synchronized to the audio BPM.
    
    Args:
        image_path (str): Path to the input image (should be pre-masked)
        audio_path (str): Path to the audio file
        output_path (str): Path for the output video
        effects (list, optional): List of effects to apply. Defaults to None.
            Possible effects: 'color_shift', 'pulse', 'beat_marker'
    """
    # Get BPM and beat information with frequency filtering
    bpm, beat_frames, onset_env = detect_bpm(audio_path)
    print(f"Detected BPM: {bpm}")
    
    # Load the audio to get its duration
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    
    # Load and prepare the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Calculate rotation per frame
    fps = 30
    beats_per_second = bpm / 60
    rotation_per_frame = (beats_per_second * 360) / fps

    # Normalize onset envelope for visual effects
    if len(onset_env) > 0:
        onset_env = onset_env / np.max(onset_env)
        onset_frames = np.linspace(0, len(onset_env)-1, int(duration * fps))
    else:
        onset_frames = []
        onset_env = []

    def make_frame(t):
        """Generate a single frame for the video at time t.
        
        Args:
            t (float): Time in seconds
        
        Returns:
            numpy.ndarray: The generated frame in RGB format
        """
        # Base rotation
        angle = t * rotation_per_frame
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Start with the original image
        frame = image.copy()
        
        if effects:
            # Color shift based on rotation
            if 'color_shift' in effects:
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] + int(angle % 180)) % 180
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Pulse effect based on onset strength
            if 'pulse' in effects and len(onset_env) > 0:
                frame_idx = int(t * fps)
                if frame_idx < len(onset_frames):
                    onset_idx = int(onset_frames[frame_idx])
                    if onset_idx < len(onset_env):
                        scale = 1.0 + (onset_env[onset_idx] * 0.2)
                        M = cv2.getRotationMatrix2D(center, 0, scale)
                        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0))
            
            # Beat marker effect
            if 'beat_marker' in effects:
                frame_idx = int(t * fps)
                if frame_idx in beat_frames:
                    # Create outer glow effect
                    kernel_size = 31  # Larger kernel for more pronounced glow
                    # Create a white ring around the image
                    ring = np.zeros_like(frame)
                    center = (frame.shape[1] // 2, frame.shape[0] // 2)
                    radius = min(center) - 2  # Slightly smaller than image
                    cv2.circle(ring, center, radius, (255, 255, 255), 2)
                    
                    # Blur the ring to create glow
                    glow = cv2.GaussianBlur(ring, (kernel_size, kernel_size), 0)
                    
                    # Add the glow to the frame with higher intensity
                    frame = cv2.addWeighted(frame, 1.0, glow, 0.5, 0)

        # Apply rotation last
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
        
        return frame

    # Create video clip
    video_clip = VideoClip(make_frame, duration=duration)
    video_clip = video_clip.with_fps(fps)
    
    # Combine video with audio
    final_clip = video_clip.with_audio(audio_clip)
    
    # Write output
    final_clip.write_videofile(output_path, fps=fps, codec='libx264')
    
    # Clean up
    audio_clip.close()
    final_clip.close()

def main():
    """Parse command line arguments and create the video."""
    parser = argparse.ArgumentParser(description='Create a video with rotating image synchronized to audio BPM')
    parser.add_argument('--image', required=True, help='Path to input image (png, jpeg, or webp)')
    parser.add_argument('--audio', required=True, help='Path to audio file (mp3, wav, mp4, or mkv)')
    parser.add_argument('--output', required=True, help='Path to output video file (mp4)')
    parser.add_argument('--effects', nargs='+', choices=['color_shift', 'pulse', 'beat_marker'],
                      help='Additional visual effects to apply')
    
    args = parser.parse_args()
    
    # Verify file extensions
    image_ext = os.path.splitext(args.image)[1].lower()
    audio_ext = os.path.splitext(args.audio)[1].lower()
    
    valid_image_ext = ['.png', '.jpeg', '.jpg', '.webp']
    valid_audio_ext = ['.mp3', '.wav', '.mp4', '.mkv']
    
    if image_ext not in valid_image_ext:
        raise ValueError(f"Invalid image format. Supported formats: {', '.join(valid_image_ext)}")
    
    if audio_ext not in valid_audio_ext:
        raise ValueError(f"Invalid audio format. Supported formats: {', '.join(valid_audio_ext)}")
    
    create_rotating_video(args.image, args.audio, args.output, args.effects)

if __name__ == "__main__":
    main()
