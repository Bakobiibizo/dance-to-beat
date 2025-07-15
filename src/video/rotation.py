import os
import cv2
import numpy as np
import logging
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from src.audio.beat_detection import get_beats
from src.audio.bpm_detection import detect_bpm

class FrameGenerator:
    """
    Class to handle image loading and preparation for video generation.
    """
    def __init__(self, path):
        self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"Failed to load image: {path}")
        if self.image.shape[2] == 4:
            logging.info("Alpha channel detected, dropping alpha")
            self.image = self.image[:, :, :3]
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

def create_rotating_video(image_path, audio_path, output_path, effects=None, debug=False):
    """
    Create a video with a rotating image synchronized to audio beats.
    
    Args:
        image_path: Path to the image file
        audio_path: Path to the audio file
        output_path: Path for the output video
        effects: List of visual effects to apply ('pulse', 'color_shift', etc.)
        debug: Enable debug mode
    """
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"Image: {image_path}, Audio: {audio_path}, Output: {output_path}, Effects: {effects}")
    effects = effects or []
    temp_video = 'output/temp_video.mp4'

    try:
        with AudioFileClip(audio_path) as audio_clip:
            bpm = detect_bpm(audio_path)
            beat_frames = get_beats(audio_path)
            if len(beat_frames) == 0:
                logging.warning("No beats detected")
            fps = 30
            duration = audio_clip.duration
            rotation_per_frame = (bpm / 60.0) * 360 / fps

            frame_gen = FrameGenerator(image_path)

            def make_frame(t):
                frame = frame_gen.image.copy()
                height, width = frame.shape[:2]
                center = (width // 2, height // 2)
                angle = (t * rotation_per_frame) % 360

                # Initialize scale for pulsing/bouncing effect
                scale = 1.0
                frame_idx = int(t * fps)
                
                # Bass-driven pulsing effect
                if 'pulse' in effects:
                    current_frame = int(t * fps)
                    
                    # Strong pulse based on bass frequencies
                    if frame_idx < len(beat_frames):
                        bass_idx = int(beat_frames[frame_idx])
                        if 0 <= bass_idx < len(beat_frames):
                            # Apply stronger scaling effect for more visible bouncing
                            # Square the bass value to make peaks more pronounced
                            bass_value = beat_frames[bass_idx]
                            # Scale from 1.0 to 1.4 based on bass intensity
                            scale = 1.0 + (bass_value ** 2) * 0.4
                    
                    # Also check for exact beat hits for extra emphasis
                    if not isinstance(beat_frames, np.ndarray):
                        beat_frames_array = np.array(beat_frames)
                    else:
                        beat_frames_array = beat_frames
                    
                    # Add extra emphasis on exact beat frames
                    if len(beat_frames_array) > 0:
                        distances = np.abs(beat_frames_array - current_frame)
                        min_distance = np.min(distances)
                        
                        # Extra boost exactly on beats
                        if min_distance == 0:  # Exactly on a beat
                            scale = max(scale, 1.4)  # Ensure maximum scale on exact beat
                        elif min_distance == 1:  # One frame away from beat
                            scale = max(scale, 1.3)  # Strong emphasis near beat

                M = cv2.getRotationMatrix2D(center, angle, scale)
                frame = cv2.warpAffine(frame, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

                if 'color_shift' in effects:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    hue_shift = int(angle / 2) % 180
                    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
                # Safer beat marker effect implementation
                if 'beat_marker' in effects:
                    # Check if current frame is on or near a beat
                    current_frame = int(t * fps)
                    beat_intensity = 0.0
                    
                    # Convert beat_frames to numpy array if it's a list
                    if not isinstance(beat_frames, np.ndarray):
                        beat_frames_array = np.array(beat_frames)
                    else:
                        beat_frames_array = beat_frames
                    
                    # Only check the 5 nearest beats for efficiency
                    if len(beat_frames_array) > 0:  # Make sure we have beats to check
                        # Find the closest beat to current frame
                        distances = np.abs(beat_frames_array - current_frame)
                        # Get indices of up to 5 closest beats (or fewer if we don't have 5)
                        num_beats_to_check = min(5, len(beat_frames_array))
                        closest_beats_idx = np.argsort(distances)[:num_beats_to_check]
                        closest_beats = beat_frames_array[closest_beats_idx]
                        
                        for beat_frame in closest_beats:
                            distance = abs(current_frame - beat_frame)
                            if distance < 3:  # Within 3 frames of a beat
                                # Linear falloff with distance
                                intensity = 1.0 - (distance / 3.0)
                                beat_intensity = max(beat_intensity, intensity)
                    
                    # Apply a very simple brightness boost if this is a beat frame
                    if beat_intensity > 0:
                        # Modest brightness boost that won't corrupt the image
                        brightness = np.ones_like(frame) * 20 * beat_intensity
                        frame = np.clip(frame + brightness, 0, 255)

                frame = np.clip(frame, 0, 255).astype(np.uint8)
                return frame

            frame_size = (frame_gen.image.shape[1], frame_gen.image.shape[0])
            writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

            if not writer.isOpened():
                raise RuntimeError("Failed to open video writer. Ensure 'mp4v' codec is available or try .avi")

            for i in range(int(duration * fps)):
                t = i / fps
                frame = make_frame(t)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
                if i % 100 == 0:
                    logging.info(f"Frame {i}/{int(duration * fps)}")
            writer.release()
            if int(t * fps) < 3:
                cv2.imwrite(f"debug_frame_{int(t*fps)}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            with VideoFileClip(temp_video) as video:
                final = video.with_audio(audio_clip)
                final.write_videofile(output_path, fps=fps)

    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)
            logging.info("Temporary video file removed")
