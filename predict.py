import cv2
import librosa
import numpy as np
from moviepy.editor import ImageSequenceClip, AudioFileClip
from pathlib import Path
from cog import BasePredictor, Input, Path

# --- Helper Function ---
def draw_symmetric_dots_vectorized(frame, overlay, x_coords, y1_coords, y2_coords, radius, color_full, color_half):
    """Vectorized version of drawing dots. Modifies 'frame' in-place."""
    h, w = frame.shape[:2]
    y1_coords = np.clip(y1_coords, radius, h - radius)
    y2_coords = np.clip(y2_coords, radius, h - radius)

    # Create masks for full and half opacity
    mask = np.random.random(len(x_coords)) < 0.5
    overlay.fill(0)  # Clear overlay

    # Draw dots with full opacity
    full_mask = mask
    if np.any(full_mask):
        for x, y1, y2 in zip(x_coords[full_mask], y1_coords[full_mask], y2_coords[full_mask]):
            cv2.circle(overlay, (int(x), int(y1)), radius, color_full[:3], -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y2)), radius, color_full[:3], -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 1.0, frame, 1.0, 0, frame)

    # Draw dots with half opacity
    overlay.fill(0)  # Clear overlay
    half_mask = ~mask
    if np.any(half_mask):
        for x, y1, y2 in zip(x_coords[half_mask], y1_coords[half_mask], y2_coords[half_mask]):
            cv2.circle(overlay, (int(x), int(y1)), radius, color_half[:3], -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y2)), radius, color_half[:3], -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5, frame, 1.0, 0, frame)


# --- Predictor Class ---
class Predictor(BasePredictor):
    def setup(self):
# Forcing a new build
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(
        self,
        audio_file: Path = Input(description="Input audio file"),
        image_file: Path = Input(description="Optional background image", default=None),
        dot_size: int = Input(description="Size of dots in pixels", default=6),
        dot_spacing: int = Input(description="Spacing between dots in pixels", default=6),
        height: int = Input(description="Height of the output video", default=720),
        width: int = Input(description="Width of the output video", default=1280),
        max_height: int = Input(description="Maximum height of visualization as a percentage", default=30),
        dot_color: str = Input(description="Dot color in hex format", default="#00FFFF"),
        fps: int = Input(description="Frames per second", default=10),
        image_effect: str = Input(
            description="Effect to apply to the image",
            default="pulse",
            choices=["pulse", "zoom_in", "none"],
        ),
        pulse_intensity: float = Input(description="Intensity of the pulse effect", default=0.1),
        pulse_smoothing: float = Input(description="Smoothing factor for the pulse (0.0 = jerky, 0.9 = very smooth)", default=0.7),
        zoom_start: float = Input(description="Base image scale", default=1.0),
        zoom_end: float = Input(description="Ending zoom scale", default=1.2),
    ) -> Path:
        """Run a single prediction on the model"""
        
        output_path = Path("/tmp/output.mp4")

        print("Loading audio file...")
        y, sr = librosa.load(str(audio_file))
        print("Finished loading audio file")

        # Load the background image if provided
        img = None
        img_h, img_w = 0, 0
        if image_file:
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"Warning: Could not load image {image_file}. Using black background.")
                image_file = None
            else:
                img_h, img_w = img.shape[:2]

        # Calculate number of frames needed
        duration = len(y) / sr
        n_frames = int(duration * fps)
        if n_frames == 0:
            raise ValueError("Audio file is too short to produce any frames.")

        # Convert hex color to RGB
        hex_color = dot_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0]) # For OpenCV

        # Pre-calculate colors with different opacities
        color_full = (*bgr_color, 255)
        color_half = (*bgr_color, 128)

        # Pre-calculate radius and other constants
        radius = (dot_size + 1) // 2
        center_y = height // 2
        max_viz_height = int(height * max_height / 100)

        # Pre-calculate x positions
        n_dots = width // (dot_size + dot_spacing)
        x_positions = np.arange(n_dots) * (dot_size + dot_spacing) + dot_size // 2

        # --- Pre-calculate amplitudes for dot waveform and image pulse ---
        samples_per_frame = len(y) // n_frames
        frame_samples = y[:n_frames * samples_per_frame].reshape(n_frames, samples_per_frame)
        chunk_size = samples_per_frame // n_dots
        if chunk_size == 0:
                print("Warning: chunk_size is zero. Adjusting n_dots.")
                n_dots = samples_per_frame
                if n_dots == 0:
                    raise ValueError("Not enough audio samples per frame. Try a lower FPS or fewer dots.")
                chunk_size = 1
                x_positions = x_positions[:n_dots]

        frame_chunks = frame_samples[:, :chunk_size * n_dots].reshape(n_frames, n_dots, chunk_size)
        amplitudes_raw = np.abs(frame_chunks).mean(axis=2)

        pulse_factors = amplitudes_raw.max(axis=1)
        pulse_max = pulse_factors.max()
        if pulse_max == 0:
            pulse_max = 1.0 # Avoid division by zero
        pulse_factors = pulse_factors / pulse_max

        max_amp_per_frame = amplitudes_raw.max(axis=1, keepdims=True)
        max_amp_per_frame[max_amp_per_frame == 0] = 1
        amplitudes_normalized_for_dots = amplitudes_raw / max_amp_per_frame
        
        # --- End amplitude pre-calculation ---

        # Initialize frame buffer and overlay
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        overlay = np.zeros_like(frame)
        frames = []

        print("Generating frames...")
        
        base_fill_scale = 1.0
        if image_file and img is not None:
            base_fill_scale_x = width / img_w
            base_fill_scale_y = height / img_h
            base_fill_scale = max(base_fill_scale_x, base_fill_scale_y)

        current_smooth_pulse = 0.0

        for frame_idx in range(n_frames):
            # --- 1. Draw Background (Image or Black) ---
            if image_file and img is not None:
                current_effect_scale = 1.0
                if image_effect == "zoom_in":
                    progress = frame_idx / (n_frames - 1) if n_frames > 1 else 0
                    current_effect_scale = zoom_start + (zoom_end - zoom_start) * progress
                elif image_effect == "pulse":
                    target_pulse = pulse_factors[frame_idx]
                    current_smooth_pulse = (current_smooth_pulse * pulse_smoothing) + (target_pulse * (1.0 - pulse_smoothing))
                    current_effect_scale = zoom_start + (current_smooth_pulse * pulse_intensity)
                elif image_effect == "none":
                    current_effect_scale = zoom_start

                final_scale = base_fill_scale * current_effect_scale

                scaled_w = int(img_w * final_scale)
                scaled_h = int(img_h * final_scale)
                offset_x = (width - scaled_w) / 2
                offset_y = (height - scaled_h) / 2

                M = np.float32([
                    [final_scale, 0, offset_x],
                    [0, final_scale, offset_y]
                ])
                
                cv2.warpAffine(img, M, (width, height),
                               dst=frame,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
            else:
                frame.fill(0)

            # --- 2. Draw Waveform (on top of background) ---
            frame_amplitudes = amplitudes_normalized_for_dots[frame_idx] 
            
            y_offsets = np.minimum(frame_amplitudes * max_viz_height // 2, max_viz_height // 2)
            n_symmetric_dots = (y_offsets // (dot_size + dot_spacing)).astype(int) + 1

            # Draw center dots
            overlay.fill(0)
            mask = np.random.random(len(x_positions)) < 0.5

            for x in x_positions[mask]:
                cv2.circle(overlay, (int(x), center_y), radius, color_full[:3], -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 1.0, frame, 1.0, 0, frame)

            overlay.fill(0)
            for x in x_positions[~mask]:
                cv2.circle(overlay, (int(x), center_y), radius, color_half[:3], -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, frame, 1.0, 0, frame)

            # Draw symmetric dots for each level
            max_dots = int(n_symmetric_dots.max())
            for j in range(1, max_dots):
                y_pos = j * (dot_size + dot_spacing)
                valid_dots = j < n_symmetric_dots

                if np.any(valid_dots):
                    x_valid = x_positions[valid_dots]
                    draw_symmetric_dots_vectorized(
                        frame, overlay, x_valid,
                        np.full_like(x_valid, center_y + y_pos),
                        np.full_like(x_valid, center_y - y_pos),
                        radius, color_full, color_half
                    )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        # -----------------------------------------------------------------
        # --- START OF FIX: Explicitly close moviepy clips to fix hang ---
        # -----------------------------------------------------------------
        print("Encoding video...")
        
        video_clip = None
        audio_clip = None
        
        try:
            video_clip = ImageSequenceClip(frames, fps=fps)
            audio_clip = AudioFileClip(str(audio_file))
            video_clip.audio = audio_clip
            
            video_clip.write_videofile(str(output_path), fps=fps, codec='libx264',
                                preset='ultrafast',
                                audio_codec='aac',
                                threads=4,
                                logger=None)
        
        finally:
            # This is the crucial part that cleans up resources
            print("Cleaning up video and audio clips...")
            if audio_clip:
                audio_clip.close()
            if video_clip:
                video_clip.close()
            print("Clearing librosa cache to prevent warm-start hangs...")
            librosa.cache.clear()        
        # -----------------------------------------------------------------
        # --- END OF FIX ---
        # -----------------------------------------------------------------
        
        print(f"Video saved to {output_path}")
        return output_path
