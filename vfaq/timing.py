import math
import logging

class TimingResolver:
    """
    Resolves and normalizes timing parameters (frames, fps, duration) based on a
    specified authority.

    The class ensures consistency among the three variables F (frames), R (fps),
    and D (duration) based on the invariant: D = F / R.

    Rules:
    - F = frames (integer only)
    - R = fps (float allowed)
    - D = duration (float with 3 decimal precision)
    - Only round frames.
    - Do NOT round fps.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def resolve(self, fps: float = None, frames: int = None, duration: float = None, authority: str = None) -> dict:
        """
        Resolves timing parameters based on the given authority.

        Args:
            fps (float): Frames per second.
            frames (int): Number of frames.
            duration (float): Duration in seconds.
            authority (str): The timing authority ("frames", "duration", or "fps").

        Returns:
            dict: A dictionary containing the resolved 'resolved_fps', 'resolved_frames',
                  and 'resolved_duration'.
        """
        initial_fps = fps
        initial_frames = frames
        initial_duration = duration

        self.logger.info(
            f"[Timing Normalize] authority={authority}\n"
            f"input: fps={initial_fps}, frames={initial_frames}, duration={initial_duration}"
        )

        resolved_fps = initial_fps
        resolved_frames = initial_frames
        resolved_duration = initial_duration

        if authority == "frames":
            if frames is None:
                raise ValueError("When authority is 'frames', 'frames' must be provided.")
            
            resolved_frames = frames # Frames are truth.

            if duration is not None:
                if duration == 0:
                    raise ValueError("Duration cannot be zero when frames authority is used with duration.")
                resolved_fps = resolved_frames / duration
                resolved_duration = duration
            elif fps is not None:
                if fps == 0:
                    raise ValueError("FPS cannot be zero when frames authority is used with fps.")
                resolved_duration = resolved_frames / fps
                resolved_fps = fps
            else:
                raise ValueError("When authority is 'frames', either 'duration' or 'fps' must be provided if not all three are present.")

        elif authority == "duration":
            if duration is None:
                raise ValueError("When authority is 'duration', 'duration' must be provided.")
            
            resolved_duration = duration # Duration is truth.

            if frames is not None:
                if resolved_duration == 0:
                    raise ValueError("Duration cannot be zero when duration authority is used with frames.")
                resolved_fps = frames / resolved_duration
                resolved_frames = frames
            elif fps is not None:
                resolved_frames = math.floor(resolved_duration * fps)
                resolved_fps = fps
            else:
                raise ValueError("When authority is 'duration', either 'frames' or 'fps' must be provided if not all three are present.")

        elif authority == "fps":
            if fps is None:
                raise ValueError("When authority is 'fps', 'fps' must be provided.")
            
            resolved_fps = fps # FPS is truth.

            if duration is not None:
                resolved_frames = round(duration * resolved_fps)
                resolved_duration = duration
            elif frames is not None:
                if resolved_fps == 0:
                    raise ValueError("FPS cannot be zero when fps authority is used with frames.")
                resolved_duration = frames / resolved_fps
                resolved_frames = frames
            else:
                raise ValueError("When authority is 'fps', either 'duration' or 'frames' must be provided if not all three are present.")
        else:
            raise ValueError(f"Unknown timing authority: {authority}. Must be 'frames', 'duration', or 'fps'.")

        # Ensure frames is an integer
        resolved_frames = int(resolved_frames) if resolved_frames is not None else None

        # Ensure duration has 3 decimal precision
        resolved_duration = round(resolved_duration, 3) if resolved_duration is not None else None

        # Check for any remaining None values after resolution - this should be caught by ValueErrors above if logic is perfect
        if resolved_fps is None or resolved_frames is None or resolved_duration is None:
            raise RuntimeError(f"Timing resolution incomplete: fps={resolved_fps}, frames={resolved_frames}, duration={resolved_duration}")


        self.logger.info(
            f"resolved: fps={resolved_fps}, frames={resolved_frames}, duration={resolved_duration:.3f}"
        )

        return {
            "resolved_fps": resolved_fps,
            "resolved_frames": resolved_frames,
            "resolved_duration": resolved_duration
        }
