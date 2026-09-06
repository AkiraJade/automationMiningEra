"""Reusable Temporal State Filter with Candidate Count, Confirmation Frames & Hysteresis."""

import time
from typing import Generic, TypeVar, Optional, Tuple

T = TypeVar("T")


class TemporalStateFilter(Generic[T]):
    """Generic temporal state filter requiring N consecutive frame confirmations before state transition."""

    def __init__(self, required_frames: int = 3, hysteresis_frames: int = 1, default_state: Optional[T] = None):
        self.required_frames = max(1, required_frames)
        self.hysteresis_frames = max(0, hysteresis_frames)
        self.current_confirmed_state: Optional[T] = default_state
        self.candidate_state: Optional[T] = None
        self.candidate_count: int = 0
        self.hysteresis_count: int = 0
        self.last_update_time: float = time.time()

    def update(self, new_raw_state: Optional[T]) -> Tuple[Optional[T], bool, int]:
        """Updates filter with current frame raw detection state.
        
        Returns:
            Tuple[current_confirmed_state, is_confirmed_this_frame, candidate_count]
        """
        self.last_update_time = time.time()

        if new_raw_state is None:
            if self.current_confirmed_state is not None:
                self.hysteresis_count += 1
                if self.hysteresis_count > self.hysteresis_frames:
                    self.current_confirmed_state = None
                    self.candidate_state = None
                    self.candidate_count = 0
                    self.hysteresis_count = 0
            else:
                self.candidate_state = None
                self.candidate_count = 0
                self.hysteresis_count = 0
            return self.current_confirmed_state, False, 0

        # Reset hysteresis count on valid detection input
        self.hysteresis_count = 0

        if new_raw_state == self.current_confirmed_state:
            self.candidate_state = new_raw_state
            self.candidate_count = self.required_frames
            return self.current_confirmed_state, True, self.candidate_count

        if new_raw_state == self.candidate_state:
            self.candidate_count += 1
        else:
            self.candidate_state = new_raw_state
            self.candidate_count = 1

        if self.candidate_count >= self.required_frames:
            self.current_confirmed_state = self.candidate_state
            return self.current_confirmed_state, True, self.candidate_count

        return self.current_confirmed_state, False, self.candidate_count

    def reset(self) -> None:
        self.current_confirmed_state = None
        self.candidate_state = None
        self.candidate_count = 0
        self.hysteresis_count = 0
