"""
Event schema for keystroke streaming.

This module defines the event types and data structures used to communicate
between the keystroke simulator (Lizhi) and the inference service (Rémi).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class EventType(Enum):
    """Types of keystroke events."""
    
    CHAR_ADD = "char_add"        # User typed a character
    CHAR_DELETE = "char_delete"  # User pressed backspace
    FLUSH = "flush"              # Debounce triggered - safe to confirm tokens
    SUBMIT = "submit"            # User submitted the query (Enter)
    END = "end"                  # End of input stream (for simulation)


@dataclass
class KeystrokeEvent:
    """
    Represents a single keystroke event in the streaming pipeline.
    
    Attributes:
        event_type: The type of event (CHAR_ADD, CHAR_DELETE, etc.)
        char: The character typed (only for CHAR_ADD events)
        timestamp_ms: Timestamp in milliseconds since session start
        current_text: The full text buffer after this event
        confirmed_text: Text that has been confirmed for KV-cache (up to last boundary)
    """
    event_type: EventType
    char: Optional[str] = None
    timestamp_ms: float = 0.0
    current_text: str = ""
    confirmed_text: str = ""
    
    def __post_init__(self):
        """Validate event data."""
        if self.event_type == EventType.CHAR_ADD and self.char is None:
            raise ValueError("CHAR_ADD events must include a character")
    
    def __repr__(self) -> str:
        if self.event_type == EventType.CHAR_ADD:
            return f"KeystrokeEvent(ADD '{self.char}', text='{self.current_text}')"
        elif self.event_type == EventType.CHAR_DELETE:
            return f"KeystrokeEvent(DEL, text='{self.current_text}')"
        else:
            return f"KeystrokeEvent({self.event_type.name}, text='{self.current_text}')"


@dataclass
class StreamingSession:
    """
    Maintains state for a keystroke streaming session.
    
    This is the shared state between the keystroke simulator and inference service.
    """
    session_id: str = ""
    start_time_ms: float = field(default_factory=lambda: time.time() * 1000)
    current_text: str = ""
    confirmed_len: int = 0  # Number of characters confirmed into KV-cache
    total_events: int = 0
    
    @property
    def confirmed_text(self) -> str:
        """Text that has been confirmed for KV-cache."""
        return self.current_text[:self.confirmed_len]
    
    @property
    def pending_text(self) -> str:
        """Text waiting to be confirmed."""
        return self.current_text[self.confirmed_len:]
    
    def elapsed_ms(self) -> float:
        """Time elapsed since session start."""
        return time.time() * 1000 - self.start_time_ms


# Boundary characters that trigger token confirmation
CONFIRMATION_BOUNDARIES = {' ', '\n', '\t', '.', ',', ';', ':', '!', '?', '(', ')', '"', "'"}


def find_last_boundary(text: str) -> int:
    """
    Find the index after the last confirmation boundary in text.
    
    Returns the position up to which tokens can be safely confirmed.
    """
    last_boundary_idx = -1
    for i, char in enumerate(text):
        if char in CONFIRMATION_BOUNDARIES:
            last_boundary_idx = i
    return last_boundary_idx + 1 if last_boundary_idx >= 0 else 0


def get_confirmable_text(current_text: str, already_confirmed: int) -> str:
    """
    Get the text that can be newly confirmed based on boundary rules.
    
    Args:
        current_text: The full current text buffer
        already_confirmed: Number of characters already confirmed
    
    Returns:
        The substring that can be newly confirmed (may be empty)
    """
    boundary_pos = find_last_boundary(current_text)
    if boundary_pos > already_confirmed:
        return current_text[already_confirmed:boundary_pos]
    return ""
