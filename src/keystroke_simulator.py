"""
Keystroke streaming simulator for anticipatory prefill.

This module simulates realistic user typing by streaming keystroke events
through an asyncio.Queue. It's designed to work with Rémi's inference service
for KV-cache prefill during typing.

Usage:
    queue = asyncio.Queue()
    await simulate_typing("Show me all patients", queue)
"""

import asyncio
import random
import time
from typing import Optional, Callable, Awaitable
from pathlib import Path

from .events import (
    EventType,
    KeystrokeEvent,
    StreamingSession,
    CONFIRMATION_BOUNDARIES,
    find_last_boundary,
)


class  KeystrokeSimulator:
    """
    Simulates realistic keystroke streaming from text input.
    
    Features:
    - Gaussian-distributed typing delays
    - Typo simulation with backspace correction
    - Debounce detection for flush events
    - Word/phrase boundary awareness
    
    Args:
        queue: asyncio.Queue to push KeystrokeEvent objects
        mean_delay_ms: Average delay between keystrokes
        std_delay_ms: Standard deviation of delay
        debounce_ms: Time after last keystroke to emit FLUSH
        typo_rate: Probability of typo per character (0.0-1.0)
        pause_after_boundary: Extra delay after space/punctuation
    """
    
    def __init__(
        self,
        queue: asyncio.Queue,
        mean_delay_ms: float = 150.0,
        std_delay_ms: float = 50.0,
        debounce_ms: float = 300.0,
        typo_rate: float = 0.02,
        pause_after_boundary: float = 50.0,
    ):
        self.queue = queue
        self.mean_delay_ms = mean_delay_ms
        self.std_delay_ms = std_delay_ms
        self.debounce_ms = debounce_ms
        self.typo_rate = typo_rate
        self.pause_after_boundary = pause_after_boundary
        
        self.session = StreamingSession()
        self._running = False
        self._debounce_task: Optional[asyncio.Task] = None
    
    async def _emit_event(self, event_type: EventType, char: Optional[str] = None):
        """Create and emit a keystroke event."""
        event = KeystrokeEvent(
            event_type=event_type,
            char=char,
            timestamp_ms=self.session.elapsed_ms(),
            current_text=self.session.current_text,
            confirmed_text=self.session.confirmed_text,
        )
        self.session.total_events += 1
        await self.queue.put(event)
        return event
    
    async def _debounce_flush(self):
        """Wait for debounce period then emit FLUSH if no new keystrokes."""
        try:
            await asyncio.sleep(self.debounce_ms / 1000.0)
            # Check if there's text to confirm
            boundary_pos = find_last_boundary(self.session.current_text)
            if boundary_pos > self.session.confirmed_len:
                self.session.confirmed_len = boundary_pos
                await self._emit_event(EventType.FLUSH)
        except asyncio.CancelledError:
            pass  # New keystroke cancelled the debounce
    
    def _reset_debounce(self):
        """Cancel existing debounce and start new one."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounce_flush())
    
    async def type_char(self, char: str):
        """Simulate typing a single character."""
        self.session.current_text += char
        await self._emit_event(EventType.CHAR_ADD, char)
        self._reset_debounce()
    
    async def delete_char(self):
        """Simulate pressing backspace."""
        if self.session.current_text:
            self.session.current_text = self.session.current_text[:-1]
            # Handle rollback if we deleted confirmed text
            if len(self.session.current_text) < self.session.confirmed_len:
                self.session.confirmed_len = find_last_boundary(self.session.current_text)
            await self._emit_event(EventType.CHAR_DELETE)
            self._reset_debounce()
    
    async def submit(self):
        """Simulate submitting the query (Enter key)."""
        # Cancel any pending debounce
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        
        # Confirm all remaining text
        self.session.confirmed_len = len(self.session.current_text)
        await self._emit_event(EventType.SUBMIT)
    
    def _get_typing_delay(self, char: str) -> float:
        """Get realistic delay before typing a character."""
        base_delay = max(20, random.gauss(self.mean_delay_ms, self.std_delay_ms))
        
        # Add pause after word boundaries (simulates thinking)
        if char in CONFIRMATION_BOUNDARIES:
            base_delay += self.pause_after_boundary
        
        return base_delay
    
    async def simulate_typing(
        self,
        text: str,
        submit_at_end: bool = True,
        on_event: Optional[Callable[[KeystrokeEvent], Awaitable[None]]] = None,
    ):
        """
        Simulate typing a full text string with realistic timing.
        
        Args:
            text: The text to type
            submit_at_end: Whether to emit SUBMIT event after typing
            on_event: Optional callback for each event (for logging/debugging)
        """
        self._running = True
        self.session = StreamingSession()
        
        typo_chars = "asdfghjklqwertyuiopzxcvbnm"
        
        for i, char in enumerate(text):
            if not self._running:
                break
            
            # Typing delay
            delay = self._get_typing_delay(char)
            await asyncio.sleep(delay / 1000.0)
            
            # Occasional typo simulation
            if random.random() < self.typo_rate and char.isalpha():
                # Type wrong character
                typo = random.choice(typo_chars)
                await self.type_char(typo)
                if on_event:
                    await on_event(KeystrokeEvent(
                        EventType.CHAR_ADD, typo,
                        self.session.elapsed_ms(), self.session.current_text
                    ))
                
                # Brief pause before noticing mistake
                await asyncio.sleep(random.uniform(50, 150) / 1000.0)
                
                # Delete the typo
                await self.delete_char()
                if on_event:
                    await on_event(KeystrokeEvent(
                        EventType.CHAR_DELETE, None,
                        self.session.elapsed_ms(), self.session.current_text
                    ))
                
                # Brief pause before correcting
                await asyncio.sleep(random.uniform(30, 80) / 1000.0)
            
            # Type the actual character
            await self.type_char(char)
            if on_event:
                event = KeystrokeEvent(
                    EventType.CHAR_ADD, char,
                    self.session.elapsed_ms(), self.session.current_text
                )
                await on_event(event)
        
        # Wait for final debounce
        if self._debounce_task:
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
        
        # Submit at end
        if submit_at_end:
            await self.submit()
            if on_event:
                await on_event(KeystrokeEvent(
                    EventType.SUBMIT, None,
                    self.session.elapsed_ms(), self.session.current_text
                ))
        
        # End signal
        await self._emit_event(EventType.END)
        self._running = False
    
    def stop(self):
        """Stop the simulation."""
        self._running = False
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()


async def simulate_typing(
    text: str,
    queue: asyncio.Queue,
    **kwargs,
) -> StreamingSession:
    """
    Convenience function to simulate typing text into a queue.
    
    Args:
        text: Text to type
        queue: Queue to push events to
        **kwargs: Additional arguments for KeystrokeSimulator
    
    Returns:
        The StreamingSession with statistics
    """
    simulator = KeystrokeSimulator(queue, **kwargs)
    await simulator.simulate_typing(text)
    return simulator.session


async def simulate_from_file(
    file_path: str | Path,
    queue: asyncio.Queue,
    **kwargs,
) -> StreamingSession:
    """
    Simulate typing from a text file.
    
    Args:
        file_path: Path to text file
        queue: Queue to push events to
        **kwargs: Additional arguments for KeystrokeSimulator
    
    Returns:
        The StreamingSession with statistics
    """
    path = Path(file_path)
    text = path.read_text(encoding='utf-8').strip()
    return await simulate_typing(text, queue, **kwargs)


class EventConsumer:
    """
    Base class for consuming keystroke events.
    
    Subclass this and implement handle_event() for custom processing.
    This is the interface that Rémi's inference service should implement.
    """
    
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self._running = False
    
    async def handle_event(self, event: KeystrokeEvent):
        """
        Handle a single keystroke event.
        
        Override this in subclasses to implement custom logic.
        """
        raise NotImplementedError
    
    async def run(self):
        """Consume events from the queue until END event."""
        self._running = True
        while self._running:
            event = await self.queue.get()
            await self.handle_event(event)
            self.queue.task_done()
            
            if event.event_type == EventType.END:
                self._running = False
    
    def stop(self):
        """Stop consuming events."""
        self._running = False


class LoggingConsumer(EventConsumer):
    """Simple consumer that logs events for debugging."""
    
    def __init__(self, queue: asyncio.Queue, verbose: bool = True):
        super().__init__(queue)
        self.verbose = verbose
        self.events: list[KeystrokeEvent] = []
    
    async def handle_event(self, event: KeystrokeEvent):
        self.events.append(event)
        if self.verbose:
            print(f"[{event.timestamp_ms:8.1f}ms] {event}")
