"""
StreamController and KVCacheManager interfaces for anticipatory prefill.

This module defines the interface that Rémi will implement for:
- Tokenization of user input
- KV-cache management (extend, crop)
- Text generation on submit

Lizhi's WebSocket server calls these methods based on frontend events.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, AsyncIterator
from enum import Enum
import asyncio
import time

from .events import KeystrokeEvent, EventType, find_last_boundary


class GenerationStatus(Enum):
    """Status of text generation."""
    IDLE = "idle"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class CacheState:
    """Current state of the KV-cache."""
    confirmed_text: str = ""
    confirmed_token_count: int = 0
    cache_seq_len: int = 0
    is_initialized: bool = False


@dataclass
class GenerationResult:
    """Result from text generation."""
    generated_text: str = ""
    tokens_generated: int = 0
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    status: GenerationStatus = GenerationStatus.IDLE


class KVCacheManager(ABC):
    """
    Abstract interface for KV-cache management.
    
    Rémi will implement this with actual HF Transformers cache operations.
    """
    
    @abstractmethod
    async def initialize(self, base_prompt: str) -> CacheState:
        """
        Initialize cache with base prompt (system prompt + schema).
        
        Args:
            base_prompt: The fixed prefix (system prompt, schema context)
        
        Returns:
            Initial cache state
        """
        pass
    
    @abstractmethod
    async def extend(self, new_text: str) -> CacheState:
        """
        Extend the KV-cache with new confirmed text.
        
        Called on FLUSH events when new tokens are confirmed.
        
        Args:
            new_text: The new text to add to cache (delta from last confirmed)
        
        Returns:
            Updated cache state
        """
        pass
    
    @abstractmethod
    async def crop(self, target_text: str) -> CacheState:
        """
        Crop the KV-cache to match target text (rollback on deletion).
        
        Called when user deletes characters that were already cached.
        
        Args:
            target_text: The text the cache should represent after cropping
        
        Returns:
            Updated cache state after cropping
        """
        pass
    
    @abstractmethod
    async def get_state(self) -> CacheState:
        """Get current cache state."""
        pass
    
    @abstractmethod
    async def reset(self):
        """Reset cache to empty state."""
        pass


class StreamController:
    """
    Controller that bridges WebSocket events to KV-cache operations.
    
    This is the main class that Lizhi's WebSocket server interacts with.
    Rémi will provide a KVCacheManager implementation.
    
    Workflow:
    1. Frontend sends keystroke events via WebSocket
    2. WebSocket server calls StreamController methods
    3. StreamController manages state and calls KVCacheManager
    4. On submit, StreamController triggers generation
    """
    
    def __init__(
        self,
        cache_manager: KVCacheManager,
        debounce_ms: float = 300.0,
    ):
        self.cache_manager = cache_manager
        self.debounce_ms = debounce_ms
        
        # Session state
        self.current_text: str = ""
        self.confirmed_text: str = ""
        self.session_start_ms: float = 0.0
        self.generation_status = GenerationStatus.IDLE
        
        # Debounce handling
        self._debounce_task: Optional[asyncio.Task] = None
        self._flush_callback: Optional[callable] = None
    
    async def start_session(self, base_prompt: str = "") -> CacheState:
        """
        Start a new typing session.
        
        Args:
            base_prompt: System prompt + schema to prefill
        
        Returns:
            Initial cache state
        """
        self.current_text = ""
        self.confirmed_text = ""
        self.session_start_ms = time.time() * 1000
        self.generation_status = GenerationStatus.IDLE
        
        if base_prompt:
            return await self.cache_manager.initialize(base_prompt)
        return await self.cache_manager.get_state()
    
    async def on_keystroke(self, char: str) -> dict:
        """
        Handle a single character keystroke.
        
        Args:
            char: The character typed
        
        Returns:
            Status dict with current state
        """
        self.current_text += char
        self._reset_debounce()
        
        return {
            "event": "keystroke",
            "current_text": self.current_text,
            "confirmed_text": self.confirmed_text,
            "pending_text": self.current_text[len(self.confirmed_text):],
        }
    
    async def on_delete(self) -> dict:
        """
        Handle backspace/delete.
        
        Returns:
            Status dict, includes rollback info if cache was affected
        """
        if not self.current_text:
            return {"event": "delete", "current_text": "", "rollback": False}
        
        self.current_text = self.current_text[:-1]
        self._reset_debounce()
        
        # Check if we need to rollback the cache
        rollback_needed = len(self.current_text) < len(self.confirmed_text)
        
        if rollback_needed:
            # Find new confirmation boundary
            new_boundary = find_last_boundary(self.current_text)
            new_confirmed = self.current_text[:new_boundary]
            
            # Crop the cache
            await self.cache_manager.crop(new_confirmed)
            self.confirmed_text = new_confirmed
        
        return {
            "event": "delete",
            "current_text": self.current_text,
            "confirmed_text": self.confirmed_text,
            "rollback": rollback_needed,
        }
    
    async def on_flush(self) -> dict:
        """
        Handle flush event (debounce triggered).
        
        Confirms tokens up to last boundary and extends cache.
        
        Returns:
            Status dict with confirmed text and cache state
        """
        boundary_pos = find_last_boundary(self.current_text)
        new_confirmed = self.current_text[:boundary_pos]
        
        if len(new_confirmed) > len(self.confirmed_text):
            # New text to confirm
            delta = new_confirmed[len(self.confirmed_text):]
            await self.cache_manager.extend(delta)
            self.confirmed_text = new_confirmed
            
            return {
                "event": "flush",
                "confirmed_text": self.confirmed_text,
                "delta": delta,
                "cache_extended": True,
            }
        
        return {
            "event": "flush",
            "confirmed_text": self.confirmed_text,
            "delta": "",
            "cache_extended": False,
        }
    
    async def on_submit(self) -> AsyncIterator[dict]:
        """
        Handle submit event (user pressed Enter).
        
        Finalizes cache and starts generation.
        
        Yields:
            Generation progress dicts with tokens as they're generated
        """
        self.generation_status = GenerationStatus.GENERATING
        start_time = time.time() * 1000
        
        # Confirm any remaining text
        if len(self.current_text) > len(self.confirmed_text):
            remaining = self.current_text[len(self.confirmed_text):]
            await self.cache_manager.extend(remaining)
            self.confirmed_text = self.current_text
        
        yield {
            "event": "submit_start",
            "final_text": self.current_text,
            "cache_ready": True,
        }
        
        # TODO: Rémi implements actual generation here
        # For now, yield placeholder progress
        ttft = time.time() * 1000 - start_time
        
        yield {
            "event": "generation_start",
            "time_to_first_token_ms": ttft,
        }
        
        # Placeholder: Rémi will replace with actual token streaming
        yield {
            "event": "generation_complete",
            "generated_text": "[SQL generation placeholder - Rémi implements]",
            "total_time_ms": time.time() * 1000 - start_time,
        }
        
        self.generation_status = GenerationStatus.COMPLETE
    
    def set_flush_callback(self, callback: callable):
        """Set callback to be called on debounce flush."""
        self._flush_callback = callback
    
    def _reset_debounce(self):
        """Cancel existing debounce and start new one."""
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounce_handler())
    
    async def _debounce_handler(self):
        """Wait for debounce period then trigger flush."""
        try:
            await asyncio.sleep(self.debounce_ms / 1000.0)
            result = await self.on_flush()
            if self._flush_callback:
                await self._flush_callback(result)
        except asyncio.CancelledError:
            pass


# =============================================================================
# Mock Implementation for Testing (Rémi will replace with real implementation)
# =============================================================================

class MockKVCacheManager(KVCacheManager):
    """
    Mock implementation for testing WebSocket integration.
    
    Rémi will replace this with actual HF Transformers cache operations.
    """
    
    def __init__(self):
        self._state = CacheState()
        self._base_prompt = ""
    
    async def initialize(self, base_prompt: str) -> CacheState:
        self._base_prompt = base_prompt
        self._state = CacheState(
            confirmed_text=base_prompt,
            confirmed_token_count=len(base_prompt.split()),  # Rough estimate
            cache_seq_len=len(base_prompt.split()),
            is_initialized=True,
        )
        print(f"[MockCache] Initialized with base prompt ({len(base_prompt)} chars)")
        return self._state
    
    async def extend(self, new_text: str) -> CacheState:
        self._state.confirmed_text += new_text
        self._state.confirmed_token_count += len(new_text.split())
        self._state.cache_seq_len += len(new_text.split())
        print(f"[MockCache] Extended with: '{new_text}'")
        return self._state
    
    async def crop(self, target_text: str) -> CacheState:
        self._state.confirmed_text = self._base_prompt + target_text
        self._state.confirmed_token_count = len(self._state.confirmed_text.split())
        self._state.cache_seq_len = self._state.confirmed_token_count
        print(f"[MockCache] Cropped to: '{target_text}'")
        return self._state
    
    async def get_state(self) -> CacheState:
        return self._state
    
    async def reset(self):
        self._state = CacheState()
        self._base_prompt = ""
        print("[MockCache] Reset")
