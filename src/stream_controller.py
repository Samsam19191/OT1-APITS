"""
StreamController and KVCacheManager interfaces for anticipatory prefill.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, AsyncIterator
from enum import Enum
import asyncio
import time

from .events import find_last_boundary


class GenerationStatus(Enum):
    IDLE = "idle"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class CacheState:
    confirmed_text: str = ""
    confirmed_token_count: int = 0
    cache_seq_len: int = 0
    is_initialized: bool = False


class KVCacheManager(ABC):
    @abstractmethod
    async def initialize(self, base_prompt: str) -> CacheState:
        pass

    @abstractmethod
    async def extend(self, new_text: str) -> CacheState:
        pass

    @abstractmethod
    async def crop(self, target_text: str) -> CacheState:
        pass

    @abstractmethod
    async def get_state(self) -> CacheState:
        pass

    @abstractmethod
    async def reset(self):
        pass


class StreamController:
    def __init__(
        self,
        cache_manager: KVCacheManager,
        debounce_ms: float = 300.0,
    ):
        self.cache_manager = cache_manager
        self.debounce_ms = debounce_ms

        self.current_text: str = ""
        self.confirmed_text: str = ""
        self.session_start_ms: float = 0.0
        self.generation_status = GenerationStatus.IDLE

        self._debounce_task: Optional[asyncio.Task] = None
        self._flush_callback: Optional[callable] = None

    async def start_session(self, base_prompt: str = "") -> CacheState:
        self.current_text = ""
        self.confirmed_text = ""
        self.session_start_ms = time.time() * 1000
        self.generation_status = GenerationStatus.IDLE

        if base_prompt:
            return await self.cache_manager.initialize(base_prompt)
        return await self.cache_manager.get_state()

    async def on_text_update(self, text: str) -> dict:
        """
        Handle full text update from frontend.
        Computes diff and updates cache accordingly.
        """
        old_text = self.current_text
        self.current_text = text
        self._reset_debounce()

        # Check if we need to rollback (text got shorter than confirmed)
        if len(text) < len(self.confirmed_text):
            new_boundary = find_last_boundary(text)
            new_confirmed = text[:new_boundary]
            await self.cache_manager.crop(new_confirmed)
            self.confirmed_text = new_confirmed

        return {
            "event": "text_update",
            "current_text": self.current_text,
            "confirmed_text": self.confirmed_text,
            "pending_text": self.current_text[len(self.confirmed_text):],
        }

    async def on_text_submit(self, text: str) -> AsyncIterator[dict]:
        """
        Handle text submit (user pressed Enter).
        """
        self.current_text = text
        self.generation_status = GenerationStatus.GENERATING
        start_time = time.time() * 1000

        # Cancel any pending debounce
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        # Confirm all remaining text
        if len(self.current_text) > len(self.confirmed_text):
            remaining = self.current_text[len(self.confirmed_text):]
            await self.cache_manager.extend(remaining)
            self.confirmed_text = self.current_text

        yield {
            "event": "submit_start",
            "final_text": self.current_text,
            "cache_ready": True,
        }

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

    async def on_flush(self) -> dict:
        """Handle flush event (debounce triggered)."""
        boundary_pos = find_last_boundary(self.current_text)
        new_confirmed = self.current_text[:boundary_pos]

        if len(new_confirmed) > len(self.confirmed_text):
            delta = new_confirmed[len(self.confirmed_text):]
            await self.cache_manager.extend(delta)
            self.confirmed_text = new_confirmed

            return {
                "event": "text_update",
                "current_text": self.current_text,
                "confirmed_text": self.confirmed_text,
                "pending_text": self.current_text[len(self.confirmed_text):],
            }

        return {
            "event": "text_update",
            "current_text": self.current_text,
            "confirmed_text": self.confirmed_text,
            "pending_text": self.current_text[len(self.confirmed_text):],
        }

    def set_flush_callback(self, callback: callable):
        self._flush_callback = callback

    def _reset_debounce(self):
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._debounce_handler())

    async def _debounce_handler(self):
        try:
            await asyncio.sleep(self.debounce_ms / 1000.0)
            result = await self.on_flush()
            if self._flush_callback:
                await self._flush_callback(result)
        except asyncio.CancelledError:
            pass


class MockKVCacheManager(KVCacheManager):
    def __init__(self):
        self._state = CacheState()
        self._base_prompt = ""

    async def initialize(self, base_prompt: str) -> CacheState:
        self._base_prompt = base_prompt
        self._state = CacheState(
            confirmed_text=base_prompt,
            confirmed_token_count=len(base_prompt.split()),
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
