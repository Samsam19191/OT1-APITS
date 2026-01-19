"""
StreamController and KVCacheManager interfaces for anticipatory prefill.
"""

from abc import ABC
from dataclasses import dataclass
from hashlib import sha256
import os
from typing import AsyncIterator, Optional, Any
from enum import Enum
import asyncio

import torch
from transformers.cache_utils import DynamicCache

from .utils import find_last_boundary, CONFIRMATION_BOUNDARIES

@dataclass
class CacheState:
    past_key_values: Any = None
    confirmed_text: str = ""
    confirmed_token_count: int = 0

    def save(self, filepath: str):
        """
        Serializes the cache state to disk.
        """
        # We save the object as a dictionary to be safe against class definition changes
        state_dict = {
            "past_key_values": self.past_key_values,
            "confirmed_text": self.confirmed_text,
            "confirmed_token_count": self.confirmed_token_count
        }
        torch.save(state_dict, filepath)
        print(f"[CacheState] Saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: str = "cpu") -> 'CacheState':
        """
        Loads the cache state from disk.
        Args:
            filepath: Path to the .pt file
            device: 'cpu' or 'cuda' (maps tensors to this device)
        """
        try:
            data = torch.load(filepath, map_location=device, weights_only=False)
            
            return cls(
                past_key_values=data["past_key_values"],
                confirmed_text=data["confirmed_text"],
                confirmed_token_count=data["confirmed_token_count"]
            )
        except FileNotFoundError:
            print(f"[CacheState] File not found: {filepath}")
            return cls() # Return empty state
        except Exception as e:
            print(f"[CacheState] Error loading cache: {e}")
            return cls()

class KVCacheManager(ABC):
    """
    Abstract interface for KV-cache management.
    
    Rémi will implement this with actual HF Transformers cache operations.
    """

    def __init__(self, model, tokenizer, device="cuda", verbose: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.verbose = verbose
        self._state = CacheState()
        self.is_first_session = True
    
    async def initialize(self, base_prompt: str) -> CacheState:
        """
        Initialize cache with base prompt (system prompt + schema).
        Resets any existing state effectively starting a fresh session.
        """

        if self.is_first_session:
            if self.verbose:
                print(f"\n[KV] Initializing with prompt: '{base_prompt}'")
            self._state = CacheState()

            if not base_prompt:
                return self._state

            inputs = self.tokenizer(base_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=True)

            self._state.past_key_values = outputs.past_key_values
            self._state.confirmed_text = base_prompt
            self._state.confirmed_token_count = inputs.input_ids.shape[1]

            filename = sha256(base_prompt.encode("utf-8")).hexdigest()
            os.makedirs("./cache", exist_ok=True)
            self._state.save("./cache/" + filename + ".pt")
            self.is_first_session = False

            if self.verbose:
                print(f"[KV] Init complete. Cache size: {self._state.confirmed_token_count} tokens.")
        else: 
            if self.verbose:
                print(f"[KV] CacheState already exists.Skipping initialization.")

            filename = sha256(base_prompt.encode("utf-8")).hexdigest()
            self._state = CacheState.load("./cache/" + filename + ".pt", device=self.device)

        return self._state
    
    async def extend(self, new_text: str) -> CacheState:
        """
        Extend the KV-cache with new confirmed text.
        Follows the logic of: ids_p2 = ids_full[:, len_p1:]
        """
        if self.verbose:
            print(f"[KV] Extending cache with delta: '{new_text}'")
        full_text_next = self._state.confirmed_text + new_text

        tokens_full = self.tokenizer(full_text_next, return_tensors="pt").to(self.device)
        input_ids_full = tokens_full.input_ids
        current_len = input_ids_full.shape[1]
        past_len = self._state.confirmed_token_count

        # If we added text, but the token count didn't increase (or shrank),
        # it means the tokenizer merged the previous last token with the new text.
        if current_len <= past_len and len(new_text) > 0:
            if self.verbose:
                print(f"[KV] Merge Detected (Old len: {past_len}, New len: {current_len}). Rolling back 1 token.")
            
            # Decrement our "safe" boundary by 1
            past_len = max(0, past_len - 1)
            
            # Crop the actual GPU cache by 1 to remove the unstable token
            # We assume the prefix up to N-1 is stable.
            if self._state.past_key_values is not None:
                kv = self._state.past_key_values
                if hasattr(kv, 'crop'):
                    # DynamicCache
                    kv.crop(past_len)
                else:
                    # Legacy Tuple
                    cropped = []
                    for k, v in kv:
                        cropped.append((k[..., :past_len, :], v[..., :past_len, :]))
                    self._state.past_key_values = tuple(cropped)

        input_ids_new = input_ids_full[:, past_len:]
        
        if self.verbose:
            print(f"[KV] Total: {current_len}. Cached: {past_len}. Computing: {input_ids_new.shape[1]}")

        if input_ids_new.shape[1] == 0:
            if self.verbose:
                print("[KV] No new tokens to compute. Updating text reference only.")
            self._state.confirmed_text = full_text_next
            return self._state

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_new,
                past_key_values=self._state.past_key_values,
                use_cache=True
            )

        self._state.past_key_values = outputs.past_key_values
        self._state.confirmed_text = full_text_next
        self._state.confirmed_token_count = input_ids_full.shape[1]
        
        if self.verbose:
            print(f"[KV] Extend success. New cache size: {self._state.confirmed_token_count}")
        return self._state
        
    async def crop(self, target_text: str) -> CacheState:
        """
        Crops the KV-cache to match the length of remaining_text.
        Called when the user deletes or edits previously confirmed text.
        """
        print(f"[KV] Cropping requested. Target text: '{target_text}'")
        
        if not target_text:
            print("[KV] Target text empty. Resetting cache.")
            return await self.initialize("")

        tokens = self.tokenizer(target_text, return_tensors="pt").to(self.device)
        target_length = tokens.input_ids.shape[1]
        
        current_len = self._state.confirmed_token_count
        print(f"[KV] Target length: {target_length} tokens. Current cache: {current_len} tokens.")

        current_kv = self._state.past_key_values

        if current_len <= target_length:
            print("[KV] Current cache is smaller/equal to target. No crop needed.")
            return self._state

        # Perform the Crop        
        if isinstance(current_kv, DynamicCache):
            try:
                current_kv.crop(target_length)
            except Exception as e:
                print(f"[KV] ERROR: DynamicCache crop failed: {e}")
                return await self.initialize(target_text)
                
        else:
            try:
                cropped_layers = []
                for k, v in current_kv:
                    k_cropped = k[..., :target_length, :]
                    v_cropped = v[..., :target_length, :]
                    cropped_layers.append((k_cropped, v_cropped))
                
                self._state.past_key_values = tuple(cropped_layers)
            except Exception as e:
                print(f"[KV] ERROR: Legacy crop failed: {e}")
                return await self.initialize(target_text)

        self._state.confirmed_text = target_text
        self._state.confirmed_token_count = target_length
        
        if self.verbose:
            print(f"[KV] Crop success. New cache size: {target_length}")
        return self._state
    
    async def generate(
        self, 
        max_new_tokens: int = 64, 
        temperature: float = 0.0  # Greedy by default
    ) -> AsyncIterator[str]:
        """
        Stream generated text based on the current confirmed cache.
        
        Mechanism:
        1. Backstep: Temporarily rewinds the cache by 1 token.
        2. Recover: Re-runs the last confirmed token to get the logits (predictions).
        3. Loop: Generates new tokens autoregressively.
        """
        text = self._state.confirmed_text
        if self.verbose:
            print(f"\n[KV] Starting GENERATION. Context: '{text}'")
        
        if not text:
            return 
            
        all_tokens = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        current_len = all_tokens.shape[1]
        
        if current_len == 0:
            yield

        # Backstep Logic
        if self.verbose:
            print("[KV] Performing Backstep (Rewind 1 token)...")
        last_token_input = all_tokens[:, -1:] 
        kv_cache = self._state.past_key_values
        
        if kv_cache is not None:
            if isinstance(kv_cache, DynamicCache):
                kv_cache.crop(current_len - 1)
            else:
                cropped = []
                for k, v in kv_cache:
                    cropped.append((k[..., :current_len-1, :], v[..., :current_len-1, :]))
                self._state.past_key_values = tuple(cropped)

        input_ids = last_token_input
        generated_text_accum = ""

        for i in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=self._state.past_key_values,
                    use_cache=True
                )
            
            self._state.past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                if self.verbose:
                    print("[KV] EOS token detected.")
                break
                
            token_str = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            
            # Print first few tokens to verify it works
            if self.verbose:
                print(f"[KV] Gen token {i}: '{token_str}'")
            
            yield token_str

            await asyncio.sleep(0) # To give time for the loop to send token

            generated_text_accum += token_str
            input_ids = next_token_id

        self._state.confirmed_text += generated_text_accum
        self._state.confirmed_token_count += len(self.tokenizer.encode(generated_text_accum))
        if self.verbose:
            print(f"[KV] Generation Complete. Generated '{generated_text_accum}'")

class StreamController:
    def __init__(
        self,
        cache_manager,  # KVCacheManager
        debounce_ms: float = 300.0,
        force_flush_confirmation: int = 2  # Trigger immediately after 2 spaces
    ):
        self.cache_manager = cache_manager
        self.debounce_ms = debounce_ms
        self.force_flush_confirmation = force_flush_confirmation
        
        # Text State
        self.current_text: str = ""
        self.confirmed_text: str = ""
        self.confirmation_counter: int = 0
        
        # Async State
        self._debounce_timer: Optional[asyncio.Task] = None
        self._active_extension_task: Optional[asyncio.Task] = None
        self._is_extending = False  # Lock to prevent parallel cache extensions
        self._flush_callback = None

    async def start_session(self, base_prompt: str = ""):
        """
        Resets the controller for a new user session.
        
        1. Cancels any pending debounce/GPU tasks from the previous session.
        2. Resets the local text buffers (User View).
        3. Initializes the KV Cache with the system prompt (Global View).
        """
        print(f"[Controller] Starting Session. Prompt len: {len(base_prompt)}")
        # Cleanup: Cancel any lingering background tasks
        # If the user refreshed the page while the GPU was working, we must stop it.
        if self._debounce_timer and not self._debounce_timer.done():
            self._debounce_timer.cancel()
            
        if self._active_extension_task and not self._active_extension_task.done():
            self._active_extension_task.cancel()

        # Reset async state flags
        self._debounce_timer = None
        self._active_extension_task = None
        self._is_extending = False
        
        # Reset Text State
        # The controller tracks only the *User's* text. 
        # The system prompt is "hidden" in the cache manager.
        self.current_text = ""
        self.confirmed_text = ""
        self.confirmation_counter = 0

        # Initialize GPU Cache (The Prefill)
        # This sends the system prompt to the GPU so it's ready before the user types 'A'.
        await self.cache_manager.initialize(base_prompt)

    async def on_text_update(self, text: str) -> dict:
        """
        Handle a keystroke. 
        - Updates text immediately.
        - Resets the 'idling' timer.
        - Checks if we should force a flush (e.g., user finished a sentence).
        """
        # Check for modifications (Backspacing/Editing)
        # Note: 'text' here represents the FULL text state from the frontend
        previous_text = text[:len(self.current_text)]
        for i, c in enumerate(reversed(self.current_text)):
            if c != previous_text[len(previous_text) - i - 1]:
                print(f"[Controller] DETECTED EDIT/BACKSPACE.")
                print(f"   Old: '{self.current_text}'")
                print(f"   New: '{previous_text}'")
                
                self.current_text = self.current_text[:len(self.current_text) - i - 1]
                
                print(f"[Controller] Calling CROP to: '{self.current_text}'")
                await self.cache_manager.crop(self.current_text)
                
                # Update pointer so we don't think we have confirmed text that doesn't exist
                if len(self.confirmed_text) > len(self.current_text):
                     self.confirmed_text = self.current_text
                     
                return {
                    "event": "keystroke",
                    "current_text": self.current_text,
                    "confirmed_text": self.confirmed_text,
                    "pending_processing": self.current_text[len(self.confirmed_text):]
                }

        # Normal Typing (Append)
        if len(text) > len(self.current_text):
             char = text[len(self.current_text)]
             self.current_text += char
             # print(f"[Controller] Key: '{char}' | Buffer: '{self.current_text}'")
        else:
             # Fallback if text didn't grow (duplicate event?)
             pass
        
        # 1. Check for immediate trigger
        if char in CONFIRMATION_BOUNDARIES:
            self.confirmation_counter += 1
            
        if self.confirmation_counter >= self.force_flush_confirmation:
            print("[Controller] Force Flush triggered (Whitespace/Punctuation).")
            self._trigger_background_extension()
            if self._debounce_timer: self._debounce_timer.cancel()
            self.confirmation_counter = 0 
        else:
            self._reset_debounce_timer()

        return {
            "event": "text_update",
            "current_text": self.current_text,
            "confirmed_text": self.confirmed_text,
            "pending_processing": self.current_text[len(self.confirmed_text):]
        }


    def _reset_debounce_timer(self):
        """Cancels the WAITING timer, but leaves the RUNNING extension alone."""
        if self._debounce_timer and not self._debounce_timer.done():
            self._debounce_timer.cancel()
        
        self._debounce_timer = asyncio.create_task(self._wait_and_trigger())

    async def _wait_and_trigger(self):
        """The 'Idling' logic: wait X ms, then try to extend."""
        try:
            await asyncio.sleep(self.debounce_ms / 1000.0)
            self._trigger_background_extension()
        except asyncio.CancelledError:
            # This is normal (user typed again before timer expired)
            pass

    def _trigger_background_extension(self):
        """
        Fire-and-forget method. 
        If the GPU is busy, we skip this trigger (the next keystroke/timer will catch it).
        """
        if self._is_extending:
            print("[Controller] GPU busy. Skipping trigger.")
            # GPU is busy. We do NOT queue a new one here to avoid complex state.
            # The currently running task will finish, and the user's next action 
            # (or the next debounce) will trigger the update for the remaining text.
            return

        # Start the heavy lifting in the background
        self._active_extension_task = asyncio.create_task(self._run_extension_logic())

    async def _run_extension_logic(self):
        """
        The actual GPU operation. 
        This is PROTECTED: user typing will NOT cancel this.
        """
        self._is_extending = True
        try:
            boundary_pos = find_last_boundary(self.current_text)
            
            if boundary_pos <= len(self.confirmed_text):
                # print("[Controller] No new complete words to confirm.")
                return

            new_confirmed_snapshot = self.current_text[:boundary_pos]
            delta = new_confirmed_snapshot[len(self.confirmed_text):]
            
            if delta:
                print(f"[Controller] Extension Task: Processing delta '{delta}'")
                await self.cache_manager.extend(delta)
                
                self.confirmed_text = new_confirmed_snapshot
                
                if self._flush_callback:
                    await self._flush_callback({
                        "event": "flush",
                        "confirmed_text": self.confirmed_text,
                        "cache_extended": True
                    })
                
        except Exception as e:
            print(f"[Controller] Extension failed: {e}")
        finally:
            self._is_extending = False

    def set_flush_callback(self, callback):
        """
        Register an async function to be called when a background flush completes.
        Signature: async def callback(data: dict) -> None
        """
        self._flush_callback = callback

    async def on_submit(self):
        """
        Handle submit event (user pressed Enter).
        
        1. Cancels background tasks.
        2. Flushes pending text to GPU.
        3. Streams generated tokens.
        """
        print(f"\n[Controller] SUBMIT received. Final text: '{self.current_text}'")

        if self._debounce_timer: 
            self._debounce_timer.cancel()
            self._debounce_timer = None

        if self._active_extension_task and not self._active_extension_task.done():
            print("[Controller] Waiting for active extension to finish...")
            try:
                await self._active_extension_task
            except asyncio.CancelledError:
                pass
        self._active_extension_task = None

        remaining = self.current_text[len(self.confirmed_text):]
        
        if remaining:
            print(f"[Controller] Final flush for remaining text: '{remaining}'")
            await self.cache_manager.extend(remaining)
            self.confirmed_text = self.current_text

        yield {
            "event": "submit_start"
        }
        
        start_time = asyncio.get_event_loop().time()
        generated_so_far = ""
        first_token = True

        async for token in self.cache_manager.generate(max_new_tokens=100):
            if first_token:
                ttft_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                yield {
                    "event": "generation_start",
                    "time_to_first_token_ms": ttft_ms
                }
                first_token = False
            generated_so_far += token
            # Yield tokens? (Depends on how you want to pipe it, usually yes)
            yield {
                "event": "generation_token",
                "token": token,
            }

        total_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        yield {
            "event": "generation_complete",
            "generated_text": generated_so_far,
            "total_time_ms": total_time
        }
