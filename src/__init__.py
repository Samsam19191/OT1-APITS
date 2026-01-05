import torch

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LOAD_IN_4BIT = True
MAX_SEQ_LEN_TYPING = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Export keystroke streaming modules
from .events import EventType, KeystrokeEvent, StreamingSession
from .keystroke_simulator import KeystrokeSimulator, EventConsumer, LoggingConsumer
