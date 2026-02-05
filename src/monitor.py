import asyncio
import time
import psutil
import torch
from threading import Thread, Event
from typing import List, Dict, Any

class ResourceMonitor:
    """
    Monitors system resources (CPU, RAM, GPU Memory) in a background thread.
    Designed to provide real-time data for the E2E dashboard graphs.
    """
    
    def __init__(self, interval_ms: int = 50):
        self.interval_sec = interval_ms / 1000.0
        self.running = False
        self._stop_event = Event()
        self._thread = None
        self.metrics: List[Dict[str, Any]] = []
        self.start_time = 0.0
        
        # Determine device for GPU monitoring
        self.device_type = "cpu"
        if torch.backends.mps.is_available():
            self.device_type = "mps"
        elif torch.cuda.is_available():
            self.device_type = "cuda"
            
    def start(self):
        """Start monitoring in a background thread."""
        if self.running:
            return
            
        self.running = True
        self._stop_event.clear()
        self.metrics = []
        self.start_time = time.time()
        
        self._thread = Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
    def stop(self) -> List[Dict[str, Any]]:
        """Stop monitoring and return the collected metrics."""
        self.running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.metrics
        
    def snapshot(self) -> Dict[str, Any]:
        """Take a single snapshot of current resources using robust system APIs."""
        import subprocess
        import plistlib

        stats = {
            "timestamp": time.time(),
            "relative_time_ms": (time.time() - self.start_time) * 1000 if self.start_time else 0,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "gpu_mb": 0.0,
            "gpu_load": 0.0,
            "gpu_vram_driver_mb": 0.0
        }
        
        # GPU Metrics via ioreg (Sudo-less & Robust)
        if self.device_type == "mps":
            try:
                # Use -a for XML (plist) output, -c IOAccelerator is more universal than -n
                res = subprocess.run(
                    ["ioreg", "-a", "-c", "IOAccelerator", "-r", "-d", "1"], 
                    capture_output=True, 
                    timeout=1.0
                )
                if res.returncode == 0:
                    data = plistlib.loads(res.stdout)
                    
                    # Recursive function to find PerformanceStatistics in nested dicts/lists
                    def find_perf_stats(obj):
                        if isinstance(obj, dict):
                            if "PerformanceStatistics" in obj:
                                return obj["PerformanceStatistics"]
                            for v in obj.values():
                                found = find_perf_stats(v)
                                if found: return found
                        elif isinstance(obj, list):
                            for item in obj:
                                found = find_perf_stats(item)
                                if found: return found
                        return None

                    perf = find_perf_stats(data)
                    if perf:
                        # Device Utilization % is the real "GPU Load"
                        stats["gpu_load"] = float(perf.get("Device Utilization %", stats["gpu_load"]))
                        
                        vram_bytes = perf.get("In use system memory", 0)
                        if vram_bytes > 0:
                            stats["gpu_vram_driver_mb"] = vram_bytes / (1024 * 1024)
                            
                        # Fallback for gpu_mb if needed, but torch.mps gives the allocation accurately
                        stats["gpu_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
            except Exception:
                pass
                
        return stats

    def _monitor_loop(self):
        """Internal loop running in thread."""
        while not self._stop_event.is_set():
            stats = self.snapshot()
            self.metrics.append(stats)
            time.sleep(self.interval_sec)
