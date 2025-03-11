import time
from typing import Dict, Any
import psutil
import torch

class PerformanceTracker:
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "gpu_usage": [],
            "memory_usage": [],
            "accuracy_scores": []
        }
    
    def track_response_time(self, start_time: float) -> float:
        """Track response generation time"""
        response_time = time.time() - start_time
        self.metrics["response_times"].append(response_time)
        return response_time

    def track_resource_usage(self):
        """Track system resource usage"""
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.memory_reserved()
            self.metrics["gpu_usage"].append(gpu_usage)
        
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.metrics["memory_usage"].append(memory_usage)

    def track_accuracy(self, predicted: str, expected: str) -> float:
        """Track response accuracy"""
        # Add your accuracy calculation logic here
        pass 