"""
Evaluation Metrics for SQL correctness.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple
from datetime import datetime


@dataclass
class EvalResult:
    """Result of evaluating a single SQL generation."""
    question_id: str
    question: str
    gold_query: str
    predicted_query: str
    
    syntax_valid: bool
    syntax_error: Optional[str] = None
    
    result_match: bool = False
    gold_row_count: int = 0
    predicted_row_count: int = 0
    
    generation_time_ms: Optional[float] = None
    ttft_ms: Optional[float] = None
    
    def is_correct(self) -> bool:
        return self.syntax_valid and self.result_match


@dataclass
class Metrics:
    """Aggregated evaluation metrics."""
    total: int = 0
    syntax_valid: int = 0
    result_match: int = 0
    
    model_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    avg_inference_time_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    
    @property
    def execution_accuracy(self) -> float:
        """% of queries that execute without error."""
        return (self.syntax_valid / self.total * 100) if self.total > 0 else 0
    
    @property
    def exact_match_accuracy(self) -> float:
        """% of queries that return correct results."""
        return (self.result_match / self.total * 100) if self.total > 0 else 0


class MetricsCollector:
    """Collects evaluation results and computes metrics."""
    
    def __init__(self, model_name: str = ""):
        self.results: List[EvalResult] = []
        self.model_name = model_name
    
    def add(self, result: EvalResult):
        self.results.append(result)
    
    def compute(self) -> Metrics:
        metrics = Metrics(
            model_name=self.model_name,
            total=len(self.results)
        )
        
        total_time = 0.0
        count_time = 0
        
        total_ttft = 0.0
        count_ttft = 0
        
        for r in self.results:
            if r.syntax_valid:
                metrics.syntax_valid += 1
            if r.result_match:
                metrics.result_match += 1
            
            if r.generation_time_ms is not None:
                total_time += r.generation_time_ms
                count_time += 1
            
            if r.ttft_ms is not None:
                total_ttft += r.ttft_ms
                count_ttft += 1
        
        if count_time > 0:
            metrics.avg_inference_time_ms = total_time / count_time
        
        if count_ttft > 0:
            metrics.avg_ttft_ms = total_ttft / count_ttft
        
        return metrics
    
    def get_failures(self) -> List[EvalResult]:
        return [r for r in self.results if not r.is_correct()]
    
    def get_successes(self) -> List[EvalResult]:
        return [r for r in self.results if r.is_correct()]
