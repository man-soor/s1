import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional
import gc
import time
import logging
import math

## TODOs ??
## 1. clean up process lambdas
## 2. Unify Single Request Fallback
## 3. add test suite

# TODO: should just replace with proper package imports instead
try:
    from eval.memory_optimizer import MemoryMonitor, AdaptiveBatcher, oom_safe_function
except ImportError:
    # Fallback: try to import from current directory or raise informative error
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent.parent / "eval"))
        from memory_optimizer import MemoryMonitor, AdaptiveBatcher, oom_safe_function
    except ImportError as e:
        raise ImportError(
            "Cannot import memory optimization utilities. "
            "Please ensure memory_optimizer.py is in the Python path."
        ) from e

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import clear_torch_cache

logger = logging.getLogger(__name__)

MIN_LOG_LIKELIHOOD = -1e6  # replace -inf for numerical stability
DEFAULT_FALLBACK_STRING = "[ERROR]" 

@register_model("hf-memory-optimized")
class MemoryOptimizedHFLM(HFLM):
    """
    Memory-optimized version of HF LM using adaptive batching,
    memory monitoring, and moreaggressive optimization techniques
    """
    
    def __init__(
        self,
        memory_threshold_gb: float = 12.0,
        adaptive_batching: bool = True,
        aggressive_cleanup: bool = True,
        enable_memory_monitoring: bool = True,
        **kwargs
    ):
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Memory optimization settings
        self.memory_threshold_gb = memory_threshold_gb
        self.adaptive_batching = adaptive_batching
        self.aggressive_cleanup = aggressive_cleanup
        self.enable_memory_monitoring = enable_memory_monitoring
        
        # Initialize memory optimization components
        if self.enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor(
                warning_threshold_gb=memory_threshold_gb * 0.8,
                critical_threshold_gb=memory_threshold_gb * 0.9
            )
            
        if self.adaptive_batching:
            self.batcher = AdaptiveBatcher(
                initial_batch_size=self.batch_size if isinstance(self.batch_size, int) else 16,
                memory_monitor=self.memory_monitor if self.enable_memory_monitoring else None
            )
            
        # Track statistics
        self.total_requests_processed = 0
        self.oom_recoveries = 0
        self.batch_size_adjustments = 0
        self.failed_requests = 0 
    
    def _memory_cleanup(self, force: bool = False):
        """Perform memory cleanup when needed"""
        if self.aggressive_cleanup or force:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # torch.cuda.synchronize()
            gc.collect()
            
            if self.enable_memory_monitoring:
                self.memory_monitor.log_memory_stats("After cleanup: ")
    
    def _calculate_adaptive_chunk_size(self, original_size: int, attempt: int = 0) -> int:
        """Calculate chunk size with exponential backoff"""
        # try  more aggressive reduction and then go to exponential backoff
        reduction_factor = 2 ** (attempt + 2)  # 4, 8, 16, 32, ...
        chunk_size = max(1, original_size // reduction_factor)
        return chunk_size
    
    @oom_safe_function(max_retries=3, fallback_batch_size=1)
    def _model_call(self, inps, attn_mask=None, labels=None):
        """Memory-safe model call with OOM protection"""
        try:
            # Monitor memory before call
            if self.enable_memory_monitoring and self.memory_monitor.should_trigger_cleanup():
                self._memory_cleanup()
                
            # std model call
            result = super()._model_call(inps, attn_mask, labels)
            
            # less aggressive cleanup condition
            if self.aggressive_cleanup and inps.size(0) > 32:  # Increased threshold
                self._memory_cleanup()
                
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM in model call, batch size: {inps.size(0)}")
                self.oom_recoveries += 1
                self._memory_cleanup(force=True)
            raise e
    
    def _adaptive_batch_process(self, requests, process_fn):
        """Process with adaptive batching"""
        if not self.adaptive_batching or len(requests) == 1:
            return process_fn(requests)
            
        def batch_process_fn(batch_requests):
            return process_fn(batch_requests)
            
        results = self.batcher.process_batch(requests, batch_process_fn)
        self.total_requests_processed += len(requests)
        
        return results
    
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        """MO loglikelihood computation"""
        
        if self.enable_memory_monitoring:
            self.memory_monitor.log_memory_stats("Before loglikelihood: ")
            
        if self.adaptive_batching and not override_bs:
            # Use adaptive batching for memory efficiency
            def process_batch(batch_requests):
                return super(MemoryOptimizedHFLM, self)._loglikelihood_tokens(
                    batch_requests, disable_tqdm=True, override_bs=len(batch_requests)
                )
            
            results = self._adaptive_batch_process(requests, process_batch)
            
        else:
            # go to standard processing with memory monitoring
            try:
                results = super()._loglikelihood_tokens(requests, disable_tqdm, override_bs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM in loglikelihood, retrying with smaller batches")
                    self._memory_cleanup(force=True)
                    
                    # set up exponential backoff for chunk size
                    results = []
                    attempt = 0
                    max_attempts = 3
                    
                    while attempt < max_attempts:
                        try:
                            chunk_size = self._calculate_adaptive_chunk_size(len(requests), attempt)
                            logger.info(f"Attempting with chunk size: {chunk_size} (attempt {attempt + 1})")
                            
                            for i in range(0, len(requests), chunk_size):
                                chunk = requests[i:i + chunk_size]
                                chunk_results = super()._loglikelihood_tokens(
                                    chunk, disable_tqdm=True, override_bs=len(chunk)
                                )
                                results.extend(chunk_results)
                                self._memory_cleanup()
                            break
                        except RuntimeError as retry_e:
                            if "out of memory" in str(retry_e).lower() and attempt < max_attempts - 1:
                                attempt += 1
                                logger.warning(f"Still OOM with chunk size {chunk_size}, reducing further")
                                continue
                            else:
                                # adding in proper fallback with finite ll
                                logger.error("Failed to process even with minimum chunk size")
                                results = [(MIN_LOG_LIKELIHOOD, False) for _ in requests]
                                self.failed_requests += len(requests)
                                break
                else:
                    raise e
                    
        if self.enable_memory_monitoring:
            self.memory_monitor.log_memory_stats("After loglikelihood: ")
            
        return results
    
    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """MO text generation"""
        
        if self.enable_memory_monitoring:
            self.memory_monitor.log_memory_stats("Before generation: ")
            
        if self.adaptive_batching:
            # Use adaptive batching for generation
            def process_batch(batch_requests):
                return super(MemoryOptimizedHFLM, self).generate_until(
                    batch_requests, disable_tqdm=True
                )
            
            results = self._adaptive_batch_process(requests, process_batch)
            
        else:
            # Use standard processing with memory monitoring
            try:
                results = super().generate_until(requests, disable_tqdm)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM in generation, retrying with smaller batches")
                    self._memory_cleanup(force=True)
                    
                    # Retry with individual requests
                    results = []
                    for request in requests:
                        try:
                            result = super().generate_until([request], disable_tqdm=True)
                            results.extend(result)
                        except RuntimeError as e2:
                            if "out of memory" in str(e2).lower():
                                logger.error(f"OOM even with single request, using fallback")
                                results.append(DEFAULT_FALLBACK_STRING)
                                self.failed_requests += 1
                            else:
                                raise e2
                        self._memory_cleanup()
                else:
                    raise e
                    
        if self.enable_memory_monitoring:
            self.memory_monitor.log_memory_stats("After generation: ")
            
        return results
    
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        """Memory-optimized rolling loglikelihood"""
        
        if self.enable_memory_monitoring:
            self.memory_monitor.log_memory_stats("Before rolling loglikelihood: ")
            
        # with rolling LL, need to be more careful with memory        
        results = []
        for i, request in enumerate(requests):
            try:
                # Process one at a time to avoid memory buildup
                result = super().loglikelihood_rolling([request], disable_tqdm=True)
                results.extend(result)
                
                # Periodic cleanup
                # was 10 before but frequency was too high 
                if i % 50 == 0 and self.aggressive_cleanup:
                    self._memory_cleanup()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM in rolling loglikelihood for request {i}")
                    self._memory_cleanup(force=True)
                    
                    # Try again more aggressively
                    try:
                        with torch.cuda.device(self.device):
                            torch.cuda.empty_cache()
                        result = super().loglikelihood_rolling([request], disable_tqdm=True)
                        results.extend(result)
                    except RuntimeError:
                        logger.error(f"Failed to process request {i} even after cleanup")
                        #results.append(float('-inf'))  # Fallback
                        results.append(MIN_LOG_LIKELIHOOD)
                        self.failed_requests += 1
                else:
                    raise e
                    
        if self.enable_memory_monitoring:
            self.memory_monitor.log_memory_stats("After rolling loglikelihood: ")
            
        return results
    
    def get_memory_stats(self) -> dict:
        """Get detailed memory and performance statistics"""
        stats = {
            "total_requests_processed": self.total_requests_processed,
            "oom_recoveries": self.oom_recoveries,
            "batch_size_adjustments": self.batch_size_adjustments,
            "failed_requests": self.failed_requests,  # add failure tracking
        }
        
        if self.enable_memory_monitoring:
            stats.update({
                "peak_memory_gb": self.memory_monitor.peak_memory_usage,
                "current_gpu_memory": self.memory_monitor.get_gpu_memory_info(),
                "current_cpu_memory": self.memory_monitor.get_cpu_memory_info(),
            })
            
        if self.adaptive_batching:
            stats.update(self.batcher.get_stats())
            
        return stats
    
    # def _retry_as_single(self, requests, fn):
    # results = []
    # for req in requests:
    #     try:
    #         results.extend(fn([req]))
    #     except RuntimeError as e:
    #         if "out of memory" in str(e).lower():
    #             logger.warning("Skipping request due to repeated OOM")
    #             results.append("")
    #         else:
    #             raise
    # return results
    
    def cleanup_resources(self):
        """Comprehensive resource cleanup - revised for safer method calls"""
        logger.info("Performing comprehensive resource cleanup...")
        
        try:
            if hasattr(self.model, 'clear_cache') and callable(getattr(self.model, 'clear_cache')):
                self.model.clear_cache()
        except Exception as e:
            logger.warning(f"Could not clear model cache: {e}")
            
        try:
            if hasattr(self.tokenizer, 'clear_cache') and callable(getattr(self.tokenizer, 'clear_cache')):
                self.tokenizer.clear_cache()
        except Exception as e:
            logger.warning(f"Could not clear tokenizer cache: {e}")
            
        # Force memory cleanup
        self._memory_cleanup(force=True)
        
        # Log final stats
        if self.enable_memory_monitoring:
            self.memory_monitor.log_memory_stats("Final cleanup: ")
            
        final_stats = self.get_memory_stats()
        logger.info(f"Final memory optimization stats: {final_stats}")

#### example confiog
MEMORY_OPTIMIZED_CONFIG = {
    "model_args": {
        "memory_threshold_gb": 16.0,
        "adaptive_batching": True,
        "aggressive_cleanup": True,
        "enable_memory_monitoring": True,
        "batch_size": "auto",
        "max_batch_size": 32,
    }
}

def create_memory_optimized_model(model_name: str, **kwargs):
    """Factory function to create memory-optimized model"""
    config = MEMORY_OPTIMIZED_CONFIG["model_args"].copy()
    config.update(kwargs)
    
    return MemoryOptimizedHFLM(
        pretrained=model_name,
        **config
    ) 