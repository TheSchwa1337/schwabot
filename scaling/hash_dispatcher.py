"""
Hash Dispatcher for Schwabot System
Manages safe dispatch of hash cycles with throttling
"""

import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Callable
from .throttle_manager import ThrottleManager, SystemState

class HashDispatcher:
    """Manages safe dispatch of hash cycles with throttling"""
    
    def __init__(self, 
                 min_workers: int = 1,
                 max_workers: Optional[int] = None,
                 batch_size: int = 16,
                 poll_interval: float = 0.5):
        """
        Initialize hash dispatcher
        
        Args:
            min_workers: Minimum number of worker threads
            max_workers: Maximum number of worker threads (default: CPU count)
            batch_size: Default batch size for hash operations
            poll_interval: Interval between throttle checks (seconds)
        """
        self.throttle = ThrottleManager()
        self.min_workers = min_workers
        self.max_workers = max_workers or self._get_cpu_count()
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
    def _get_cpu_count(self) -> int:
        """Get number of CPU cores"""
        import multiprocessing
        return multiprocessing.cpu_count()
        
    def _adjust_workers(self, throttle_factor: float) -> int:
        """Adjust number of workers based on throttle factor"""
        with self._lock:
            workers = max(
                self.min_workers,
                int(self.max_workers * throttle_factor)
            )
            if workers != self._executor._max_workers:
                self._executor._max_workers = workers
            return workers
            
    def _hash_chunk(self, data: bytes) -> str:
        """Hash a single chunk of data"""
        return hashlib.sha256(data).hexdigest()
        
    def dispatch_hashes(self, 
                       data_chunks: List[bytes],
                       callback: Optional[Callable[[str], None]] = None) -> List[str]:
        """
        Dispatch hash operations with throttling
        
        Args:
            data_chunks: List of data chunks to hash
            callback: Optional callback for each hash result
            
        Returns:
            List of hash results
        """
        results = []
        futures = []
        
        for chunk in data_chunks:
            # Check throttle state
            state, factor = self.throttle.update_state()
            
            if state == SystemState.CRITICAL:
                # Pause briefly in critical state
                time.sleep(self.poll_interval)
                continue
                
            # Adjust worker count
            self._adjust_workers(factor)
            
            # Submit hash job
            future = self._executor.submit(self._hash_chunk, chunk)
            futures.append(future)
            
            # Process results as they complete
            while futures:
                done, futures = futures, []
                for f in done:
                    try:
                        result = f.result()
                        results.append(result)
                        if callback:
                            callback(result)
                    except Exception as e:
                        print(f"Hash error: {e}")
                        
        return results
        
    def shutdown(self):
        """Shutdown the dispatcher"""
        self._stop_event.set()
        self._executor.shutdown(wait=True) 