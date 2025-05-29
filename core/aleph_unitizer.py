"""
AlephUnitizer: Recursive 256² block unitizer for price entropy mapping
Implements optimized hash tree generation and memory management for Schwabot
"""

import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import json
from pathlib import Path

@dataclass
class UnitizerState:
    """State container for AlephUnitizer"""
    root_hash: str
    entropy_score: float
    pattern_vector: List[int]
    tree_depth: int
    generation_time: float
    memory_usage: int
    cache_hits: int
    cache_misses: int

class AlephUnitizer:
    """
    Recursive 256² block unitizer for price entropy mapping.
    Generates and manages recursive hash trees for price data analysis.
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        max_depth: int = 2,
        min_price_change: float = 0.001,
        parallel_processing: bool = True
    ):
        """
        Initialize AlephUnitizer with configuration parameters.
        
        Args:
            cache_size: Maximum number of hash trees to cache
            max_depth: Maximum depth of recursive hash tree
            min_price_change: Minimum price change to trigger full tree generation
            parallel_processing: Whether to use parallel processing for hash generation
        """
        self.cache = {}
        self.cache_size = cache_size
        self.max_depth = max_depth
        self.min_price_change = min_price_change
        self.parallel_processing = parallel_processing
        
        self.last_price = None
        self.last_hash_tree = None
        self.state = UnitizerState(
            root_hash="",
            entropy_score=0.0,
            pattern_vector=[],
            tree_depth=0,
            generation_time=0.0,
            memory_usage=0,
            cache_hits=0,
            cache_misses=0
        )
        
        # Initialize thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4) if parallel_processing else None
        
    def __del__(self):
        """Cleanup thread pool on deletion"""
        if self.thread_pool:
            self.thread_pool.shutdown()
    
    def generate_root_hash(self, price: float, timestamp: float) -> str:
        """
        Generate initial SHA256 hash from price and timestamp.
        
        Args:
            price: Current price
            timestamp: Current timestamp
            
        Returns:
            str: SHA256 hash of price and timestamp
        """
        raw_data = f"{price}_{timestamp}"
        return hashlib.sha256(raw_data.encode()).hexdigest()
    
    def _generate_hash_branch(
        self,
        byte_val: int,
        depth: int,
        max_depth: int
    ) -> Dict:
        """
        Generate a single branch of the hash tree.
        
        Args:
            byte_val: Byte value to hash
            depth: Current depth in tree
            max_depth: Maximum depth to generate
            
        Returns:
            Dict: Hash tree branch
        """
        if depth >= max_depth:
            return {}
            
        new_hash = hashlib.sha256(bytes([byte_val])).hexdigest()
        new_bytes = bytes.fromhex(new_hash)
        
        if self.parallel_processing and depth < max_depth - 1:
            # Use thread pool for deeper levels
            futures = []
            for b in new_bytes:
                futures.append(
                    self.thread_pool.submit(
                        self._generate_hash_branch,
                        b,
                        depth + 1,
                        max_depth
                    )
                )
            children = {f.result()['key']: f.result()['value'] 
                       for f in futures}
        else:
            # Generate children recursively
            children = {
                b: self._generate_hash_branch(b, depth + 1, max_depth)
                for b in new_bytes
            }
            
        return {
            "hash": new_hash,
            "children": children
        }
    
    def _should_generate_full_tree(self, price: float) -> bool:
        """
        Determine if full 256² tree generation is needed.
        
        Args:
            price: Current price
            
        Returns:
            bool: Whether to generate full tree
        """
        if self.last_price is None:
            return True
            
        price_change = abs(price - self.last_price) / self.last_price
        return price_change > self.min_price_change
    
    def _calculate_entropy_score(self, hash_tree: Dict) -> float:
        """
        Calculate entropy score from hash tree.
        
        Args:
            hash_tree: Hash tree to analyze
            
        Returns:
            float: Entropy score
        """
        # Convert hash values to numeric form
        values = []
        def extract_values(node):
            if isinstance(node, dict):
                if "hash" in node:
                    values.append(int(node["hash"][:8], 16))
                if "children" in node:
                    for child in node["children"].values():
                        extract_values(child)
        
        extract_values(hash_tree)
        
        if not values:
            return 0.0
            
        # Calculate entropy using numpy
        values = np.array(values)
        hist, _ = np.histogram(values, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log2(hist))
    
    def _extract_pattern_vector(self, hash_tree: Dict) -> List[int]:
        """
        Extract pattern vector from hash tree.
        
        Args:
            hash_tree: Hash tree to analyze
            
        Returns:
            List[int]: Pattern vector
        """
        # Take first 8 bytes of root hash as pattern vector
        root_hash = hash_tree.get("root_hash", "")
        if not root_hash:
            return [0] * 8
            
        try:
            return [int(root_hash[i:i+2], 16) for i in range(0, 16, 2)]
        except ValueError:
            return [0] * 8
    
    def unitize_price(
        self,
        price: float,
        timestamp: Optional[float] = None
    ) -> Dict:
        """
        Generate unitizer tree for price data.
        
        Args:
            price: Current price
            timestamp: Current timestamp (defaults to current time)
            
        Returns:
            Dict: Unitizer tree with metadata
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Check cache first
        cache_key = f"{price}_{timestamp}"
        if cache_key in self.cache:
            self.state.cache_hits += 1
            return self.cache[cache_key]
            
        self.state.cache_misses += 1
        start_time = time.time()
        
        # Generate root hash
        root_hash = self.generate_root_hash(price, timestamp)
        root_bytes = bytes.fromhex(root_hash)
        
        # Determine if full tree generation is needed
        if not self._should_generate_full_tree(price):
            # Use simplified tree for small price changes
            tree_depth = 1
        else:
            tree_depth = self.max_depth
            
        # Generate hash tree
        tree = {
            b: self._generate_hash_branch(b, 0, tree_depth)
            for b in root_bytes
        }
        
        # Calculate entropy and pattern vector
        entropy_score = self._calculate_entropy_score(tree)
        pattern_vector = self._extract_pattern_vector({"root_hash": root_hash})
        
        # Create result
        result = {
            "root_hash": root_hash,
            "tree": tree,
            "entropy_score": entropy_score,
            "pattern_vector": pattern_vector,
            "generation_time": time.time() - start_time,
            "tree_depth": tree_depth
        }
        
        # Update state
        self.state = UnitizerState(
            root_hash=root_hash,
            entropy_score=entropy_score,
            pattern_vector=pattern_vector,
            tree_depth=tree_depth,
            generation_time=result["generation_time"],
            memory_usage=len(json.dumps(result)),
            cache_hits=self.state.cache_hits,
            cache_misses=self.state.cache_misses
        )
        
        # Cache result
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = result
        
        # Update last price and tree
        self.last_price = price
        self.last_hash_tree = result
        
        return result
    
    def get_state(self) -> UnitizerState:
        """
        Get current unitizer state.
        
        Returns:
            UnitizerState: Current state
        """
        return self.state
    
    def clear_cache(self):
        """Clear the hash tree cache"""
        self.cache.clear()
        
    def save_state(self, path: Union[str, Path]):
        """
        Save current state to file.
        
        Args:
            path: Path to save state
        """
        state_dict = {
            "root_hash": self.state.root_hash,
            "entropy_score": self.state.entropy_score,
            "pattern_vector": self.state.pattern_vector,
            "tree_depth": self.state.tree_depth,
            "generation_time": self.state.generation_time,
            "memory_usage": self.state.memory_usage,
            "cache_hits": self.state.cache_hits,
            "cache_misses": self.state.cache_misses
        }
        
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2)
            
    def load_state(self, path: Union[str, Path]):
        """
        Load state from file.
        
        Args:
            path: Path to load state from
        """
        with open(path, 'r') as f:
            state_dict = json.load(f)
            
        self.state = UnitizerState(**state_dict) 