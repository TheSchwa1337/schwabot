from utils.safe_print import safe_print, info, warn, error, success, debug
from core.unified_math_system import unified_math
#!/usr/bin/env python3
"""
Cache Store - Mathematical Cache Optimization and Memory Management
================================================================

This module implements a comprehensive caching system for Schwabot,
providing mathematical cache optimization, LRU eviction, and advanced
memory management capabilities.

Core Mathematical Functions:
- Cache Hit Ratio: H = hits / (hits + misses)
- LRU Eviction Score: S = access_time + (size_factor \\u00d7 item_size)
- Memory Optimization: M_opt = \\u03a3(w\\u1d62 \\u00d7 v\\u1d62) where w\\u1d62 are access weights

Core Functionality:
- Multi-level caching with mathematical optimization
- LRU/LFU eviction algorithms
- Memory usage optimization and monitoring
- Cache hit ratio analysis and prediction
- Distributed caching support
- Cache invalidation and consistency management
"""

import logging
import json
import time
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from core.unified_math_system import unified_math
from collections import defaultdict, deque, OrderedDict
import os
import pickle
import threading

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    L1 = "l1"  # Fastest, smallest
    L2 = "l2"  # Medium speed, medium size
    L3 = "l3"  # Slowest, largest


class EvictionPolicy(Enum):
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    RANDOM = "random"  # Random eviction


@dataclass
class CacheItem:
    item_id: str
    key: str
    value: Any
    size: int
    access_count: int
    last_access: datetime
    created_at: datetime
    ttl: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMetrics:
    metrics_id: str
    cache_level: CacheLevel
    hit_count: int
    miss_count: int
    eviction_count: int
    memory_usage: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheOptimization:
    optimization_id: str
    cache_level: CacheLevel
    optimization_type: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvement_score: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheStore:
    pass


def __init__(self, config_path: str = "./config/cache_config.json"):
    self.config_path = config_path
    self.cache_levels: Dict[CacheLevel, Dict[str, CacheItem] = {}
    self.cache_metrics: Dict[str, CacheMetrics] = {}
    self.optimizations: Dict[str, CacheOptimization] = {}
    self.access_history: deque = deque(maxlen=10000)
    self.eviction_history: deque = deque(maxlen=5000)
    self._load_configuration()
    self._initialize_cache()
    self._start_cache_monitoring()
    logger.info("Cache Store initialized")

def _load_configuration(self) -> None:
    """Load cache configuration."""
    try:
    pass
    if os.path.exists(self.config_path):
    with open(self.config_path, 'r') as f:
    config = json.load(f)

    logger.info(f"Loaded cache configuration")
    else:
    self._create_default_configuration()

    except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    self._create_default_configuration()

def _create_default_configuration(self) -> None:
    """Create default cache configuration."""
    config = {
    "cache_levels": {
    "l1": {
    "max_size": 1000,
    "max_memory_mb": 100,
    "eviction_policy": "lru",
    "ttl_seconds": 300
    },
    "l2": {
    "max_size": 10000,
    "max_memory_mb": 500,
    "eviction_policy": "lfu",
    "ttl_seconds": 3600
    },
    "l3": {
    "max_size": 100000,
    "max_memory_mb": 2000,
    "eviction_policy": "lru",
    "ttl_seconds": 86400
    }
    },
    "optimization": {
    "auto_optimize": True,
    "optimization_interval": 300,
    "hit_ratio_threshold": 0.8
    }
    }

    try:
    pass
    os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
    with open(self.config_path, 'w') as f:
    json.dump(config, f, indent=2)
    except Exception as e:
    logger.error(f"Error saving configuration: {e}")

def _initialize_cache(self) -> None:
    """Initialize cache levels."""
    # Initialize each cache level
    for level in CacheLevel:
    self.cache_levels[level] = {}

    # Initialize cache configurations
    self.cache_configs = {
    CacheLevel.L1: {"max_size": 1000, "max_memory": 100 * 1024 * 1024, "eviction_policy": EvictionPolicy.LRU},
    CacheLevel.L2: {"max_size": 10000, "max_memory": 500 * 1024 * 1024, "eviction_policy": EvictionPolicy.LFU},
    CacheLevel.L3: {"max_size": 100000, "max_memory": 2000 * 1024 * 1024, "eviction_policy": EvictionPolicy.LRU}
    }

    logger.info(f"Initialized {len(self.cache_levels)} cache levels")

def _start_cache_monitoring(self) -> None:
    """Start cache monitoring system."""
    # This would start background monitoring tasks
    logger.info("Cache monitoring started")

def set(self, key: str, value: Any, ttl: Optional[float)=None,
    level: CacheLevel=CacheLevel.L1) -> bool:
    """Set a value in the cache."""
    try:
    pass
    # Generate item ID
    item_id = hashlib.md5(f"{key}_{level.value}".encode()).hexdigest()

    # Calculate item size
    item_size = self._calculate_item_size(value)

    # Check if cache level is full
    if self._is_cache_full(level, item_size):
    self._evict_items(level, item_size)

    # Create cache item
    cache_item = CacheItem(
    item_id=item_id,
    key=key,
    value=value,
    size=item_size,
    access_count=1,
    last_access=datetime.now(),
    created_at=datetime.now(],
    ttl=ttl,
    metadata={"level": level.value}
    ]

    # Store in cache
    self.cache_levels[level][key] = cache_item

    # Record access
    self._record_access(key, level, "set")

    logger.debug(f"Set cache item {key} in {level.value}")
    return True

    except Exception as e:
    logger.error(f"Error setting cache item: {e}")
    return False

def get(self, key: str, level: CacheLevel=CacheLevel.L1) -> Optional[Any]:
    """Get a value from the cache."""
    try:
    pass
    # Try to get from specified level
    if level in self.cache_levels and key in self.cache_levels[level]:
    item = self.cache_levels[level][key]

    # Check TTL
    if item.ttl and (datetime.now() - item.created_at).total_seconds() > item.ttl:
    self.delete(key, level)
    self._record_access(key, level, "miss_expired")
    return None

    # Update access statistics
    item.access_count += 1
    item.last_access = datetime.now()

    # Record hit
    self._record_access(key, level, "hit")

    logger.debug(f"Cache hit for {key} in ({level.value}")
    return item.value

    # Try other levels for {level.value}")
    return item.value

    # Try other levels in (({level.value}")
    return item.value

    # Try other levels for ({level.value}")
    return item.value

    # Try other levels in ((({level.value}")
    return item.value

    # Try other levels for (({level.value}")
    return item.value

    # Try other levels in (((({level.value}")
    return item.value

    # Try other levels for ((({level.value}")
    return item.value

    # Try other levels in ((((({level.value}")
    return item.value

    # Try other levels for (((({level.value}")
    return item.value

    # Try other levels in (((((({level.value}")
    return item.value

    # Try other levels for ((((({level.value}")
    return item.value

    # Try other levels in (((((({level.value}")
    return item.value

    # Try other levels if not found
    for other_level in CacheLevel)))))))))))):
    if other_level != level and other_level in self.cache_levels:
    if key in self.cache_levels[other_level]:
    item = self.cache_levels[other_level][key]

    # Check TTL
    if item.ttl and (datetime.now() - item.created_at).total_seconds() > item.ttl:
    self.delete(key, other_level)
    continue

    # Move to requested level
    self.set(key, item.value, item.ttl, level)
    self.delete(key, other_level)

    logger.debug(f"Cache hit for {key} in {other_level.value}, moved to {level.value}")
    return item.value

    # Record miss
    self._record_access(key, level, "miss")
    logger.debug(f"Cache miss for {key} in {level.value}")
    return None

    except Exception as e:
    logger.error(f"Error getting cache item: {e}")
    return None

def delete(self, key: str, level: CacheLevel=CacheLevel.L1) -> bool:
    """Delete a value from the cache."""
    try:
    pass
    if level in self.cache_levels and key in self.cache_levels[level]:
    item = self.cache_levels[level][key]

    # Record eviction
    self._record_eviction(key, level, item.size)

    # Remove from cache
    del self.cache_levels[level][key]

    logger.debug(f"Deleted cache item {key} from {level.value}")
    return True

    return False

    except Exception as e:
    logger.error(f"Error deleting cache item: {e}")
    return False

def _calculate_item_size(self, value: Any) -> int:
    """Calculate the size of a cache item."""
    try:
    pass
    # Serialize the value to estimate size
    serialized = pickle.dumps(value)
    return len(serialized)
    except Exception as e:
    logger.error(f"Error calculating item size: {e}")
    return 1024  # Default size

def _is_cache_full(self, level: CacheLevel, item_size: int) -> bool:
    """Check if cache level is full."""
    try:
    pass
    config = self.cache_configs[level]
    current_size = len(self.cache_levels[level])
    current_memory = sum(item.size for item in self.cache_levels[level).values(]]

    return (current_size >= config["max_size"] or
    (current_memory + item_size] > config["max_memory"])

    except Exception as e:
    logger.error(f"Error checking cache fullness: {e}")
    return False

def _evict_items(self, level: CacheLevel, required_size: int) -> None:
    """Evict items from cache level."""
    try:
    pass
    config = self.cache_configs[level]
    cache = self.cache_levels[level]

    if config["eviction_policy"] == EvictionPolicy.LRU:
    self._evict_lru(level, required_size)
    elif config["eviction_policy"] == EvictionPolicy.LFU:
    self._evict_lfu(level, required_size)
    elif config["eviction_policy"] == EvictionPolicy.FIFO:
    self._evict_fifo(level, required_size)
    elif config["eviction_policy"] == EvictionPolicy.RANDOM:
    self._evict_random(level, required_size)

    except Exception as e:
    logger.error(f"Error evicting items: {e}")

def _evict_lru(self, level: CacheLevel, required_size: int) -> None:
    """Evict items using LRU policy."""
    try:
    pass
    cache = self.cache_levels[level]

    # Sort items by last access time
    sorted_items = sorted(cache.items(), key=lambda x: x[1].last_access)

    freed_size = 0
    for key, item in sorted_items:
    if freed_size >= required_size:
    break

    self.delete(key, level)
    freed_size += item.size

    except Exception as e:
    logger.error(f"Error in LRU eviction: {e}")

def _evict_lfu(self, level: CacheLevel, required_size: int) -> None:
    """Evict items using LFU policy."""
    try:
    pass
    cache = self.cache_levels[level]

    # Sort items by access count
    sorted_items = sorted(cache.items(), key=lambda x: x[1].access_count)

    freed_size = 0
    for key, item in sorted_items:
    if freed_size >= required_size:
    break

    self.delete(key, level)
    freed_size += item.size

    except Exception as e:
    logger.error(f"Error in LFU eviction: {e}")

def _evict_fifo(self, level: CacheLevel, required_size: int) -> None:
    """Evict items using FIFO policy."""
    try:
    pass
    cache = self.cache_levels[level]

    # Sort items by creation time
    sorted_items = sorted(cache.items(), key=lambda x: x[1].created_at)

    freed_size = 0
    for key, item in sorted_items:
    if freed_size >= required_size:
    break

    self.delete(key, level)
    freed_size += item.size

    except Exception as e:
    logger.error(f"Error in FIFO eviction: {e}")

def _evict_random(self, level: CacheLevel, required_size: int) -> None:
    """Evict items using random policy."""
    try:
    pass
    cache = self.cache_levels[level]
    items = list(cache.items())

    freed_size = 0
    while freed_size < required_size and items:
    # Select random item
import random
key, item = random.choice(items)
items.remove((key, item))

self.delete(key, level)
freed_size += item.size

except Exception as e:
    logger.error(f"Error in random eviction: {e}")

def _record_access(self, key: str, level: CacheLevel, access_type: str) -> None:
    """Record cache access."""
    try:
    pass
    self.access_history.append({
    "key": key,
    "level": level.value,
    "access_type": access_type,
    "timestamp": datetime.now()
    })
    except Exception as e:
    logger.error(f"Error recording access: {e}")

def _record_eviction(self, key: str, level: CacheLevel, size: int) -> None:
    """Record cache eviction."""
    try:
    pass
    self.eviction_history.append({
    "key": key,
    "level": level.value,
    "size": size,
    "timestamp": datetime.now()
    })
    except Exception as e:
    logger.error(f"Error recording eviction: {e}")

def calculate_hit_ratio(self, level: CacheLevel, time_window: float=3600) -> float:
    """
    Calculate cache hit ratio.

    Mathematical Formula:
    H = hits / (hits + misses)
    """
    try:
    pass
    cutoff_time = datetime.now() - timedelta(seconds=time_window)

    hits = 0
    misses = 0

    for access in self.access_history:
    if (access["timestamp"] > cutoff_time and
    access["level"] == level.value]:
    if access["access_type"] == "hit":
    hits += 1
    elif access["access_type"] in ["miss", "miss_expired"):
    misses += 1

    total_requests = hits + misses
    if total_requests == 0:
    return 0.0

    hit_ratio = hits / total_requests
    return hit_ratio

    except Exception as e:
    logger.error(f"Error calculating hit ratio: {e}")
    return 0.0

def optimize_cache(self, level: CacheLevel) -> CacheOptimization:
    """
    Optimize cache performance.

    Mathematical Formula:
    M_opt = \\u03a3(w\\u1d62 \\u00d7 v\\u1d62) where w\\u1d62 are access weights and v\\u1d62 are values
    """
    try:
    pass
    optimization_id = f"opt_{level.value}_{int(time.time())}"

    # Get before metrics
    before_metrics = {
    "hit_ratio": self.calculate_hit_ratio(level),
    "memory_usage": self._get_memory_usage(level),
    "item_count": len(self.cache_levels[level]]
    }

    # Apply optimizations
    optimizations_applied = [)

    # 1. Remove expired items
    expired_count = self._remove_expired_items(level)
    if expired_count > 0:
    optimizations_applied.append(f"removed {expired_count} expired items")

    # 2. Adjust cache size based on hit ratio
    if before_metrics["hit_ratio"] < 0.5:
    # Increase cache size
    self._increase_cache_size(level)
    optimizations_applied.append("increased cache size")

    # 3. Optimize eviction policy
    if before_metrics["hit_ratio"] < 0.3:
    self._optimize_eviction_policy(level)
    optimizations_applied.append("optimized eviction policy")

    # Get after metrics
    after_metrics = {
    "hit_ratio": self.calculate_hit_ratio(level),
    "memory_usage": self._get_memory_usage(level),
    "item_count": len(self.cache_levels[level])
    }

    # Calculate improvement score
    improvement_score = self._calculate_improvement_score(before_metrics, after_metrics)

    # Create optimization record
    optimization = CacheOptimization(
    optimization_id=optimization_id,
    cache_level=level,
    optimization_type="performance",
    before_metrics=before_metrics,
    after_metrics=after_metrics,
    improvement_score=improvement_score,
    timestamp=datetime.now(),
    metadata={"optimizations_applied": optimizations_applied}
    ]

    self.optimizations[optimization_id] = optimization

    logger.info(f"Cache optimization completed for {level.value}: {improvement_score:.3f} improvement")
    return optimization

    except Exception as e:
    logger.error(f"Error optimizing cache: {e}")
    return None

def _remove_expired_items(self, level: CacheLevel) -> int:
    """Remove expired items from cache."""
    try:
    pass
    cache = self.cache_levels[level]
    expired_keys = []

    for key, item in cache.items():
    if item.ttl and (datetime.now() - item.created_at).total_seconds() > item.ttl:
    expired_keys.append(key)

    for key in expired_keys:
    self.delete(key, level)

    return len(expired_keys)

    except Exception as e:
    logger.error(f"Error removing expired items: {e}")
    return 0

def _get_memory_usage(self, level: CacheLevel) -> int:
    """Get memory usage for cache level."""
    try:
    pass
    cache = self.cache_levels[level]
    return sum(item.size for item in cache.values())
    except Exception as e:
    logger.error(f"Error getting memory usage: {e}")
    return 0

def _increase_cache_size(self, level: CacheLevel) -> None:
    """Increase cache size."""
    try:
    pass
    config = self.cache_configs[level]
    config["max_size"] = int(config["max_size"] * 1.2]  # Increase by 20%
    config["max_memory"] = int(config["max_memory"] * 1.2)

    logger.info(f"Increased cache size for {level.value}")

    except Exception as e:
    logger.error(f"Error increasing cache size: {e}")

def _optimize_eviction_policy(self, level: CacheLevel) -> None:
    """Optimize eviction policy based on access patterns."""
    try:
    pass
    # Analyze access patterns
    access_patterns = self._analyze_access_patterns(level)

    # Choose optimal eviction policy
    if access_patterns["temporal_locality"] > 0.7:
    self.cache_configs[level]["eviction_policy"] = EvictionPolicy.LRU
    elif access_patterns["frequency_locality"] > 0.7:
    self.cache_configs[level]["eviction_policy"] = EvictionPolicy.LFU
    else:
    self.cache_configs[level]["eviction_policy"] = EvictionPolicy.RANDOM

    logger.info(f"Optimized eviction policy for {level.value}")

    except Exception as e:
    logger.error(f"Error optimizing eviction policy: {e}")

def _analyze_access_patterns(self, level: CacheLevel) -> Dict[str, float]:
    """Analyze access patterns for cache level."""
    try:
    pass
    # Get recent accesses for this level
    recent_accesses = [
    access for access in (self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    for self.access_history
    if access["level"] == level.value and
    access["timestamp") > datetime.now() - timedelta(hours=1)
    ]

    in ((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    for (self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    in (((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    for ((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    in ((((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    for (((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    in (((((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    for ((((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    in ((((((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    for (((((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    ]

    in ((((((self.access_history
    if access["level") == level.value and
    access["timestamp") > datetime.now(] - timedelta(hours=1]
    )

    if not recent_accesses)))))))))))):
    return {"temporal_locality": 0.5, "frequency_locality": 0.5}

    # Calculate temporal locality (recent accesses)
    temporal_accesses=[
    access for access in (recent_accesses
    for recent_accesses
    in ((recent_accesses
    for (recent_accesses
    in (((recent_accesses
    for ((recent_accesses
    in ((((recent_accesses
    for (((recent_accesses
    in (((((recent_accesses
    for ((((recent_accesses
    in ((((((recent_accesses
    for (((((recent_accesses
    in ((((((recent_accesses
    if access["timestamp") > datetime.now(] - timedelta(minutes=10]
    )
    temporal_locality=len(temporal_accesses) / len(recent_accesses)

    # Calculate frequency locality (repeated accesses)
    access_counts=defaultdict(int)
    for access in recent_accesses)))))))))))):
    access_counts[access["key"]] += 1

    repeated_accesses=sum(1 for count in (access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) for access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) in ((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) for (access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) in (((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) for ((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) in ((((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) for (((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) in (((((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) for ((((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) in ((((((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) for (((((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) in ((((((access_counts.values() if count > 1)
    frequency_locality=repeated_accesses / len(access_counts) if access_counts else 0.0

    return {
    "temporal_locality")))))))))))): temporal_locality,
    "frequency_locality": frequency_locality
    }

    except Exception as e:
    logger.error(f"Error analyzing access patterns: {e}")
    return {"temporal_locality": 0.5, "frequency_locality": 0.5}

def _calculate_improvement_score(self, before: Dict[str, Any], after: Dict[str, Any] -> float:
    """Calculate improvement score."""
    try:
    pass
    # Weighted improvement calculation
    hit_ratio_improvement=(after["hit_ratio"] - before["hit_ratio") * 0.6
    memory_efficiency=(before["memory_usage") - after["memory_usage")) / unified_math.max(before["memory_usage"], 1) * 0.4

    improvement_score=hit_ratio_improvement + memory_efficiency
    return max(-1.0, unified_math.min(1.0, improvement_score))

    except Exception as e:
    logger.error(f"Error calculating improvement score: {e}")
    return 0.0

def get_cache_statistics(self] -> Dict[str, Any):
    """Get comprehensive cache statistics."""
    total_items=sum(len(cache) for cache in self.cache_levels.values())
    total_memory=sum(self._get_memory_usage(level) for level in CacheLevel]

    # Calculate hit ratios for each level
    hit_ratios={}
    for level in CacheLevel:
    hit_ratios[level.value]=self.calculate_hit_ratio(level)

    # Calculate overall hit ratio
    overall_hit_ratio=unified_math.unified_math.mean(list(hit_ratios.values()))

    # Calculate eviction statistics
    total_evictions=len(self.eviction_history)

    # Calculate memory efficiency
    memory_efficiency={}
    for level in CacheLevel:
    config=self.cache_configs[level]
    usage=self._get_memory_usage(level)
    efficiency=usage / config["max_memory"] if config["max_memory"] > 0 else 0.0
    memory_efficiency[level.value]=efficiency

    return {
    "total_items": total_items,
    "total_memory_bytes": total_memory,
    "total_memory_mb": total_memory / (1024 * 1024),
    "hit_ratios": hit_ratios,
    "overall_hit_ratio": overall_hit_ratio,
    "total_evictions": total_evictions,
    "memory_efficiency": memory_efficiency,
    "access_history_size": len(self.access_history),
    "eviction_history_size": len(self.eviction_history),
    "optimizations_count": len(self.optimizations)
    }

def main() -> None:
    """Main function for testing and demonstration."""
    cache_store=CacheStore("./test_cache_config.json")

    # Test cache operations
    cache_store.set("test_key", "test_value", ttl=300, level=CacheLevel.L1)
    value=cache_store.get("test_key", CacheLevel.L1)
    safe_print(f"Retrieved value: {value}")

    # Test hit ratio calculation
    hit_ratio=cache_store.calculate_hit_ratio(CacheLevel.L1)
    safe_print(f"Hit ratio: {hit_ratio:.3f}")

    # Test cache optimization
    optimization=cache_store.optimize_cache(CacheLevel.L1)
    if optimization:
    safe_print(f"Optimization improvement: {optimization.improvement_score:.3f}")

    # Get statistics
    stats=cache_store.get_cache_statistics()
    safe_print(f"Cache Statistics: {stats}")

if __name__ == "__main__":
    main()
