#!/usr/bin/env python3
"""
Common - Shared Data Structures and Base Classes
===============================================

This module provides common data structures, base classes, and shared
functionality used throughout the Schwabot system.

Core Data Structures:
- Base classes for all components
- Common data models and types
- Shared interfaces and protocols
- Utility classes and mixins

Core Functionality:
- Base classes for extensibility
- Common data structures
- Shared interfaces
- Utility functions and decorators
- Type definitions and aliases
"""

import logging
import json
import time
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, TypeVar, Generic, Protocol
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import asyncio
import threading
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class ComponentStatus(Enum):
    """Component status enumeration."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"

class ComponentType(Enum):
    """Component type enumeration."""
    SYSTEM = "system"
    TRADING = "trading"
    DATA = "data"
    ANALYSIS = "analysis"
    RISK = "risk"
    STRATEGY = "strategy"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    LOGGING = "logging"

@dataclass
class ComponentInfo:
    """Component information."""
    component_id: str
    name: str
    component_type: ComponentType
    version: str
    description: str
    status: ComponentStatus = ComponentStatus.INACTIVE
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    timestamp: datetime
    component_id: str
    cpu_usage: float
    memory_usage: float
    response_time: float
    throughput: float
    error_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Event:
    """Event data structure."""
    event_id: str
    event_type: str
    component_id: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Message:
    """Message data structure."""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    timestamp: datetime
    content: Any
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Task:
    """Task data structure."""
    task_id: str
    task_type: str
    component_id: str
    created_time: datetime
    scheduled_time: Optional[datetime] = None
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    status: str = "pending"
    priority: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Configuration:
    """Configuration data structure."""
    config_id: str
    component_id: str
    config_type: str
    created_time: datetime
    updated_time: datetime
    data: Dict[str, Any]
    version: str = "1.0.0"
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataPoint:
    """Data point structure."""
    timestamp: datetime
    symbol: str
    data_type: str
    data: Dict[str, Any]
    source: str
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trade:
    """Trade data structure."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    order_id: str
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop', etc.
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    created_time: datetime
    updated_time: datetime
    status: str = "pending"
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Position data structure."""
    position_id: str
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    created_time: datetime
    updated_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Signal:
    """Trading signal structure."""
    signal_id: str
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    source: str
    confidence: float = 0.5
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert structure."""
    alert_id: str
    alert_type: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    timestamp: datetime
    component_id: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_time: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseComponent(ABC):
    """Base class for all components."""
    
    def __init__(self, component_id: str, name: str, component_type: ComponentType, 
                 description: str = "", version: str = "1.0.0"):
        self.component_id = component_id
        self.name = name
        self.component_type = component_type
        self.description = description
        self.version = version
        self.status = ComponentStatus.INACTIVE
        self.created_time = datetime.now()
        self.updated_time = datetime.now()
        self.metadata: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{component_id}")
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.start_time: Optional[datetime] = None
        self.total_operations = 0
        self.failed_operations = 0
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the component."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the component."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the component."""
        pass
    
    def get_info(self) -> ComponentInfo:
        """Get component information."""
        return ComponentInfo(
            component_id=self.component_id,
            name=self.name,
            component_type=self.component_type,
            version=self.version,
            description=self.description,
            status=self.status,
            created_time=self.created_time,
            updated_time=self.updated_time,
            metadata=self.metadata
        )
    
    def update_status(self, status: ComponentStatus) -> None:
        """Update component status."""
        self.status = status
        self.updated_time = datetime.now()
        self.logger.info(f"Component status updated to {status.value}")
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler."""
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """Remove event handler."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].remove(handler)
    
    def emit_event(self, event: Event) -> None:
        """Emit an event."""
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
    
    def add_message_handler(self, message_type: str, handler: Callable) -> None:
        """Add message handler."""
        self.message_handlers[message_type].append(handler)
    
    def remove_message_handler(self, message_type: str, handler: Callable) -> None:
        """Remove message handler."""
        if message_type in self.message_handlers:
            self.message_handlers[message_type].remove(handler)
    
    async def handle_message(self, message: Message) -> Any:
        """Handle a message."""
        handlers = self.message_handlers.get(message.message_type, [])
        results = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(message)
                else:
                    result = handler(message)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")
                results.append(None)
        return results
    
    def record_performance(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        self.performance_history.append(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_history:
            return {}
        
        metrics = list(self.performance_history)
        
        return {
            'total_operations': self.total_operations,
            'failed_operations': self.failed_operations,
            'success_rate': (self.total_operations - self.failed_operations) / max(self.total_operations, 1),
            'avg_cpu_usage': np.mean([m.cpu_usage for m in metrics]),
            'avg_memory_usage': np.mean([m.memory_usage for m in metrics]),
            'avg_response_time': np.mean([m.response_time for m in metrics]),
            'avg_throughput': np.mean([m.throughput for m in metrics]),
            'avg_error_rate': np.mean([m.error_rate for m in metrics]),
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert component to dictionary."""
        return {
            'component_id': self.component_id,
            'name': self.name,
            'component_type': self.component_type.value,
            'version': self.version,
            'description': self.description,
            'status': self.status.value,
            'created_time': self.created_time.isoformat(),
            'updated_time': self.updated_time.isoformat(),
            'metadata': self.metadata,
            'performance': self.get_performance_summary()
        }

class ObservableMixin:
    """Mixin for observable objects."""
    
    def __init__(self):
        self._observers: List[Callable] = []
        self._observer_lock = threading.Lock()
    
    def add_observer(self, observer: Callable) -> None:
        """Add an observer."""
        with self._observer_lock:
            if observer not in self._observers:
                self._observers.append(observer)
    
    def remove_observer(self, observer: Callable) -> None:
        """Remove an observer."""
        with self._observer_lock:
            if observer in self._observers:
                self._observers.remove(observer)
    
    def notify_observers(self, event: str, data: Any = None) -> None:
        """Notify all observers."""
        with self._observer_lock:
            observers = self._observers.copy()
        
        for observer in observers:
            try:
                observer(event, data)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")

class SingletonMixin:
    """Mixin for singleton classes."""
    
    _instances: Dict[str, Any] = {}
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__new__(cls)
        return cls._instances[cls]

class CacheMixin:
    """Mixin for caching functionality."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._max_size = max_size
        self._ttl = ttl
        self._cache_lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._cache_lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if (datetime.now() - timestamp).total_seconds() < self._ttl:
                    return value
                else:
                    del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._cache_lock:
            if len(self._cache) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            
            self._cache[key] = (value, datetime.now())
    
    def clear(self) -> None:
        """Clear cache."""
        with self._cache_lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        with self._cache_lock:
            return len(self._cache)

class RateLimiterMixin:
    """Mixin for rate limiting functionality."""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()
        self._rate_limit_lock = threading.Lock()
    
    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        now = datetime.now()
        
        with self._rate_limit_lock:
            # Remove old requests
            while self.requests and (now - self.requests[0]).total_seconds() > self.time_window:
                self.requests.popleft()
            
            # Check if we can make a new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
        
        return False
    
    def wait_if_needed(self) -> None:
        """Wait if rate limit is exceeded."""
        while not self.can_proceed():
            time.sleep(0.1)

class ValidatorMixin:
    """Mixin for validation functionality."""
    
    def __init__(self):
        self.validators: Dict[str, List[Callable]] = defaultdict(list)
    
    def add_validator(self, field: str, validator: Callable) -> None:
        """Add a validator for a field."""
        self.validators[field].append(validator)
    
    def validate_field(self, field: str, value: Any) -> List[str]:
        """Validate a field value."""
        errors = []
        for validator in self.validators[field]:
            try:
                if not validator(value):
                    errors.append(f"Validation failed for {field}")
            except Exception as e:
                errors.append(f"Validation error for {field}: {e}")
        return errors
    
    def validate_all(self, data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate all fields."""
        errors = {}
        for field, value in data.items():
            field_errors = self.validate_field(field, value)
            if field_errors:
                errors[field] = field_errors
        return errors

class MetricsMixin:
    """Mixin for metrics collection."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.metrics_lock = threading.Lock()
    
    def increment_metric(self, metric: str, value: int = 1) -> None:
        """Increment a metric."""
        with self.metrics_lock:
            self.metrics[metric] += value
    
    def set_metric(self, metric: str, value: Any) -> None:
        """Set a metric value."""
        with self.metrics_lock:
            self.metrics[metric] = value
    
    def get_metric(self, metric: str, default: Any = 0) -> Any:
        """Get a metric value."""
        with self.metrics_lock:
            return self.metrics.get(metric, default)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self.metrics_lock:
            return dict(self.metrics)

class EventEmitter:
    """Event emitter class."""
    
    def __init__(self):
        self._events: Dict[str, List[Callable]] = defaultdict(list)
        self._event_lock = threading.Lock()
    
    def on(self, event: str, handler: Callable) -> None:
        """Register event handler."""
        with self._event_lock:
            self._events[event].append(handler)
    
    def off(self, event: str, handler: Callable) -> None:
        """Remove event handler."""
        with self._event_lock:
            if event in self._events:
                self._events[event].remove(handler)
    
    def emit(self, event: str, *args, **kwargs) -> None:
        """Emit an event."""
        with self._event_lock:
            handlers = self._events[event].copy()
        
        for handler in handlers:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

class DataProcessor(Protocol):
    """Protocol for data processors."""
    
    def process(self, data: Any) -> Any:
        """Process data."""
        ...

class AsyncDataProcessor(Protocol):
    """Protocol for async data processors."""
    
    async def process(self, data: Any) -> Any:
        """Process data asynchronously."""
        ...

class DataValidator(Protocol):
    """Protocol for data validators."""
    
    def validate(self, data: Any) -> bool:
        """Validate data."""
        ...

class DataTransformer(Protocol):
    """Protocol for data transformers."""
    
    def transform(self, data: Any) -> Any:
        """Transform data."""
        ...

class DataAggregator(Protocol):
    """Protocol for data aggregators."""
    
    def aggregate(self, data: List[Any]) -> Any:
        """Aggregate data."""
        ...

def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())

def generate_hash(data: str) -> str:
    """Generate hash of data."""
    return hashlib.sha256(data.encode()).hexdigest()

def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """Convert timestamp to datetime."""
    return datetime.fromtimestamp(timestamp)

def datetime_to_timestamp(dt: datetime) -> float:
    """Convert datetime to timestamp."""
    return dt.timestamp()

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))

def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value between 0 and 1."""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def denormalize(normalized_value: float, min_val: float, max_val: float) -> float:
    """Denormalize value from 0-1 range."""
    return min_val + normalized_value * (max_val - min_val)

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def calculate_compound_growth_rate(initial_value: float, final_value: float, periods: int) -> float:
    """Calculate compound growth rate."""
    if initial_value <= 0 or periods <= 0:
        return 0.0
    return (final_value / initial_value) ** (1 / periods) - 1

def is_valid_email(email: str) -> bool:
    """Check if email is valid."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def is_valid_url(url: str) -> bool:
    """Check if URL is valid."""
    import re
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    return bool(re.match(pattern, url))

def retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                  backoff_factor: float = 2.0, exceptions: Tuple = (Exception,)):
    """Decorator to retry function on error."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator

def async_retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                        backoff_factor: float = 2.0, exceptions: Tuple = (Exception,)):
    """Decorator to retry async function on error."""
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Async attempt {attempt + 1} failed for {func.__name__}: {e}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} async attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator

def time_function(func: Callable) -> Callable:
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper

def async_time_function(func: Callable) -> Callable:
    """Decorator to time async function execution."""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper

def main():
    """Main function for testing."""
    try:
        print("Schwabot Common Module")
        print("=" * 50)
        
        # Test utility functions
        print(f"Generated ID: {generate_id()}")
        print(f"Hash of 'test': {generate_hash('test')}")
        print(f"Safe divide (10/2): {safe_divide(10, 2)}")
        print(f"Clamp (5, 0, 10): {clamp(5, 0, 10)}")
        print(f"Normalize (5, 0, 10): {normalize(5, 0, 10)}")
        print(f"Percentage change (100, 110): {calculate_percentage_change(100, 110)}%")
        
        # Test validation functions
        print(f"Valid email: {is_valid_email('test@example.com')}")
        print(f"Valid URL: {is_valid_url('https://example.com')}")
        
        # Test time functions
        now = datetime.now()
        timestamp = datetime_to_timestamp(now)
        dt = timestamp_to_datetime(timestamp)
        print(f"Timestamp conversion: {now} -> {timestamp} -> {dt}")
        
        # Test formatting
        print(f"Duration format: {format_duration(3661)}")
        
        # Test base component
        class TestComponent(BaseComponent):
            async def initialize(self) -> bool:
                return True
            
            async def start(self) -> bool:
                return True
            
            async def stop(self) -> bool:
                return True
            
            async def shutdown(self) -> bool:
                return True
        
        component = TestComponent("test", "Test Component", ComponentType.SYSTEM)
        print(f"Component info: {component.get_info()}")
        
        print("\nAll common functionality tests completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 