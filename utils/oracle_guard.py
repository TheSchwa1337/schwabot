"""
Oracle Guard Utilities

Provides safe wrappers and error handling for Oracle interactions to prevent
cascading failures in live trading environments.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from functools import wraps

# Configure logger
logger = logging.getLogger("oracle_guard")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("[ORACLE GUARD] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def safe_oracle_call(fn: Callable, *args, fallback: Any = None, **kwargs) -> Any:
    """
    Safely call any Oracle function, capturing exceptions and returning a fallback.
    
    Args:
        fn: The Oracle function to call
        *args: Positional arguments for the function
        fallback: Value to return if the call fails
        **kwargs: Keyword arguments for the function
        
    Returns:
        The function result or fallback value if the call fails
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Function {fn.__name__} failed: {e}")
        return fallback

def safe_oracle_field(insight: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Safely retrieve and convert a field from oracle insight to float.
    
    Args:
        insight: Dictionary containing Oracle insights
        key: The key to retrieve from the insight
        default: Default value if retrieval or conversion fails
        
    Returns:
        The float value or default if retrieval/conversion fails
    """
    try:
        return float(insight.get(key, default))
    except Exception:
        logger.warning(f"Failed to retrieve float field '{key}' from oracle insight.")
        return default

def safe_plugin_execution(plugins: List[Callable], *args, **kwargs) -> List[Any]:
    """
    Safely execute a list of plugin functions, collecting non-failing outputs.
    
    Args:
        plugins: List of plugin functions to execute
        *args: Positional arguments for the plugins
        **kwargs: Keyword arguments for the plugins
        
    Returns:
        List of successful plugin outputs
    """
    results = []
    for fn in plugins:
        try:
            results.append(fn(*args, **kwargs))
        except Exception as e:
            logger.warning(f"Plugin {fn.__name__} execution failed: {e}")
    return results

def safe_log_fingerprint(fingerprint_logger: Any, strategy_name: str, 
                        oracle_sig: str, profit_delta: float) -> None:
    """
    Safely log a strategy fingerprint with Oracle signature.
    
    Args:
        fingerprint_logger: The fingerprint logging instance
        strategy_name: Name of the strategy
        oracle_sig: Oracle signature string
        profit_delta: Profit delta value
    """
    try:
        fingerprint_logger.log(strategy_name, oracle_sig, profit_delta)
    except Exception as e:
        logger.warning(f"Failed to log fingerprint: {e}")

def oracle_guard(fallback: Any = None):
    """
    Decorator for Oracle-interfacing functions to provide safe execution.
    
    Args:
        fallback: Value to return if the decorated function fails
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return safe_oracle_call(func, *args, fallback=fallback, **kwargs)
        return wrapper
    return decorator 