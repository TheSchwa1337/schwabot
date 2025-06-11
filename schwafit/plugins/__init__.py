"""
Schwafit Plugin System

Provides a modular framework for registering and executing plugins that can
contribute to Schwafit's scoring and strategy selection logic.
"""

from typing import List, Callable, Any, Dict
import logging
from schwabot.utils.oracle_guard import safe_plugin_execution

logger = logging.getLogger("schwafit_plugins")

# Registry for plugin functions
registered_plugins: List[Callable] = []

def register_plugin(fn: Callable) -> Callable:
    """
    Decorator to register a plugin function.
    
    Args:
        fn: The plugin function to register
        
    Returns:
        The original function
    """
    registered_plugins.append(fn)
    logger.info(f"Registered plugin: {fn.__name__}")
    return fn

def run_plugins(market_state: Dict[str, Any], oracle_insight: Dict[str, Any]) -> List[Any]:
    """
    Safely execute all registered plugins with the given market state and Oracle insight.
    
    Args:
        market_state: Current market state
        oracle_insight: Oracle insights for the current state
        
    Returns:
        List of plugin outputs
    """
    return safe_plugin_execution(
        registered_plugins,
        market_state=market_state,
        oracle_insight=oracle_insight
    )

def clear_plugins() -> None:
    """Clear all registered plugins."""
    registered_plugins.clear()
    logger.info("Cleared all registered plugins") 