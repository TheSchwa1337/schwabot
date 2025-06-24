#!/usr/bin/env python3
"""
Constants - Mathematical Constants and System Configuration
==========================================================

This module defines all constants used throughout the Schwabot system,
including mathematical constants, trading parameters, and system configuration.

Core Mathematical Constants:
- Mathematical constants (π, e, etc.)
- Financial constants (risk-free rates, etc.)
- Trading constants (timeframes, limits, etc.)
- System constants (defaults, limits, etc.)

Core Functionality:
- Mathematical and financial constants
- Trading system parameters
- System configuration defaults
- Error codes and messages
- Performance thresholds
"""

import math
from typing import Dict, List, Any
from enum import Enum

# Mathematical Constants
PI = math.pi
E = math.e
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
EULER_MASCHERONI = 0.5772156649015329
CATALAN_CONSTANT = 0.9159655941772190

# Financial Constants
RISK_FREE_RATE_ANNUAL = 0.02  # 2% annual risk-free rate
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / 252  # Daily risk-free rate
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 24
TRADING_MINUTES_PER_HOUR = 60

# Time Constants
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365

# Trading Constants
DEFAULT_COMMISSION_RATE = 0.001  # 0.1% commission
DEFAULT_SLIPPAGE = 0.0005  # 0.05% slippage
MAX_POSITION_SIZE = 1.0  # 100% of portfolio
MIN_POSITION_SIZE = 0.001  # 0.1% of portfolio
DEFAULT_STOP_LOSS = 0.02  # 2% stop loss
DEFAULT_TAKE_PROFIT = 0.04  # 4% take profit

# Risk Management Constants
MAX_DAILY_DRAWDOWN = 0.05  # 5% maximum daily drawdown
MAX_PORTFOLIO_RISK = 0.02  # 2% maximum portfolio risk per trade
VAR_CONFIDENCE_LEVEL = 0.95  # 95% VaR confidence level
MAX_CORRELATION_THRESHOLD = 0.7  # Maximum correlation between positions

# Performance Constants
MIN_SHARPE_RATIO = 1.0  # Minimum acceptable Sharpe ratio
MIN_SORTINO_RATIO = 1.0  # Minimum acceptable Sortino ratio
MAX_CALMAR_RATIO = 3.0  # Maximum acceptable Calmar ratio
MIN_WIN_RATE = 0.5  # Minimum acceptable win rate

# Technical Analysis Constants
DEFAULT_RSI_PERIOD = 14
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BOLLINGER_PERIOD = 20
DEFAULT_BOLLINGER_STD = 2
DEFAULT_ATR_PERIOD = 14

# Data Constants
DEFAULT_LOOKBACK_PERIOD = 100
MAX_LOOKBACK_PERIOD = 10000
MIN_DATA_POINTS = 30
DEFAULT_TICK_INTERVAL = 1  # seconds
DEFAULT_CANDLE_INTERVAL = 60  # seconds

# System Constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_CACHE_SIZE = 1000
DEFAULT_THREAD_POOL_SIZE = 4
DEFAULT_TIMEOUT = 30  # seconds
MAX_RETRY_ATTEMPTS = 3
DEFAULT_BATCH_SIZE = 100

# Memory Constants
MAX_MEMORY_USAGE_MB = 1024
MAX_CACHE_SIZE_MB = 100
MAX_LOG_SIZE_MB = 50
MAX_BACKUP_SIZE_MB = 500

# Network Constants
DEFAULT_REQUEST_TIMEOUT = 10  # seconds
MAX_CONCURRENT_REQUESTS = 10
DEFAULT_RATE_LIMIT = 100  # requests per minute
MAX_RETRY_DELAY = 60  # seconds

# Database Constants
DEFAULT_DB_POOL_SIZE = 10
DEFAULT_DB_TIMEOUT = 30  # seconds
MAX_DB_CONNECTIONS = 50
DEFAULT_BATCH_SIZE_DB = 1000

# API Constants
DEFAULT_API_VERSION = "v1"
DEFAULT_API_TIMEOUT = 30  # seconds
MAX_API_RETRIES = 3
DEFAULT_API_RATE_LIMIT = 100  # requests per minute

# Error Codes
class ErrorCodes:
    SUCCESS = 0
    GENERAL_ERROR = 1000
    VALIDATION_ERROR = 1001
    CONFIGURATION_ERROR = 1002
    DATABASE_ERROR = 1003
    NETWORK_ERROR = 1004
    API_ERROR = 1005
    TIMEOUT_ERROR = 1006
    MEMORY_ERROR = 1007
    PERMISSION_ERROR = 1008
    AUTHENTICATION_ERROR = 1009
    AUTHORIZATION_ERROR = 1010
    RATE_LIMIT_ERROR = 1011
    INSUFFICIENT_FUNDS_ERROR = 1012
    INSUFFICIENT_LIQUIDITY_ERROR = 1013
    ORDER_REJECTED_ERROR = 1014
    MARKET_CLOSED_ERROR = 1015
    INVALID_SYMBOL_ERROR = 1016
    INVALID_ORDER_TYPE_ERROR = 1017
    INVALID_QUANTITY_ERROR = 1018
    INVALID_PRICE_ERROR = 1019

# Status Codes
class StatusCodes:
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    EXPIRED = "expired"
    PARTIAL = "partial"
    REJECTED = "rejected"

# Order Types
class OrderTypes:
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"

# Order Sides
class OrderSides:
    BUY = "buy"
    SELL = "sell"

# Time Frames
class TimeFrames:
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    HOUR = "1h"
    FOUR_HOUR = "4h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"

# Data Types
class DataTypes:
    OHLCV = "ohlcv"
    TICK = "tick"
    TRADE = "trade"
    ORDERBOOK = "orderbook"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"

# Strategy Types
class StrategyTypes:
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    GRID_TRADING = "grid_trading"
    DCA = "dca"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"

# Risk Models
class RiskModels:
    VAR = "var"
    CVAR = "cvar"
    KELLY = "kelly"
    BLACK_LITTERMAN = "black_litterman"
    MARKOWITZ = "markowitz"
    MONTE_CARLO = "monte_carlo"

# Optimization Methods
class OptimizationMethods:
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    PARTICLE_SWARM = "particle_swarm"

# Validation Types
class ValidationTypes:
    TYPE = "type"
    RANGE = "range"
    FORMAT = "format"
    DEPENDENCY = "dependency"
    BUSINESS_LOGIC = "business_logic"
    CROSS_FIELD = "cross_field"

# Log Levels
class LogLevels:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Cache Types
class CacheTypes:
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    DATABASE = "database"

# Database Types
class DatabaseTypes:
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    INFLUXDB = "influxdb"

# Exchange Types
class ExchangeTypes:
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"
    SWAPS = "swaps"

# Market Types
class MarketTypes:
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"

# Signal Types
class SignalTypes:
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

# Indicator Types
class IndicatorTypes:
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    OSCILLATOR = "oscillator"

# Configuration Sections
class ConfigSections:
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    STRATEGY = "strategy"
    API = "api"
    DATABASE = "database"
    LOGGING = "logging"
    CACHE = "cache"
    NETWORK = "network"
    SECURITY = "security"

# File Extensions
class FileExtensions:
    JSON = ".json"
    YAML = ".yaml"
    YML = ".yml"
    CSV = ".csv"
    TXT = ".txt"
    LOG = ".log"
    DB = ".db"
    SQLITE = ".sqlite"

# Default File Paths
class DefaultPaths:
    CONFIG = "config/"
    LOGS = "logs/"
    DATA = "data/"
    BACKUP = "backup/"
    CACHE = "cache/"
    TEMP = "temp/"
    EXPORTS = "exports/"
    REPORTS = "reports/"

# Default File Names
class DefaultFiles:
    CONFIG = "config.json"
    LOG = "schwabot.log"
    DATABASE = "schwabot.db"
    BACKUP = "backup.json"
    CACHE = "cache.db"
    STATE = "state.json"

# Mathematical Functions Constants
class MathConstants:
    # Precision constants
    EPSILON = 1e-10
    INFINITY = float('inf')
    NEGATIVE_INFINITY = float('-inf')
    
    # Statistical constants
    CONFIDENCE_95 = 1.96
    CONFIDENCE_99 = 2.576
    CONFIDENCE_99_9 = 3.291
    
    # Financial constants
    COMPOUNDING_FREQUENCIES = {
        'daily': 365,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4,
        'semi_annual': 2,
        'annual': 1
    }

# Trading System Constants
class TradingConstants:
    # Position sizing
    KELLY_FRACTION = 0.25  # Conservative Kelly fraction
    MAX_LEVERAGE = 10.0
    MIN_LEVERAGE = 1.0
    
    # Order management
    MAX_ORDERS_PER_SYMBOL = 10
    MAX_ACTIVE_ORDERS = 100
    ORDER_TIMEOUT = 300  # 5 minutes
    
    # Risk limits
    MAX_DAILY_TRADES = 1000
    MAX_DAILY_VOLUME = 1000000  # $1M
    MAX_POSITION_DURATION = 86400  # 24 hours in seconds
    
    # Performance thresholds
    MIN_PROFIT_FACTOR = 1.1
    MAX_DRAWDOWN_DURATION = 30  # days
    MIN_RETURN_ON_CAPITAL = 0.1  # 10%

# System Performance Constants
class PerformanceConstants:
    # Response time thresholds
    MAX_API_RESPONSE_TIME = 1.0  # seconds
    MAX_DB_QUERY_TIME = 0.1  # seconds
    MAX_CACHE_ACCESS_TIME = 0.001  # seconds
    
    # Throughput thresholds
    MIN_TRADES_PER_SECOND = 10
    MIN_DATA_POINTS_PER_SECOND = 1000
    MAX_MEMORY_USAGE_PERCENT = 80
    
    # Error rate thresholds
    MAX_ERROR_RATE = 0.01  # 1%
    MAX_TIMEOUT_RATE = 0.05  # 5%
    MAX_FAILURE_RATE = 0.001  # 0.1%

# Security Constants
class SecurityConstants:
    # Authentication
    TOKEN_EXPIRY_HOURS = 24
    REFRESH_TOKEN_EXPIRY_DAYS = 30
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 30
    
    # Encryption
    KEY_SIZE = 256
    SALT_SIZE = 32
    ITERATION_COUNT = 100000
    
    # API Security
    MAX_API_KEYS_PER_USER = 5
    API_KEY_EXPIRY_DAYS = 365
    REQUIRED_PERMISSIONS = ['read', 'trade']

# Notification Constants
class NotificationConstants:
    # Email
    SMTP_TIMEOUT = 30
    MAX_EMAIL_RECIPIENTS = 10
    EMAIL_RATE_LIMIT = 10  # per hour
    
    # SMS
    SMS_RATE_LIMIT = 5  # per hour
    MAX_SMS_LENGTH = 160
    
    # Webhook
    WEBHOOK_TIMEOUT = 10
    MAX_WEBHOOK_RETRIES = 3
    WEBHOOK_RATE_LIMIT = 100  # per minute

# Default Values Dictionary
DEFAULT_VALUES = {
    'commission_rate': DEFAULT_COMMISSION_RATE,
    'slippage': DEFAULT_SLIPPAGE,
    'stop_loss': DEFAULT_STOP_LOSS,
    'take_profit': DEFAULT_TAKE_PROFIT,
    'max_position_size': MAX_POSITION_SIZE,
    'min_position_size': MIN_POSITION_SIZE,
    'lookback_period': DEFAULT_LOOKBACK_PERIOD,
    'rsi_period': DEFAULT_RSI_PERIOD,
    'macd_fast': DEFAULT_MACD_FAST,
    'macd_slow': DEFAULT_MACD_SLOW,
    'macd_signal': DEFAULT_MACD_SIGNAL,
    'bollinger_period': DEFAULT_BOLLINGER_PERIOD,
    'bollinger_std': DEFAULT_BOLLINGER_STD,
    'atr_period': DEFAULT_ATR_PERIOD,
    'timeout': DEFAULT_TIMEOUT,
    'batch_size': DEFAULT_BATCH_SIZE,
    'cache_size': DEFAULT_CACHE_SIZE,
    'thread_pool_size': DEFAULT_THREAD_POOL_SIZE,
    'log_level': DEFAULT_LOG_LEVEL,
    'api_timeout': DEFAULT_API_TIMEOUT,
    'db_timeout': DEFAULT_DB_TIMEOUT,
    'rate_limit': DEFAULT_RATE_LIMIT
}

# Error Messages Dictionary
ERROR_MESSAGES = {
    ErrorCodes.SUCCESS: "Operation completed successfully",
    ErrorCodes.GENERAL_ERROR: "An unexpected error occurred",
    ErrorCodes.VALIDATION_ERROR: "Data validation failed",
    ErrorCodes.CONFIGURATION_ERROR: "Configuration error",
    ErrorCodes.DATABASE_ERROR: "Database operation failed",
    ErrorCodes.NETWORK_ERROR: "Network connection failed",
    ErrorCodes.API_ERROR: "API request failed",
    ErrorCodes.TIMEOUT_ERROR: "Operation timed out",
    ErrorCodes.MEMORY_ERROR: "Insufficient memory",
    ErrorCodes.PERMISSION_ERROR: "Permission denied",
    ErrorCodes.AUTHENTICATION_ERROR: "Authentication failed",
    ErrorCodes.AUTHORIZATION_ERROR: "Authorization failed",
    ErrorCodes.RATE_LIMIT_ERROR: "Rate limit exceeded",
    ErrorCodes.INSUFFICIENT_FUNDS_ERROR: "Insufficient funds",
    ErrorCodes.INSUFFICIENT_LIQUIDITY_ERROR: "Insufficient liquidity",
    ErrorCodes.ORDER_REJECTED_ERROR: "Order rejected",
    ErrorCodes.MARKET_CLOSED_ERROR: "Market is closed",
    ErrorCodes.INVALID_SYMBOL_ERROR: "Invalid symbol",
    ErrorCodes.INVALID_ORDER_TYPE_ERROR: "Invalid order type",
    ErrorCodes.INVALID_QUANTITY_ERROR: "Invalid quantity",
    ErrorCodes.INVALID_PRICE_ERROR: "Invalid price"
}

# Success Messages Dictionary
SUCCESS_MESSAGES = {
    'order_placed': "Order placed successfully",
    'order_cancelled': "Order cancelled successfully",
    'position_opened': "Position opened successfully",
    'position_closed': "Position closed successfully",
    'strategy_started': "Strategy started successfully",
    'strategy_stopped': "Strategy stopped successfully",
    'config_saved': "Configuration saved successfully",
    'backup_created': "Backup created successfully",
    'data_exported': "Data exported successfully",
    'system_started': "System started successfully",
    'system_stopped': "System stopped successfully"
}

def get_constant(name: str, default: Any = None) -> Any:
    """Get a constant value by name."""
    try:
        return globals().get(name, default)
    except Exception:
        return default

def get_default_value(key: str, default: Any = None) -> Any:
    """Get a default value by key."""
    return DEFAULT_VALUES.get(key, default)

def get_error_message(code: int) -> str:
    """Get error message by error code."""
    return ERROR_MESSAGES.get(code, "Unknown error")

def get_success_message(key: str) -> str:
    """Get success message by key."""
    return SUCCESS_MESSAGES.get(key, "Operation completed successfully")

def main():
    """Main function for testing."""
    try:
        print("Schwabot Constants Module")
        print("=" * 50)
        
        # Test mathematical constants
        print(f"PI: {PI}")
        print(f"E: {E}")
        print(f"Golden Ratio: {GOLDEN_RATIO}")
        
        # Test financial constants
        print(f"Risk-free rate (annual): {RISK_FREE_RATE_ANNUAL}")
        print(f"Risk-free rate (daily): {RISK_FREE_RATE_DAILY}")
        print(f"Trading days per year: {TRADING_DAYS_PER_YEAR}")
        
        # Test trading constants
        print(f"Default commission rate: {DEFAULT_COMMISSION_RATE}")
        print(f"Default slippage: {DEFAULT_SLIPPAGE}")
        print(f"Max position size: {MAX_POSITION_SIZE}")
        
        # Test system constants
        print(f"Default log level: {DEFAULT_LOG_LEVEL}")
        print(f"Default cache size: {DEFAULT_CACHE_SIZE}")
        print(f"Default timeout: {DEFAULT_TIMEOUT}")
        
        # Test error codes
        print(f"Success code: {ErrorCodes.SUCCESS}")
        print(f"General error code: {ErrorCodes.GENERAL_ERROR}")
        print(f"Validation error code: {ErrorCodes.VALIDATION_ERROR}")
        
        # Test helper functions
        print(f"Constant PI: {get_constant('PI')}")
        print(f"Default commission: {get_default_value('commission_rate')}")
        print(f"Error message: {get_error_message(ErrorCodes.SUCCESS)}")
        print(f"Success message: {get_success_message('order_placed')}")
        
        print("\nAll constants loaded successfully!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 