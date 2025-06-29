#!/usr/bin/env python3
"""
Price Feed Integration System for Schwabot
==========================================

Provides unified price feed access with multiple sources and robust fallback mechanisms:
- CCXT (primary for live trading)
- CoinMarketCap (backup with API key)
- CoinGecko (free backup)
- Local cache and mock data (emergency fallback)

Features:
- Automatic failover between sources
- Rate limiting and caching
- Portfolio-based fallback strategies
- Real-time price updates
- Historical data backlogging
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from core.price_precision_utils import (
    format_price,
    hash_price,
    get_active_decimals,
    get_active_hash_bits,
)
from core.price_event import PriceEvent, EventOrigin
from core.price_event_registry import record as record_price_event
from core.decimals_autotuner import autotune_loop

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceSource(Enum):
    """Available price sources."""
    CCXT = "ccxt"
    COINMARKETCAP = "coinmarketcap"
    COINGECKO = "coingecko"
    CACHE = "cache"
    MOCK = "mock"

@dataclass
class PriceData:
    """Price data structure."""
    symbol: str
    price: float
    volume: float
    timestamp: float
    source: PriceSource
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PriceFeedConfig:
    """Configuration for price feeds."""
    ccxt_enabled: bool = True
    coinmarketcap_enabled: bool = True
    coingecko_enabled: bool = True
    cache_enabled: bool = True
    mock_enabled: bool = True
    
    # Rate limiting
    ccxt_rate_limit: int = 60  # requests per minute
    coinmarketcap_rate_limit: int = 30
    coingecko_rate_limit: int = 50
    
    # Cache settings
    cache_ttl: int = 30  # seconds
    max_cache_size: int = 1000
    
    # Fallback settings
    primary_source: PriceSource = PriceSource.CCXT
    fallback_sequence: List[PriceSource] = field(default_factory=lambda: [
        PriceSource.CCXT,
        PriceSource.COINMARKETCAP,
        PriceSource.COINGECKO,
        PriceSource.CACHE,
        PriceSource.MOCK
    ])

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    def can_make_request(self) -> bool:
        """Check if a request can be made."""
        now = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.window_seconds]
        
        return len(self.requests) < self.max_requests
    
    def record_request(self):
        """Record a request."""
        self.requests.append(time.time())
    
    async def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        while not self.can_make_request():
            await asyncio.sleep(1)

class PriceCache:
    """Local price cache with TTL."""
    
    def __init__(self, ttl_seconds: int = 30, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Tuple[PriceData, float]] = {}
    
    def get(self, symbol: str) -> Optional[PriceData]:
        """Get price data from cache."""
        if symbol not in self.cache:
            return None
        
        price_data, timestamp = self.cache[symbol]
        if time.time() - timestamp > self.ttl_seconds:
            del self.cache[symbol]
            return None
        
        return price_data
    
    def set(self, symbol: str, price_data: PriceData):
        """Set price data in cache."""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[symbol] = (price_data, time.time())
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()

class CCXTPriceFeed:
    """CCXT-based price feed."""
    
    def __init__(self, config: PriceFeedConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.ccxt_rate_limit)
        self.available = False
        self.exchanges = {}
        
        # Try to import CCXT
        try:
            import ccxt
            self.ccxt = ccxt
            self.available = True
            self._initialize_exchanges()
            logger.info("✅ CCXT price feed initialized")
        except ImportError:
            logger.warning("❌ CCXT not available, skipping CCXT price feed")
    
    def _initialize_exchanges(self):
        """Initialize supported exchanges."""
        exchanges_to_try = ['binance', 'coinbase', 'kraken']
        
        for exchange_name in exchanges_to_try:
            try:
                exchange_class = getattr(self.ccxt, exchange_name)
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                
                # Test connection
                exchange.load_markets()
                self.exchanges[exchange_name] = exchange
                logger.info(f"✅ Connected to {exchange_name}")
                
            except Exception as e:
                logger.warning(f"❌ Failed to connect to {exchange_name}: {e}")
    
    async def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get price from CCXT."""
        if not self.available or not self.exchanges:
            return None
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            # Try exchanges in order
            for exchange_name, exchange in self.exchanges.items():
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    
                    price_data = PriceData(
                        symbol=symbol,
                        price=float(ticker['last']),
                        volume=float(ticker['baseVolume']),
                        timestamp=ticker['timestamp'] / 1000,
                        source=PriceSource.CCXT,
                        metadata={
                            'exchange': exchange_name,
                            'bid': ticker.get('bid'),
                            'ask': ticker.get('ask'),
                            'high': ticker.get('high'),
                            'low': ticker.get('low')
}
                    )
                    
                    self.rate_limiter.record_request()
                    return price_data
                    
                except Exception as e:
                    logger.debug(f"Failed to get price from {exchange_name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"CCXT price fetch failed: {e}")
            return None

class CoinMarketCapPriceFeed:
    """CoinMarketCap API price feed."""
    
    def __init__(self, config: PriceFeedConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.coinmarketcap_rate_limit)
        self.api_key = os.getenv('COINMARKETCAP_API_KEY')
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.available = bool(self.api_key)
        
        if self.available:
            logger.info("✅ CoinMarketCap price feed initialized")
        else:
            logger.warning("❌ CoinMarketCap API key not found")
    
    async def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get price from CoinMarketCap."""
        if not self.available:
            return None
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            import aiohttp
            
            # Convert symbol format (e.g., "BTC/USDC" -> "BTC")
            base_symbol = symbol.split('/')[0]
            
            url = f"{self.base_url}/cryptocurrency/quotes/latest"
            params = {
                'symbol': base_symbol,
                'convert': 'USD'
}
            headers = {
                'X-CMC_PRO_API_KEY': self.api_key,
                'Accept': 'application/json'
}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data and base_symbol in data['data']:
                            quote_data = data['data'][base_symbol]
                            quote = quote_data.get('quote', {}).get('USD', {})
                            
                            price_data = PriceData(
                                symbol=symbol,
                                price=float(quote.get('price', 0)),
                                volume=float(quote.get('volume_24h', 0)),
                                timestamp=time.time(),
                                source=PriceSource.COINMARKETCAP,
                                metadata={
                                    'market_cap': quote.get('market_cap'),
                                    'percent_change_24h': quote.get('percent_change_24h'),
                                    'cmc_rank': quote_data.get('cmc_rank')
}
                            )
                            
                            self.rate_limiter.record_request()
                            return price_data
                    
                    logger.warning(f"CoinMarketCap API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"CoinMarketCap price fetch failed: {e}")
            return None

class CoinGeckoPriceFeed:
    """CoinGecko API price feed."""
    
    def __init__(self, config: PriceFeedConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.coingecko_rate_limit)
        self.base_url = "https://api.coingecko.com/api/v3"
        self.available = True
        logger.info("✅ CoinGecko price feed initialized")
    
    async def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get price from CoinGecko."""
        if not self.available:
            return None
        
        try:
            await self.rate_limiter.wait_if_needed()
            
            import aiohttp
            
            # Convert symbol format (e.g., "BTC/USDC" -> "bitcoin")
            symbol_mapping = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'SOL': 'solana',
                'MATIC': 'matic-network'
}
            base_symbol = symbol.split('/')[0]
            coin_id = symbol_mapping.get(base_symbol, base_symbol.lower())
            
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if coin_id in data:
                            coin_data = data[coin_id]
                            
                            price_data = PriceData(
                                symbol=symbol,
                                price=float(coin_data.get('usd', 0)),
                                volume=float(coin_data.get('usd_24h_vol', 0)),
                                timestamp=time.time(),
                                source=PriceSource.COINGECKO,
                                metadata={
                                    'market_cap': coin_data.get('usd_market_cap'),
                                    'price_change_24h': coin_data.get('usd_24h_change')
}
                            )
                            
                            self.rate_limiter.record_request()
                            return price_data
                    
                    logger.warning(f"CoinGecko API error: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"CoinGecko price fetch failed: {e}")
            return None

class MockPriceFeed:
    """Mock price feed for testing and fallback."""
    
    def __init__(self, config: PriceFeedConfig):
        self.config = config
        self.base_prices = {
            'BTC/USDC': 50000.0,
            'ETH/USDC': 3000.0,
            'SOL/USDC': 100.0,
            'MATIC/USDC': 1.0
}
        self.price_history = {}
        logger.info("✅ Mock price feed initialized")
    
    async def get_price(self, symbol: str) -> Optional[PriceData]:
        """Get mock price data."""
        try:
            base_price = self.base_prices.get(symbol, 100.0)
            
            # Add some realistic price movement
            import random
            price_change = random.uniform(-0.02, 0.02)  # ±2%
            current_price = base_price * (1 + price_change)
            
            # Update base price for next call
            self.base_prices[symbol] = current_price
            
            price_data = PriceData(
                symbol=symbol,
                price=current_price,
                volume=random.uniform(100000, 1000000),
                timestamp=time.time(),
                source=PriceSource.MOCK,
                confidence=0.5,  # Lower confidence for mock data
                metadata={
                    'price_change': price_change,
                    'is_mock': True
}
            )
            
            return price_data
            
        except Exception as e:
            logger.error(f"Mock price generation failed: {e}")
            return None

class UnifiedPriceFeed:
    """Unified price feed with multiple sources and fallback."""
    
    def __init__(self, config: PriceFeedConfig = None):
        self.config = config or PriceFeedConfig()
        self.cache = PriceCache(self.config.cache_ttl, self.config.max_cache_size)
        
        # Initialize price feeds
        self.feeds = {}
        
        if self.config.ccxt_enabled:
            self.feeds[PriceSource.CCXT] = CCXTPriceFeed(self.config)
        
        if self.config.coinmarketcap_enabled:
            self.feeds[PriceSource.COINMARKETCAP] = CoinMarketCapPriceFeed(self.config)
        
        if self.config.coingecko_enabled:
            self.feeds[PriceSource.COINGECKO] = CoinGeckoPriceFeed(self.config)
        
        if self.config.mock_enabled:
            self.feeds[PriceSource.MOCK] = MockPriceFeed(self.config)
        
        # Price history for backlogging
        self.price_history: Dict[str, List[PriceData]] = {}
        self.max_history_size = 1000
        
        logger.info(f"✅ Unified price feed initialized with {len(self.feeds)} sources")

        # launch decimals autotuner
        try:
            asyncio.get_event_loop().create_task(autotune_loop())
        except RuntimeError:
            # if not in running loop (e.g., synchronous import), ignore
            pass
    
    async def get_price(self, symbol: str, force_source: Optional[PriceSource] = None) -> Optional[PriceData]:
        """Get price with automatic fallback."""
        try:
            # Check cache first
            if self.config.cache_enabled:
                cached_price = self.cache.get(symbol)
                if cached_price:
                    logger.debug(f"Cache hit for {symbol}")
                    return cached_price
            
            # Determine sources to try
            if force_source:
                sources_to_try = [force_source]
            else:
                sources_to_try = self.config.fallback_sequence.copy()
            
            # Try each source
            for source in sources_to_try:
                if source not in self.feeds:
                    continue
                
                try:
                    price_data = await self.feeds[source].get_price(symbol)
                    
                    if price_data:
                        # Cache the result
                        if self.config.cache_enabled:
                            self.cache.set(symbol, price_data)
                        
                        # Add to history
                        self._add_to_history(price_data)
                        
                        logger.debug(f"Price for {symbol}: ${price_data.price:.2f} from {source.value}")
                        return price_data
                    
                except Exception as e:
                    logger.warning(f"Failed to get price from {source.value}: {e}")
                    continue
            
            logger.error(f"All price sources failed for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Price fetch failed for {symbol}: {e}")
            return None
    
    def _add_to_history(self, price_data: PriceData):
        """Add price data to history for backlogging and emit PriceEvent."""
        symbol = price_data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price_data)
        
        # Keep only recent history
        if len(self.price_history[symbol]) > self.max_history_size:
            self.price_history[symbol] = self.price_history[symbol][-self.max_history_size:]
        
        # -------------------- NEW: record PriceEvent --------------------
        try:
            origin = (
                EventOrigin.DEMO if price_data.source == PriceSource.MOCK else EventOrigin.LIVE
            )
            if price_data.metadata.get("backtest_id"):
                origin = EventOrigin.BACKTEST

            dec = get_active_decimals()
            bits = get_active_hash_bits()
            price_str = format_price(price_data.price, decimals=dec)
            hash_slice = hash_price(price_data.price, decimals=dec, bits=bits)

            record_price_event(
                PriceEvent(
                    timestamp=datetime.utcfromtimestamp(price_data.timestamp),
                    origin=origin,
                    source=price_data.source.value,
                    raw_price=price_data.price,
                    decimals=dec,
                    price_str=price_str,
                    hash_bits=bits,
                    hash_slice=hash_slice,
                    meta=price_data.metadata,
                )
            )
        except Exception as e:
            logger.debug(f"PriceEvent recording failed: {e}")
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, PriceData]:
        """Get prices for multiple symbols."""
        results = {}
        
        # Use asyncio.gather for concurrent requests
        tasks = [self.get_price(symbol) for symbol in symbols]
        price_data_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, price_data in zip(symbols, price_data_list):
            if isinstance(price_data, Exception):
                logger.error(f"Failed to get price for {symbol}: {price_data}")
            elif price_data:
                results[symbol] = price_data
        
        return results
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[PriceData]:
        """Get price history for a symbol."""
        return self.price_history.get(symbol, [])[-limit:]
    
    def get_portfolio_fallback_prices(self, portfolio_symbols: List[str]) -> Dict[str, float]:
        """Get fallback prices for portfolio calculation."""
        fallback_prices = {}
        
        for symbol in portfolio_symbols:
            # Try to get from cache first
            cached = self.cache.get(symbol)
            if cached:
                fallback_prices[symbol] = cached.price
                continue
            
            # Use mock prices as fallback
            mock_feed = self.feeds.get(PriceSource.MOCK)
            if mock_feed:
                fallback_prices[symbol] = mock_feed.base_prices.get(symbol, 100.0)
        
        return fallback_prices

# Global instance
price_feed = UnifiedPriceFeed()

async def test_price_feed():
    """Test the price feed system."""
    print("🚀 Testing Price Feed Integration")
    print("=" * 50)
    
    symbols = ["BTC/USDC", "ETH/USDC", "SOL/USDC"]
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}:")
        
        # Test each source
        for source in [PriceSource.CCXT, PriceSource.COINMARKETCAP, PriceSource.COINGECKO, PriceSource.MOCK]:
            try:
                price_data = await price_feed.get_price(symbol, force_source=source)
                if price_data:
                    print(f"  ✅ {source.value}: ${price_data.price:.2f}")
                else:
                    print(f"  ❌ {source.value}: Failed")
            except Exception as e:
                print(f"  ❌ {source.value}: Error - {e}")
        
        # Test unified feed
        print(f"\n🔄 Testing unified feed for {symbol}:")
        price_data = await price_feed.get_price(symbol)
        if price_data:
            print(f"  ✅ Unified: ${price_data.price:.2f} from {price_data.source.value}")
        else:
            print(f"  ❌ Unified: Failed")
    
    # Test multiple symbols
    print(f"\n📈 Testing multiple symbols:")
    results = await price_feed.get_multiple_prices(symbols)
    for symbol, price_data in results.items():
        print(f"  ✅ {symbol}: ${price_data.price:.2f} from {price_data.source.value}")
    
    print("\n✅ Price feed test completed!")

if __name__ == "__main__":
    asyncio.run(test_price_feed()) 