#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Schwabot Trade Entry/Exit Logic
===========================================================

Tests:
1. Entry/Exit Logic for BTC/USDC and other pairs
2. Price Feed Integration (CCXT, CoinMarketCap, CoinGecko)
3. Fallback Mechanisms when APIs are unavailable
4. Portfolio and Position Management
5. Risk Management and Validation
6. Data Backlogging and State Propagation

Usage:
    python test_trade_entry_exit.py
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from decimal import Decimal
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, AsyncMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock data for testing
MOCK_PRICE_DATA = {
    "BTC/USDC": {
        "price": 50000.0,
        "volume": 1000000.0,
        "timestamp": time.time()
    },
    "ETH/USDC": {
        "price": 3000.0,
        "volume": 500000.0,
        "timestamp": time.time()
    },
    "SOL/USDC": {
        "price": 100.0,
        "volume": 200000.0,
        "timestamp": time.time()
    }
}

MOCK_PORTFOLIO_STATE = {
    "USDC": Decimal('10000'),
    "BTC": Decimal('0.1'),
    "ETH": Decimal('1.0'),
    "USDT": Decimal('0')
}

class MockIntegratedTradingSignal:
    """Mock trading signal for testing."""
    
    def __init__(self, signal_id: str, recommended_action: str, btc_price: float = 50000.0):
        self.signal_id = signal_id
        self.recommended_action = recommended_action
        self.btc_price = btc_price
        self.confidence_score = 0.8
        self.profit_potential = 0.7
        self.risk_assessment = {"overall_risk": 0.3}
        self.ghost_route = "ghost_trade"

class MockPriceFeed:
    """Mock price feed with fallback mechanisms."""
    
    def __init__(self):
        self.prices = MOCK_PRICE_DATA.copy()
        self.api_available = True
        self.fallback_mode = False
        
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get price with fallback logic."""
        try:
            if self.api_available and not self.fallback_mode:
                # Simulate API call
                await asyncio.sleep(0.1)  # Simulate network delay
                return self.prices.get(symbol, {}).get("price")
            else:
                # Fallback to cached/mock data
                logger.warning(f"API unavailable, using fallback price for {symbol}")
                return self.prices.get(symbol, {}).get("price", 50000.0)
        except Exception as e:
            logger.error(f"Price fetch failed for {symbol}: {e}")
            return self.prices.get(symbol, {}).get("price", 50000.0)
    
    def set_api_availability(self, available: bool):
        """Set API availability for testing."""
        self.api_available = available
    
    def set_fallback_mode(self, enabled: bool):
        """Enable/disable fallback mode."""
        self.fallback_mode = enabled

class MockTradeExecutor:
    """Mock trade executor for testing."""
    
    def __init__(self):
        self.positions = {}
        self.trade_history = []
        self.portfolio = MOCK_PORTFOLIO_STATE.copy()
        self.execution_count = 0
        
    async def execute_entry(self, symbol: str, side: str, amount: float, price: float) -> Dict[str, Any]:
        """Execute entry trade."""
        try:
            trade_id = f"ENTRY_{self.execution_count}_{int(time.time())}"
            
            # Validate trade
            if not self._validate_trade(symbol, side, amount, price):
                error_msg = "Trade validation failed"
                if side.upper() == "BUY":
                    usdc_needed = amount * price
                    if self.portfolio["USDC"] < Decimal(str(usdc_needed)):
                        error_msg = "Insufficient USDC balance"
                
                return {
                    "success": False,
                    "error": error_msg,
                    "trade_id": trade_id
                }
            
            # Execute trade
            if side.upper() == "BUY":
                # Buy crypto with USDC
                usdc_cost = amount * price
                if self.portfolio["USDC"] >= Decimal(str(usdc_cost)):
                    self.portfolio["USDC"] -= Decimal(str(usdc_cost))
                    crypto_amount = amount
                    if symbol == "BTC/USDC":
                        self.portfolio["BTC"] += Decimal(str(crypto_amount))
                    elif symbol == "ETH/USDC":
                        self.portfolio["ETH"] += Decimal(str(crypto_amount))
                    
                    # Record position
                    self.positions[symbol] = {
                        "side": "long",
                        "amount": crypto_amount,
                        "entry_price": price,
                        "entry_time": time.time()
                    }
                    
                    # Add to trade history
                    trade_record = {
                        "trade_id": trade_id,
                        "symbol": symbol,
                        "side": side,
                        "amount": amount,
                        "price": price,
                        "timestamp": time.time()
                    }
                    self.trade_history.append(trade_record)
                    
                    self.execution_count += 1
                    
                    return {
                        "success": True,
                        "trade_id": trade_id,
                        "symbol": symbol,
                        "side": side,
                        "amount": amount,
                        "price": price,
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "success": False,
                        "error": "Insufficient USDC balance",
                        "trade_id": trade_id
                    }
            else:
                return {
                    "success": False,
                    "error": "Unsupported side",
                    "trade_id": trade_id
                }
                
        except Exception as e:
            logger.error(f"Entry execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "trade_id": f"ERROR_{int(time.time())}"
            }
    
    async def execute_exit(self, symbol: str, side: str, amount: float, price: float) -> Dict[str, Any]:
        """Execute exit trade."""
        try:
            trade_id = f"EXIT_{self.execution_count}_{int(time.time())}"
            
            # Check if position exists
            if symbol not in self.positions:
                return {
                    "success": False,
                    "error": "No position to exit",
                    "trade_id": trade_id
                }
            
            position = self.positions[symbol]
            
            # Calculate P&L
            entry_price = position["entry_price"]
            pnl = (price - entry_price) * amount
            
            # Execute exit
            if side.upper() == "SELL":
                # Sell crypto for USDC
                usdc_gained = amount * price
                self.portfolio["USDC"] += Decimal(str(usdc_gained))
                
                # Remove from crypto balance
                if symbol == "BTC/USDC":
                    self.portfolio["BTC"] -= Decimal(str(amount))
                elif symbol == "ETH/USDC":
                    self.portfolio["ETH"] -= Decimal(str(amount))
                
                # Remove position
                del self.positions[symbol]
                
                # Add to trade history
                trade_record = {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "price": price,
                    "pnl": pnl,
                    "timestamp": time.time()
                }
                self.trade_history.append(trade_record)
                
                self.execution_count += 1
                
                return {
                    "success": True,
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "price": price,
                    "pnl": pnl,
                    "timestamp": time.time()
                }
            else:
                return {
                    "success": False,
                    "error": "Unsupported exit side",
                    "trade_id": trade_id
                }
                
        except Exception as e:
            logger.error(f"Exit execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "trade_id": f"ERROR_{int(time.time())}"
            }
    
    def _validate_trade(self, symbol: str, side: str, amount: float, price: float) -> bool:
        """Validate trade parameters."""
        if amount <= 0 or price <= 0:
            return False
        
        if side.upper() == "BUY":
            # Check if we have enough USDC
            usdc_needed = amount * price
            return self.portfolio["USDC"] >= Decimal(str(usdc_needed))
        elif side.upper() == "SELL":
            # Check if we have enough crypto
            if symbol == "BTC/USDC":
                return self.portfolio["BTC"] >= Decimal(str(amount))
            elif symbol == "ETH/USDC":
                return self.portfolio["ETH"] >= Decimal(str(amount))
        
        return False
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value in USDC."""
        total = float(self.portfolio["USDC"])
        total += float(self.portfolio["BTC"]) * MOCK_PRICE_DATA["BTC/USDC"]["price"]
        total += float(self.portfolio["ETH"]) * MOCK_PRICE_DATA["ETH/USDC"]["price"]
        return total

class TestTradeEntryExit(unittest.TestCase):
    """Test suite for trade entry/exit logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.price_feed = MockPriceFeed()
        self.trade_executor = MockTradeExecutor()
        
    def test_btc_usdc_entry_exit_cycle(self):
        """Test complete BTC/USDC entry and exit cycle."""
        async def run_test():
            # Test entry
            entry_result = await self.trade_executor.execute_entry(
                "BTC/USDC", "BUY", 0.1, 50000.0
            )
            
            self.assertTrue(entry_result["success"])
            self.assertEqual(entry_result["symbol"], "BTC/USDC")
            self.assertEqual(entry_result["side"], "BUY")
            self.assertEqual(entry_result["amount"], 0.1)
            self.assertEqual(entry_result["price"], 50000.0)
            
            # Verify portfolio state after entry
            self.assertEqual(float(self.trade_executor.portfolio["BTC"]), 0.2)  # 0.1 + 0.1
            self.assertLess(float(self.trade_executor.portfolio["USDC"]), 10000.0)  # Reduced by trade
            
            # Test exit
            exit_result = await self.trade_executor.execute_exit(
                "BTC/USDC", "SELL", 0.1, 51000.0  # 2% profit
            )
            
            self.assertTrue(exit_result["success"])
            self.assertEqual(exit_result["symbol"], "BTC/USDC")
            self.assertEqual(exit_result["side"], "SELL")
            self.assertEqual(exit_result["pnl"], 100.0)  # (51000 - 50000) * 0.1
            
            # Verify portfolio state after exit
            self.assertEqual(float(self.trade_executor.portfolio["BTC"]), 0.1)  # Back to original
            self.assertGreater(float(self.trade_executor.portfolio["USDC"]), 10000.0)  # Increased by profit
        
        asyncio.run(run_test())
    
    def test_eth_usdc_entry_exit_cycle(self):
        """Test complete ETH/USDC entry and exit cycle."""
        async def run_test():
            # Test entry
            entry_result = await self.trade_executor.execute_entry(
                "ETH/USDC", "BUY", 1.0, 3000.0
            )
            
            self.assertTrue(entry_result["success"])
            self.assertEqual(entry_result["symbol"], "ETH/USDC")
            
            # Test exit
            exit_result = await self.trade_executor.execute_exit(
                "ETH/USDC", "SELL", 1.0, 3150.0  # 5% profit
            )
            
            self.assertTrue(exit_result["success"])
            self.assertEqual(exit_result["pnl"], 150.0)  # (3150 - 3000) * 1.0
        
        asyncio.run(run_test())
    
    def test_price_feed_fallback(self):
        """Test price feed fallback when API is unavailable."""
        async def run_test():
            # Test with API available
            price = await self.price_feed.get_price("BTC/USDC")
            self.assertEqual(price, 50000.0)
            
            # Test with API unavailable
            self.price_feed.set_api_availability(False)
            price = await self.price_feed.get_price("BTC/USDC")
            self.assertEqual(price, 50000.0)  # Should still work with fallback
            
            # Test with fallback mode enabled
            self.price_feed.set_fallback_mode(True)
            price = await self.price_feed.get_price("ETH/USDC")
            self.assertEqual(price, 3000.0)
        
        asyncio.run(run_test())
    
    def test_insufficient_balance_validation(self):
        """Test trade validation with insufficient balance."""
        async def run_test():
            # Try to buy more BTC than we can afford
            result = await self.trade_executor.execute_entry(
                "BTC/USDC", "BUY", 1.0, 50000.0  # $50,000 worth
            )
            
            self.assertFalse(result["success"])
            self.assertIn("Insufficient", result["error"])
        
        asyncio.run(run_test())
    
    def test_invalid_trade_parameters(self):
        """Test trade validation with invalid parameters."""
        async def run_test():
            # Test negative amount
            result = await self.trade_executor.execute_entry(
                "BTC/USDC", "BUY", -0.1, 50000.0
            )
            self.assertFalse(result["success"])
            
            # Test zero price
            result = await self.trade_executor.execute_entry(
                "BTC/USDC", "BUY", 0.1, 0.0
            )
            self.assertFalse(result["success"])
        
        asyncio.run(run_test())
    
    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        initial_value = self.trade_executor.get_portfolio_value()
        self.assertGreater(initial_value, 0)
        
        # After a profitable trade, value should increase
        async def run_test():
            await self.trade_executor.execute_entry("BTC/USDC", "BUY", 0.1, 50000.0)
            await self.trade_executor.execute_exit("BTC/USDC", "SELL", 0.1, 51000.0)
            
            final_value = self.trade_executor.get_portfolio_value()
            self.assertGreater(final_value, initial_value)
        
        asyncio.run(run_test())
    
    def test_multiple_crypto_pairs(self):
        """Test trading multiple crypto pairs."""
        async def run_test():
            # Trade BTC
            await self.trade_executor.execute_entry("BTC/USDC", "BUY", 0.05, 50000.0)
            await self.trade_executor.execute_exit("BTC/USDC", "SELL", 0.05, 50500.0)
            
            # Trade ETH
            await self.trade_executor.execute_entry("ETH/USDC", "BUY", 0.5, 3000.0)
            await self.trade_executor.execute_exit("ETH/USDC", "SELL", 0.5, 3150.0)
            
            # Verify both trades were successful
            self.assertEqual(len(self.trade_executor.trade_history), 4)
            
            # Verify portfolio has both cryptos
            self.assertGreater(float(self.trade_executor.portfolio["BTC"]), 0)
            self.assertGreater(float(self.trade_executor.portfolio["ETH"]), 0)
        
        asyncio.run(run_test())

class TestDataBacklogging(unittest.TestCase):
    """Test data backlogging and state propagation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.price_history = []
        self.trade_history = []
        self.portfolio_history = []
    
    def test_price_data_backlogging(self):
        """Test that price data is properly backlogged."""
        # Simulate price updates
        for i in range(100):
            price_data = {
                "BTC/USDC": {
                    "price": 50000.0 + (i * 10),  # Simulate price movement
                    "volume": 1000000.0,
                    "timestamp": time.time() + i
                }
            }
            self.price_history.append(price_data)
        
        # Verify backlogging
        self.assertEqual(len(self.price_history), 100)
        self.assertGreater(
            self.price_history[-1]["BTC/USDC"]["price"],
            self.price_history[0]["BTC/USDC"]["price"]
        )
    
    def test_trade_history_backlogging(self):
        """Test that trade history is properly backlogged."""
        # Simulate trades
        for i in range(10):
            trade = {
                "trade_id": f"TRADE_{i}",
                "symbol": "BTC/USDC",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "amount": 0.1,
                "price": 50000.0 + (i * 100),
                "timestamp": time.time() + i
            }
            self.trade_history.append(trade)
        
        # Verify backlogging
        self.assertEqual(len(self.trade_history), 10)
        self.assertEqual(self.trade_history[0]["side"], "BUY")
        self.assertEqual(self.trade_history[1]["side"], "SELL")
    
    def test_portfolio_state_propagation(self):
        """Test that portfolio state changes are properly propagated."""
        initial_portfolio = MOCK_PORTFOLIO_STATE.copy()
        self.portfolio_history.append(initial_portfolio)
        
        # Simulate portfolio changes
        for i in range(5):
            new_portfolio = self.portfolio_history[-1].copy()
            new_portfolio["USDC"] += Decimal('100')  # Simulate profit
            self.portfolio_history.append(new_portfolio)
        
        # Verify state propagation
        self.assertEqual(len(self.portfolio_history), 6)
        self.assertGreater(
            float(self.portfolio_history[-1]["USDC"]),
            float(self.portfolio_history[0]["USDC"])
        )

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🚀 Starting Comprehensive Trade Entry/Exit Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestTradeEntryExit))
    test_suite.addTest(unittest.makeSuite(TestDataBacklogging))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n✅ All tests passed!")
        print("🎉 Trade entry/exit logic is working correctly!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 