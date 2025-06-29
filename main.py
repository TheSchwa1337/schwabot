#!/usr/bin/env python3
"""
Schwabot Main Entry Point with Full Schwafit Integration
=======================================================

Main entry point for the Schwabot trading system with complete Schwafit integration.
Handles initialization, configuration, and graceful shutdown of all core components.

Key Features:
- Full Schwafit trading system integration
- Real-time market data processing
- Automated trade execution with mathematical frameworks
- Portfolio management and risk controls
- Performance monitoring and adaptive learning
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Import core mathematical systems
try:
    from core.unified_math_system import UnifiedMathSystem
    from core.schwafit_trading_integration import SchwafitTradingIntegration, schwafit_trading_integration
    from dual_unicore_handler import DualUnicoreHandler
    from schwabot import get_info, initialize, shutdown
    CORE_SYSTEMS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Core systems not fully available: {e}")
    CORE_SYSTEMS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize core systems if available
unified_math = UnifiedMathSystem() if CORE_SYSTEMS_AVAILABLE else None
unicore = DualUnicoreHandler() if CORE_SYSTEMS_AVAILABLE else None


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('schwabot.log', encoding='utf-8')
]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Schwabot Trading System with Schwafit Integration")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/default.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual trading)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with simulated trading"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=["BTC/USDT", "ETH/USDT"],
        help="Trading symbols to monitor"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Number of trading cycles to run"
    )
    return parser.parse_args()


async def run_schwafit_trading_cycle(
    schwafit_integration: SchwafitTradingIntegration,
    symbols: List[str],
    cycle_count: int
) -> None:
    """
    Run Schwafit trading cycles for specified symbols.
    
    Args:
        schwafit_integration: Schwafit trading integration instance
        symbols: List of trading symbols to monitor
        cycle_count: Number of cycles to run
    """
    logger.info(f"🚀 Starting Schwafit trading cycles for {len(symbols)} symbols")
    
    for cycle in range(cycle_count):
        try:
            logger.info(f"📊 Trading Cycle {cycle + 1}/{cycle_count}")
            
            # Process each symbol
            for symbol in symbols:
                # Generate simulated market data (in real implementation, this would come from exchange)
                market_data = generate_market_data(symbol)
                
                # Run Schwafit trading cycle
                result = await schwafit_integration.run_trading_cycle(market_data)
                
                if "error" in result:
                    logger.error(f"Error in trading cycle for {symbol}: {result['error']}")
                    continue
                
                # Log results
                signal = result.get("signal")
                execution = result.get("execution_result", {})
                portfolio = result.get("portfolio_state")
                
                if signal:
                    logger.info(f"📈 {symbol}: {signal.signal_type.value} | "
                              f"Confidence: {signal.confidence:.3f} | "
                              f"Schwafit Score: {signal.schwafit_score:.3f}")
                
                if execution.get("executed"):
                    logger.info(f"✅ {symbol}: Trade executed successfully")
                
                if portfolio:
                    logger.info(f"💰 Portfolio Health: {portfolio.schwafit_health_score:.3f} | "
                              f"Risk Exposure: {portfolio.risk_exposure:.3f}")
            
            # Get performance summary
            if cycle % 10 == 0:  # Every 10 cycles
                performance = schwafit_integration.get_performance_summary()
                logger.info(f"📊 Performance Summary: "
                          f"Trades: {performance['total_trades']} | "
                          f"Health: {performance['schwafit_health']:.3f}")
            
            # Wait between cycles
            await asyncio.sleep(5)  # 5-second intervals between cycles
            
        except KeyboardInterrupt:
            logger.info("🛑 Trading cycles interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Error in trading cycle {cycle + 1}: {e}")
            await asyncio.sleep(10)  # Wait longer on error
    
    logger.info("✅ Schwafit trading cycles completed")


def generate_market_data(symbol: str) -> Dict[str, Any]:
    """
    Generate simulated market data for testing.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dictionary with market data
    """
    # Simulate realistic price movements
    base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
    
    # Generate price series with some volatility
    prices = []
    current_price = base_price
    
    for _ in range(20):
        # Add some random movement
        change_pct = np.random.normal(0, 0.02)  # 2% standard deviation
        current_price *= (1 + change_pct)
        prices.append(current_price)
    
    # Generate volume data
    volumes = [np.random.randint(1000, 5000) for _ in range(20)]
    
    return {
        "symbol": symbol,
        "prices": prices,
        "volumes": volumes,
        "current_price": current_price,
        "timestamp": datetime.now().isoformat()
}
async def run_demo_mode(symbols: List[str], cycles: int) -> None:
    """
    Run Schwabot in demo mode with simulated trading.
    
    Args:
        symbols: Trading symbols to monitor
        cycles: Number of trading cycles
    """
    logger.info("🎮 Starting Schwabot Demo Mode")
    
    if not CORE_SYSTEMS_AVAILABLE:
        logger.error("Core systems not available for demo mode")
        return
    
    # Initialize Schwafit trading integration
    schwafit_integration = SchwafitTradingIntegration({
        "demo_mode": True,
        "simulate_trading": True
    })
    
    # Run trading cycles
    await run_schwafit_trading_cycle(schwafit_integration, symbols, cycles)
    
    # Final performance report
    final_performance = schwafit_integration.get_performance_summary()
    logger.info("📊 Final Demo Performance Report:")
    logger.info(f"   Total Trades: {final_performance['total_trades']}")
    logger.info(f"   Portfolio Health: {final_performance['schwafit_health']:.3f}")
    logger.info(f"   Active Signals: {final_performance['active_signals']}")


async def run_live_mode(symbols: List[str], cycles: int, dry_run: bool = False) -> None:
    """
    Run Schwabot in live trading mode.
    
    Args:
        symbols: Trading symbols to monitor
        cycles: Number of trading cycles
        dry_run: Whether to run in dry-run mode
    """
    logger.info("🚀 Starting Schwabot Live Trading Mode")
    
    if not CORE_SYSTEMS_AVAILABLE:
        logger.error("Core systems not available for live trading")
        return
    
    # Initialize Schwafit trading integration with live configuration
    config = {
        "live_trading": True,
        "dry_run": dry_run,
        "risk_management": {
            "max_position_size": 0.1,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05
}
}
    schwafit_integration = SchwafitTradingIntegration(config)
    
    if dry_run:
        logger.info("🔒 Running in DRY-RUN mode - no actual trades will be executed")
    
    # Run trading cycles
    await run_schwafit_trading_cycle(schwafit_integration, symbols, cycles)
    
    # Final performance report
    final_performance = schwafit_integration.get_performance_summary()
    logger.info("📊 Final Live Performance Report:")
    logger.info(f"   Total Trades: {final_performance['total_trades']}")
    logger.info(f"   Portfolio Health: {final_performance['schwafit_health']:.3f}")
    logger.info(f"   Active Signals: {final_performance['active_signals']}")


async def main() -> int:
    """Main entry point for Schwabot with Schwafit integration."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log_level)
        logger.info("🚀 Starting Schwabot Trading System with Schwafit Integration")
        
        # Check core systems availability
        if not CORE_SYSTEMS_AVAILABLE:
            logger.error("Core systems not available. Cannot start Schwabot.")
            return 1
        
        # Initialize system
        logger.info("📊 Initializing Schwabot core systems...")
        init_result = initialize()
        
        if not init_result.get("success", False):
            logger.error(f"Failed to initialize Schwabot: {init_result.get('error', 'Unknown error')}")
            return 1
        
        logger.info("✅ Schwabot initialized successfully")
        
        # Get system info
        info = get_info()
        logger.info(f"📈 System Info: {info}")
        
        # Run appropriate mode
        if args.demo:
            await run_demo_mode(args.symbols, args.cycles)
        else:
            await run_live_mode(args.symbols, args.cycles, args.dry_run)
        
        logger.info("✅ Schwabot trading session completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
        return 0
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {e}")
        return 1
    finally:
        # Graceful shutdown
        logger.info("🔄 Shutting down Schwabot...")
        try:
            shutdown()
            logger.info("✅ Schwabot shutdown completed")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)