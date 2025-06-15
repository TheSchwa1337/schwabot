"""
Schwabot AI Oracle - Fitness Module
Enhanced Central Orchestrator for Profit-Seeking Navigation
Integrates ProfitOracle, RegimeOracle, RittleGEMM, and DLT systems
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import yaml
import logging
from pathlib import Path
import asyncio
from collections import deque

# Import our existing mathematical engines
from schwabot.ai_oracles.profit_oracle import ProfitOracle, ProfitSignal
from schwabot.ai_oracles.reigime_oracle import RegimeDetector, MarketRegime
from rittle_gemm import RittleGEMM, RingLayer
from core.fault_bus import FaultBus, FaultType, FaultBusEvent
from profit_cycle_navigator import ProfitCycleNavigator, ProfitVector, ProfitCycleState

logger = logging.getLogger(__name__)

# ------------------------
# Enhanced Data Models
# ------------------------

@dataclass
class UnifiedMarketState:
    """Comprehensive market state combining all oracle inputs"""
    timestamp: datetime
    price_data: Dict[str, float]  # current, high, low, open, close
    volume_data: Dict[str, float]  # volume, avg_volume, volume_ratio
    
    # Regime Analysis
    regime: Optional[MarketRegime] = None
    regime_confidence: float = 0.0
    
    # Profit Signals
    profit_signals: List[ProfitSignal] = None
    profit_momentum: float = 0.0
    
    # Ring Values (RITTLE-GEMM)
    ring_state: Dict = None
    ring_triggers: List[str] = None
    
    # Technical Indicators
    volatility: float = 0.0
    trend_strength: float = 0.0
    volume_profile: float = 0.0
    correlation_matrix: Optional[np.ndarray] = None

@dataclass
class FitnessReport:
    """Enhanced fitness report with actionable trading recommendations"""
    timestamp: datetime
    
    # Core Fitness Metrics
    overall_fitness: float  # -1.0 to +1.0 (sell to buy)
    regime_fitness: float   # How well current conditions fit the regime
    profit_fitness: float   # Profit opportunity strength
    risk_fitness: float     # Risk-adjusted fitness
    
    # Market Analysis
    market_regime: str
    regime_confidence: float
    dominant_factors: Dict[str, float]
    
    # Trading Recommendations
    action_recommendation: str  # "BUY", "SELL", "HOLD", "REDUCE"
    position_size_ratio: float  # 0.0 to 1.0 of available capital
    confidence_level: float     # 0.0 to 1.0
    
    # Risk Management
    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    max_hold_duration: Optional[timedelta] = None
    
    # Supporting Data
    profit_signals: List[ProfitSignal] = None
    ring_analysis: Dict = None
    fault_correlations: Dict = None

@dataclass
class AdaptiveWeights:
    """Dynamic weights that adapt based on market regime and performance"""
    regime_weights: Dict[str, Dict[str, float]]
    performance_multipliers: Dict[str, float]
    adaptation_rate: float = 0.1
    last_update: datetime = datetime.now()

# ------------------------
# Enhanced Fitness Oracle
# ------------------------

class FitnessOracle:
    """
    Central orchestrator for profit-seeking navigation
    Integrates all mathematical engines into unified decision system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize all mathematical engines
        self.profit_oracle = ProfitOracle()
        self.regime_detector = RegimeDetector()
        self.rittle_gemm = RittleGEMM(ring_size=1000)
        self.fault_bus = FaultBus()
        self.profit_navigator = ProfitCycleNavigator(self.fault_bus)
        
        # Fitness calculation components
        self.adaptive_weights = self._initialize_adaptive_weights()
        self.market_memory = deque(maxlen=1000)  # Historical market states
        self.fitness_history = deque(maxlen=500)  # Historical fitness reports
        self.performance_tracker = PerformanceTracker()
        
        # Navigation state
        self.current_regime = "unknown"
        self.regime_stability = 0.0
        self.profit_tier_active = False
        self.last_fitness_update = datetime.now()
        
        # Ring-based pattern recognition
        self.ring_pattern_memory = {}
        self.profit_tier_detector = ProfitTierDetector()
        
        logger.info("FitnessOracle initialized with integrated mathematical engines")

    def _load_config(self, path: Optional[str]) -> Dict:
        """Load configuration with sensible defaults"""
        default_config = {
            "regime_weights": {
                "trending": {"profit": 0.4, "momentum": 0.3, "volatility": 0.1, "pattern": 0.2},
                "ranging": {"profit": 0.2, "momentum": 0.1, "volatility": 0.4, "pattern": 0.3},
                "breakout": {"profit": 0.5, "momentum": 0.3, "volatility": 0.1, "pattern": 0.1}
            },
            "fitness_thresholds": {
                "strong_buy": 0.7, "buy": 0.3, "hold": 0.1, "sell": -0.3, "strong_sell": -0.7
            },
            "adaptation_settings": {
                "learning_rate": 0.05, "performance_window": 50, "regime_stability_threshold": 0.8
            }
        }
        
        if path and Path(path).exists():
            try:
                with open(path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Config load failed, using defaults: {e}")
        
        return default_config

    def _initialize_adaptive_weights(self) -> AdaptiveWeights:
        """Initialize adaptive weight system"""
        return AdaptiveWeights(
            regime_weights=self.config["regime_weights"].copy(),
            performance_multipliers={"profit": 1.0, "risk": 1.0, "regime": 1.0},
            adaptation_rate=self.config["adaptation_settings"]["learning_rate"]
        )

    async def analyze_market_state(self, market_data: Dict[str, Any]) -> UnifiedMarketState:
        """
        Analyze current market state using all integrated engines
        This is where all the mathematical engines work together
        """
        timestamp = datetime.now()
        
        # Extract basic market data
        price_data = {
            "current": market_data.get("price", 0.0),
            "high": market_data.get("high", 0.0),
            "low": market_data.get("low", 0.0),
            "open": market_data.get("open", 0.0),
            "close": market_data.get("close", 0.0)
        }
        
        volume_data = {
            "volume": market_data.get("volume", 0.0),
            "avg_volume": market_data.get("avg_volume", 0.0),
            "volume_ratio": market_data.get("volume_ratio", 1.0)
        }
        
        # 1. REGIME DETECTION
        price_series = market_data.get("price_series", [price_data["current"]])
        volume_series = market_data.get("volume_series", [volume_data["volume"]])
        
        regime = self.regime_detector.detect(price_series, volume_series)
        if regime:
            self.current_regime = regime.regime_id
            self.regime_stability = regime.confidence
        
        # 2. PROFIT SIGNAL ANALYSIS
        # Convert market data to trade snapshots for profit oracle
        from schwabot.schemas.trade_models import TradeSnapshot
        snapshots = [
            TradeSnapshot(
                price=p, 
                volume=v, 
                timestamp=timestamp - timedelta(minutes=i)
            )
            for i, (p, v) in enumerate(zip(price_series[-10:], volume_series[-10:]))
        ]
        
        profit_signal = self.profit_oracle.detect_profit_signal(snapshots)
        profit_signals = [profit_signal] if profit_signal else []
        
        # 3. RITTLE-GEMM RING ANALYSIS
        tick_data = {
            "timestamp": int(timestamp.timestamp()),
            "profit": profit_signal.projected_gain if profit_signal else 0.0,
            "return": (price_data["current"] - price_data["open"]) / price_data["open"] if price_data["open"] > 0 else 0.0,
            "volume": volume_data["volume"],
            "hash_rec": hash(str(market_data)) % 1000 / 1000.0,  # Normalized hash
            "z_score": 0.0,  # Will be calculated
            "drift": 0.0,    # Will be calculated
            "executed": 0,   # No execution yet
            "rebuy": 0       # No rebuy signal yet
        }
        
        ring_state = self.rittle_gemm.process_tick(tick_data)
        should_trigger, strategy_id = self.rittle_gemm.check_strategy_trigger()
        ring_triggers = [strategy_id] if should_trigger else []
        
        # 4. TECHNICAL INDICATORS
        volatility = self._calculate_volatility(price_series)
        trend_strength = self._calculate_trend_strength(price_series)
        volume_profile = self._calculate_volume_profile(volume_series)
        
        # 5. UPDATE PROFIT NAVIGATOR
        profit_vector = self.profit_navigator.update_market_state(
            current_price=price_data["current"],
            current_volume=volume_data["volume"],
            timestamp=timestamp
        )
        
        # Create unified market state
        unified_state = UnifiedMarketState(
            timestamp=timestamp,
            price_data=price_data,
            volume_data=volume_data,
            regime=regime,
            regime_confidence=regime.confidence if regime else 0.0,
            profit_signals=profit_signals,
            profit_momentum=profit_vector.magnitude,
            ring_state=ring_state,
            ring_triggers=ring_triggers,
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile
        )
        
        # Store in memory
        self.market_memory.append(unified_state)
        
        return unified_state

    def calculate_fitness(self, market_state: UnifiedMarketState) -> FitnessReport:
        """
        Calculate comprehensive fitness score using all engine outputs
        This is the core profit-seeking navigation logic
        """
        timestamp = market_state.timestamp
        
        # 1. REGIME FITNESS - How well do conditions match the current regime
        regime_fitness = self._calculate_regime_fitness(market_state)
        
        # 2. PROFIT FITNESS - Strength of profit opportunities
        profit_fitness = self._calculate_profit_fitness(market_state)
        
        # 3. RISK FITNESS - Risk-adjusted fitness
        risk_fitness = self._calculate_risk_fitness(market_state)
        
        # 4. PATTERN FITNESS - RITTLE-GEMM ring pattern analysis
        pattern_fitness = self._calculate_pattern_fitness(market_state)
        
        # 5. ADAPTIVE WEIGHT APPLICATION
        current_weights = self.adaptive_weights.regime_weights.get(
            self.current_regime, 
            self.adaptive_weights.regime_weights["trending"]  # fallback
        )
        
        # Calculate weighted overall fitness
        overall_fitness = (
            profit_fitness * current_weights.get("profit", 0.4) +
            regime_fitness * current_weights.get("regime", 0.2) +
            pattern_fitness * current_weights.get("pattern", 0.2) +
            risk_fitness * current_weights.get("risk", 0.2)
        )
        
        # Apply performance multipliers
        overall_fitness *= self.adaptive_weights.performance_multipliers.get("profit", 1.0)
        
        # Normalize to [-1, 1] range
        overall_fitness = np.tanh(overall_fitness)
        
        # 6. GENERATE TRADING RECOMMENDATION
        action_recommendation, position_size_ratio, confidence_level = self._generate_trade_recommendation(
            overall_fitness, market_state
        )
        
        # 7. RISK MANAGEMENT LEVELS
        stop_loss_level, take_profit_level, max_hold_duration = self._calculate_risk_levels(
            market_state, action_recommendation
        )
        
        # Create fitness report
        fitness_report = FitnessReport(
            timestamp=timestamp,
            overall_fitness=overall_fitness,
            regime_fitness=regime_fitness,
            profit_fitness=profit_fitness,
            risk_fitness=risk_fitness,
            market_regime=self.current_regime,
            regime_confidence=market_state.regime_confidence,
            dominant_factors={
                "profit": profit_fitness,
                "regime": regime_fitness,
                "pattern": pattern_fitness,
                "risk": risk_fitness
            },
            action_recommendation=action_recommendation,
            position_size_ratio=position_size_ratio,
            confidence_level=confidence_level,
            stop_loss_level=stop_loss_level,
            take_profit_level=take_profit_level,
            max_hold_duration=max_hold_duration,
            profit_signals=market_state.profit_signals,
            ring_analysis=market_state.ring_state,
            fault_correlations=self._get_fault_correlations()
        )
        
        # Store fitness history
        self.fitness_history.append(fitness_report)
        
        # Update performance tracking
        self.performance_tracker.update(fitness_report)
        
        # Adapt weights based on performance
        self._adapt_weights_based_on_performance()
        
        return fitness_report

    def _calculate_regime_fitness(self, market_state: UnifiedMarketState) -> float:
        """Calculate how well current conditions match the identified regime"""
        if not market_state.regime:
            return 0.0
        
        regime = market_state.regime
        
        # Regime-specific fitness calculations
        if "trending" in regime.regime_id.lower():
            # For trending regimes, strong trend + low volatility = high fitness
            fitness = regime.trend * (1.0 - regime.volatility) * regime.confidence
        elif "ranging" in regime.regime_id.lower():
            # For ranging regimes, low trend + moderate volatility = high fitness
            fitness = (1.0 - abs(regime.trend)) * regime.volatility * regime.confidence
        else:
            # Default: use regime confidence
            fitness = regime.confidence
        
        return np.clip(fitness, -1.0, 1.0)

    def _calculate_profit_fitness(self, market_state: UnifiedMarketState) -> float:
        """Calculate strength of profit opportunities"""
        fitness = 0.0
        
        # Profit signals from oracle
        if market_state.profit_signals:
            signal_strength = sum(
                signal.projected_gain * signal.confidence 
                for signal in market_state.profit_signals
            ) / len(market_state.profit_signals)
            fitness += signal_strength
        
        # Profit momentum from navigator
        fitness += market_state.profit_momentum * 0.5
        
        # Ring-based profit indicators
        if market_state.ring_triggers:
            fitness += 0.3  # Bonus for ring triggers
        
        return np.clip(fitness, -1.0, 1.0)

    def _calculate_risk_fitness(self, market_state: UnifiedMarketState) -> float:
        """Calculate risk-adjusted fitness"""
        base_fitness = 0.5  # Neutral starting point
        
        # Volatility adjustment (high vol = higher risk = lower fitness)
        vol_adjustment = 1.0 - market_state.volatility
        
        # Volume adjustment (low volume = higher risk = lower fitness)
        volume_ratio = market_state.volume_data.get("volume_ratio", 1.0)
        volume_adjustment = min(volume_ratio, 1.0)
        
        # Regime stability adjustment
        regime_adjustment = market_state.regime_confidence
        
        risk_fitness = base_fitness * vol_adjustment * volume_adjustment * regime_adjustment
        
        return np.clip(risk_fitness * 2 - 1, -1.0, 1.0)  # Scale to [-1, 1]

    def _calculate_pattern_fitness(self, market_state: UnifiedMarketState) -> float:
        """Calculate fitness based on RITTLE-GEMM ring patterns"""
        if not market_state.ring_state:
            return 0.0
        
        fitness = 0.0
        
        # Ring state analysis
        ring_snapshot = self.rittle_gemm.get_ring_snapshot()
        
        # R1 (profit ring) - direct profit signal
        fitness += ring_snapshot.get('R1', 0.0) * 0.3
        
        # R3 (EMA profit) - smoothed profit signal
        fitness += ring_snapshot.get('R3', 0.0) * 0.2
        
        # R8 (executed profit) - proven profit signal
        fitness += ring_snapshot.get('R8', 0.0) * 0.3
        
        # R10 (rebuy signal) - momentum signal
        if ring_snapshot.get('R10', 0.0) > 0.5:
            fitness += 0.2
        
        return np.clip(fitness, -1.0, 1.0)

    def _generate_trade_recommendation(self, fitness: float, market_state: UnifiedMarketState) -> Tuple[str, float, float]:
        """Generate actionable trading recommendation"""
        thresholds = self.config["fitness_thresholds"]
        
        # Determine action
        if fitness >= thresholds["strong_buy"]:
            action = "STRONG_BUY"
            position_size = 0.8  # 80% of available capital
        elif fitness >= thresholds["buy"]:
            action = "BUY"
            position_size = 0.5  # 50% of available capital
        elif fitness >= thresholds["hold"]:
            action = "HOLD"
            position_size = 0.0
        elif fitness >= thresholds["sell"]:
            action = "SELL"
            position_size = 0.3  # Reduce position by 30%
        else:
            action = "STRONG_SELL"
            position_size = 0.8  # Sell 80% of position
        
        # Adjust for regime confidence
        confidence = market_state.regime_confidence
        position_size *= confidence  # Scale position by confidence
        
        # Adjust for volatility (reduce size in high volatility)
        vol_adjustment = 1.0 - market_state.volatility * 0.5
        position_size *= vol_adjustment
        
        # Ensure position size is reasonable
        position_size = np.clip(position_size, 0.0, 1.0)
        
        return action, position_size, confidence

    def _calculate_risk_levels(self, market_state: UnifiedMarketState, action: str) -> Tuple[Optional[float], Optional[float], Optional[timedelta]]:
        """Calculate stop loss, take profit, and max hold duration"""
        if action in ["HOLD"]:
            return None, None, None
        
        current_price = market_state.price_data["current"]
        volatility = market_state.volatility
        
        # Dynamic stop loss based on volatility
        stop_loss_pct = 0.02 + volatility * 0.03  # 2-5% based on volatility
        
        # Dynamic take profit based on profit signals
        take_profit_pct = 0.04 + volatility * 0.02  # 4-6% based on volatility
        
        if action in ["BUY", "STRONG_BUY"]:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        elif action in ["SELL", "STRONG_SELL"]:
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)
        else:
            stop_loss = take_profit = None
        
        # Max hold duration based on regime and volatility
        if volatility > 0.5:
            max_hold = timedelta(hours=2)  # High volatility = shorter holds
        elif self.current_regime == "trending":
            max_hold = timedelta(days=1)   # Trending = longer holds
        else:
            max_hold = timedelta(hours=6)  # Default
        
        return stop_loss, take_profit, max_hold

    def _get_fault_correlations(self) -> Dict:
        """Get current fault correlations from fault bus"""
        try:
            correlations = self.fault_bus.get_profit_correlations()
            return {
                "correlation_count": len(correlations),
                "average_strength": np.mean([c.correlation_strength for c in correlations]) if correlations else 0.0,
                "high_confidence_correlations": len([c for c in correlations if c.confidence > 0.7])
            }
        except Exception as e:
            logger.warning(f"Could not get fault correlations: {e}")
            return {}

    def _adapt_weights_based_on_performance(self):
        """Adapt weights based on recent performance"""
        if len(self.fitness_history) < 10:
            return
        
        # Simple adaptation: increase weights for factors that led to good outcomes
        # This would be enhanced with actual trade outcome data
        recent_reports = list(self.fitness_history)[-10:]
        
        # For now, just adapt based on confidence levels
        avg_confidence = np.mean([r.confidence_level for r in recent_reports])
        
        if avg_confidence > 0.8:
            # High confidence = current weights are good, small adjustments
            pass
        elif avg_confidence < 0.5:
            # Low confidence = need to adjust weights more aggressively
            # This is where you'd implement more sophisticated adaptation
            pass

    # Helper methods for technical calculations
    def _calculate_volatility(self, price_series: List[float]) -> float:
        if len(price_series) < 2:
            return 0.0
        returns = np.diff(price_series) / price_series[:-1]
        return float(np.std(returns))

    def _calculate_trend_strength(self, price_series: List[float]) -> float:
        if len(price_series) < 2:
            return 0.0
        x = np.arange(len(price_series))
        slope, _ = np.polyfit(x, price_series, 1)
        return float(slope / np.mean(price_series))  # Normalized slope

    def _calculate_volume_profile(self, volume_series: List[float]) -> float:
        if not volume_series:
            return 0.0
        return float(np.mean(volume_series))

    async def run_continuous_analysis(self, market_data_stream, callback=None):
        """Run continuous market analysis and fitness calculation"""
        logger.info("Starting continuous fitness analysis")
        
        async for market_data in market_data_stream:
            try:
                # Analyze current market state
                market_state = await self.analyze_market_state(market_data)
                
                # Calculate fitness
                fitness_report = self.calculate_fitness(market_state)
                
                # Log the analysis
                logger.info(
                    f"Fitness: {fitness_report.overall_fitness:.3f} | "
                    f"Action: {fitness_report.action_recommendation} | "
                    f"Confidence: {fitness_report.confidence_level:.3f} | "
                    f"Regime: {fitness_report.market_regime}"
                )
                
                # Call callback if provided
                if callback:
                    await callback(fitness_report, market_state)
                
            except Exception as e:
                logger.error(f"Analysis cycle failed: {e}")
                await asyncio.sleep(1)  # Brief pause before retry

class PerformanceTracker:
    """Track performance of fitness oracle decisions"""
    
    def __init__(self):
        self.decisions = deque(maxlen=1000)
        self.performance_metrics = {}
    
    def update(self, fitness_report: FitnessReport):
        """Update with new fitness report"""
        self.decisions.append({
            'timestamp': fitness_report.timestamp,
            'fitness': fitness_report.overall_fitness,
            'action': fitness_report.action_recommendation,
            'confidence': fitness_report.confidence_level
        })
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.decisions:
            return {}
        
        return {
            'total_decisions': len(self.decisions),
            'avg_fitness': np.mean([d['fitness'] for d in self.decisions]),
            'avg_confidence': np.mean([d['confidence'] for d in self.decisions]),
            'action_distribution': {
                action: len([d for d in self.decisions if d['action'] == action])
                for action in set(d['action'] for d in self.decisions)
            }
        }

class ProfitTierDetector:
    """Detect genuine profit tiers using JuMBO-style clustering"""
    
    def __init__(self):
        self.profit_history = deque(maxlen=200)
        self.tier_clusters = []
    
    def detect_profit_tier(self, fitness_report: FitnessReport) -> bool:
        """Detect if current conditions represent a genuine profit tier"""
        self.profit_history.append(fitness_report.overall_fitness)
        
        if len(self.profit_history) < 20:
            return False
        
        current_fitness = fitness_report.overall_fitness
        historical_fitness = list(self.profit_history)[:-1]
        
        # Statistical anomaly detection
        mean_fitness = np.mean(historical_fitness)
        std_fitness = np.std(historical_fitness)
        
        if std_fitness == 0:
            return False
        
        z_score = abs(current_fitness - mean_fitness) / std_fitness
        
        # Check for clustering (multiple recent high-fitness readings)
        recent_high_fitness = sum(1 for f in list(self.profit_history)[-5:] if f > mean_fitness + std_fitness)
        
        # Profit tier detected if: high z-score AND clustering
        return z_score > 2.0 and recent_high_fitness >= 3
