#!/usr/bin/env python3
"""
Decision Engine - Mathematical Decision Models and AI-Driven Trading Logic
=======================================================================

This module implements a comprehensive decision engine for Schwabot,
providing mathematical decision models, risk assessment, and AI-driven
trading decisions.

Core Mathematical Functions:
- Decision Score: D(x) = Σ(wᵢ × sᵢ(x)) where wᵢ are decision weights
- Risk Assessment: R(x) = √(Σ(risk_factorᵢ²))
- Confidence Level: C(x) = sigmoid(decision_score / risk_threshold)
- Action Probability: P(a|x) = softmax(decision_scores)

Core Functionality:
- Multi-factor decision analysis
- Risk assessment and management
- Confidence scoring and validation
- Action recommendation system
- Decision history tracking
- Performance analytics
"""

import logging
import json
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import os
import queue
import weakref

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    WAIT = "wait"
    EXIT = "exit"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class DecisionConfig:
    decision_threshold: float = 0.6
    risk_threshold: float = 0.8
    confidence_threshold: float = 0.7
    max_position_size: float = 1.0
    min_position_size: float = 0.1
    enable_risk_management: bool = True
    enable_confidence_scoring: bool = True
    enable_action_validation: bool = True
    decision_history_size: int = 10000

@dataclass
class MarketData:
    timestamp: datetime
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionFactor:
    factor_id: str
    factor_name: str
    weight: float
    value: float
    normalized_value: float
    contribution: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecisionResult:
    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    confidence_level: ConfidenceLevel
    risk_level: RiskLevel
    decision_score: float
    risk_score: float
    confidence_score: float
    factors: List[DecisionFactor]
    recommended_action: str
    position_size: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class DecisionFactorCalculator:
    """Decision factor calculation engine."""
    
    def __init__(self):
        self.factor_weights: Dict[str, float] = {}
        self.factor_history: deque = deque(maxlen=10000)
        self._initialize_factor_weights()
    
    def _initialize_factor_weights(self):
        """Initialize default factor weights."""
        self.factor_weights = {
            'price_momentum': 0.25,
            'volume_trend': 0.20,
            'technical_indicators': 0.30,
            'market_sentiment': 0.15,
            'volatility': 0.10
        }
    
    def calculate_factors(self, market_data: MarketData, 
                         historical_data: List[MarketData] = None) -> List[DecisionFactor]:
        """Calculate decision factors from market data."""
        try:
            factors = []
            
            # Price momentum factor
            momentum_factor = self._calculate_price_momentum(market_data, historical_data)
            factors.append(momentum_factor)
            
            # Volume trend factor
            volume_factor = self._calculate_volume_trend(market_data, historical_data)
            factors.append(volume_factor)
            
            # Technical indicators factor
            technical_factor = self._calculate_technical_indicators(market_data, historical_data)
            factors.append(technical_factor)
            
            # Market sentiment factor
            sentiment_factor = self._calculate_market_sentiment(market_data, historical_data)
            factors.append(sentiment_factor)
            
            # Volatility factor
            volatility_factor = self._calculate_volatility(market_data, historical_data)
            factors.append(volatility_factor)
            
            # Normalize and calculate contributions
            factors = self._normalize_factors(factors)
            
            # Record factor history
            self.factor_history.append({
                'timestamp': market_data.timestamp,
                'factors': [f.__dict__ for f in factors]
            })
            
            return factors
            
        except Exception as e:
            logger.error(f"Error calculating factors: {e}")
            return []
    
    def _calculate_price_momentum(self, market_data: MarketData, 
                                historical_data: List[MarketData] = None) -> DecisionFactor:
        """Calculate price momentum factor."""
        try:
            if not historical_data or len(historical_data) < 2:
                momentum_value = 0.0
            else:
                # Calculate price change over recent periods
                recent_prices = [d.price for d in historical_data[-5:]]
                if len(recent_prices) >= 2:
                    price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                    momentum_value = np.tanh(price_change * 10)  # Normalize to [-1, 1]
                else:
                    momentum_value = 0.0
            
            return DecisionFactor(
                factor_id="price_momentum",
                factor_name="Price Momentum",
                weight=self.factor_weights.get('price_momentum', 0.25),
                value=momentum_value,
                normalized_value=0.0,  # Will be normalized later
                contribution=0.0,  # Will be calculated later
                metadata={'price_change': momentum_value}
            )
            
        except Exception as e:
            logger.error(f"Error calculating price momentum: {e}")
            return DecisionFactor(
                factor_id="price_momentum",
                factor_name="Price Momentum",
                weight=self.factor_weights.get('price_momentum', 0.25),
                value=0.0,
                normalized_value=0.0,
                contribution=0.0
            )
    
    def _calculate_volume_trend(self, market_data: MarketData, 
                              historical_data: List[MarketData] = None) -> DecisionFactor:
        """Calculate volume trend factor."""
        try:
            if not historical_data or len(historical_data) < 2:
                volume_trend = 0.0
            else:
                # Calculate volume trend
                recent_volumes = [d.volume for d in historical_data[-5:]]
                if len(recent_volumes) >= 2:
                    volume_change = (recent_volumes[-1] - recent_volumes[0]) / max(recent_volumes[0], 1)
                    volume_trend = np.tanh(volume_change * 5)  # Normalize to [-1, 1]
                else:
                    volume_trend = 0.0
            
            return DecisionFactor(
                factor_id="volume_trend",
                factor_name="Volume Trend",
                weight=self.factor_weights.get('volume_trend', 0.20),
                value=volume_trend,
                normalized_value=0.0,
                contribution=0.0,
                metadata={'volume_change': volume_trend}
            )
            
        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return DecisionFactor(
                factor_id="volume_trend",
                factor_name="Volume Trend",
                weight=self.factor_weights.get('volume_trend', 0.20),
                value=0.0,
                normalized_value=0.0,
                contribution=0.0
            )
    
    def _calculate_technical_indicators(self, market_data: MarketData, 
                                      historical_data: List[MarketData] = None) -> DecisionFactor:
        """Calculate technical indicators factor."""
        try:
            if not historical_data or len(historical_data) < 10:
                technical_score = 0.0
            else:
                # Calculate simple moving averages
                prices = [d.price for d in historical_data[-10:]]
                sma_5 = np.mean(prices[-5:])
                sma_10 = np.mean(prices)
                
                # Calculate RSI-like indicator
                price_changes = np.diff(prices)
                gains = np.sum(price_changes[price_changes > 0])
                losses = abs(np.sum(price_changes[price_changes < 0]))
                
                if losses > 0:
                    rs = gains / losses
                    rsi = 100 - (100 / (1 + rs))
                    rsi_normalized = (rsi - 50) / 50  # Normalize to [-1, 1]
                else:
                    rsi_normalized = 1.0
                
                # Combine indicators
                sma_signal = 1.0 if market_data.price > sma_5 > sma_10 else -1.0
                technical_score = (sma_signal + rsi_normalized) / 2
            
            return DecisionFactor(
                factor_id="technical_indicators",
                factor_name="Technical Indicators",
                weight=self.factor_weights.get('technical_indicators', 0.30),
                value=technical_score,
                normalized_value=0.0,
                contribution=0.0,
                metadata={'sma_signal': sma_signal if 'sma_signal' in locals() else 0.0,
                         'rsi_normalized': rsi_normalized if 'rsi_normalized' in locals() else 0.0}
            )
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return DecisionFactor(
                factor_id="technical_indicators",
                factor_name="Technical Indicators",
                weight=self.factor_weights.get('technical_indicators', 0.30),
                value=0.0,
                normalized_value=0.0,
                contribution=0.0
            )
    
    def _calculate_market_sentiment(self, market_data: MarketData, 
                                  historical_data: List[MarketData] = None) -> DecisionFactor:
        """Calculate market sentiment factor."""
        try:
            # Simplified sentiment calculation
            # In practice, this would use news sentiment, social media, etc.
            sentiment_score = 0.0
            
            if historical_data and len(historical_data) >= 5:
                # Use price volatility as a sentiment proxy
                prices = [d.price for d in historical_data[-5:]]
                volatility = np.std(prices) / np.mean(prices)
                sentiment_score = -np.tanh(volatility * 10)  # Higher volatility = lower sentiment
            
            return DecisionFactor(
                factor_id="market_sentiment",
                factor_name="Market Sentiment",
                weight=self.factor_weights.get('market_sentiment', 0.15),
                value=sentiment_score,
                normalized_value=0.0,
                contribution=0.0,
                metadata={'volatility': volatility if 'volatility' in locals() else 0.0}
            )
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return DecisionFactor(
                factor_id="market_sentiment",
                factor_name="Market Sentiment",
                weight=self.factor_weights.get('market_sentiment', 0.15),
                value=0.0,
                normalized_value=0.0,
                contribution=0.0
            )
    
    def _calculate_volatility(self, market_data: MarketData, 
                            historical_data: List[MarketData] = None) -> DecisionFactor:
        """Calculate volatility factor."""
        try:
            if not historical_data or len(historical_data) < 5:
                volatility_score = 0.0
            else:
                # Calculate price volatility
                prices = [d.price for d in historical_data[-5:]]
                volatility = np.std(prices) / np.mean(prices)
                volatility_score = -np.tanh(volatility * 20)  # Higher volatility = lower score
            
            return DecisionFactor(
                factor_id="volatility",
                factor_name="Volatility",
                weight=self.factor_weights.get('volatility', 0.10),
                value=volatility_score,
                normalized_value=0.0,
                contribution=0.0,
                metadata={'volatility': volatility if 'volatility' in locals() else 0.0}
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return DecisionFactor(
                factor_id="volatility",
                factor_name="Volatility",
                weight=self.factor_weights.get('volatility', 0.10),
                value=0.0,
                normalized_value=0.0,
                contribution=0.0
            )
    
    def _normalize_factors(self, factors: List[DecisionFactor]) -> List[DecisionFactor]:
        """Normalize factor values and calculate contributions."""
        try:
            if not factors:
                return factors
            
            # Normalize values to [0, 1] range
            values = [f.value for f in factors]
            min_val, max_val = min(values), max(values)
            
            if max_val != min_val:
                for factor in factors:
                    factor.normalized_value = (factor.value - min_val) / (max_val - min_val)
            else:
                for factor in factors:
                    factor.normalized_value = 0.5
            
            # Calculate weighted contributions
            total_weight = sum(f.weight for f in factors)
            if total_weight > 0:
                for factor in factors:
                    factor.contribution = (factor.normalized_value * factor.weight) / total_weight
            
            return factors
            
        except Exception as e:
            logger.error(f"Error normalizing factors: {e}")
            return factors

class RiskAssessor:
    """Risk assessment engine."""
    
    def __init__(self, config: DecisionConfig):
        self.config = config
        self.risk_factors: Dict[str, Callable] = {}
        self.risk_history: deque = deque(maxlen=10000)
        self._initialize_risk_factors()
    
    def _initialize_risk_factors(self):
        """Initialize risk assessment factors."""
        self.risk_factors = {
            'price_volatility': self._assess_price_volatility,
            'volume_volatility': self._assess_volume_volatility,
            'market_conditions': self._assess_market_conditions,
            'position_size': self._assess_position_size,
            'liquidity': self._assess_liquidity
        }
    
    def assess_risk(self, market_data: MarketData, 
                   historical_data: List[MarketData] = None,
                   position_size: float = 0.0) -> Tuple[float, RiskLevel]:
        """Assess overall risk level."""
        try:
            risk_scores = []
            
            for factor_name, factor_func in self.risk_factors.items():
                try:
                    risk_score = factor_func(market_data, historical_data, position_size)
                    risk_scores.append(risk_score)
                except Exception as e:
                    logger.error(f"Error in risk factor {factor_name}: {e}")
                    risk_scores.append(0.5)  # Default moderate risk
            
            # Calculate overall risk score
            overall_risk = np.sqrt(np.mean(np.array(risk_scores) ** 2))
            
            # Determine risk level
            if overall_risk < 0.3:
                risk_level = RiskLevel.LOW
            elif overall_risk < 0.6:
                risk_level = RiskLevel.MEDIUM
            elif overall_risk < 0.8:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            # Record risk assessment
            self.risk_history.append({
                'timestamp': market_data.timestamp,
                'overall_risk': overall_risk,
                'risk_level': risk_level.value,
                'factor_scores': dict(zip(self.risk_factors.keys(), risk_scores))
            })
            
            return overall_risk, risk_level
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return 0.5, RiskLevel.MEDIUM
    
    def _assess_price_volatility(self, market_data: MarketData, 
                               historical_data: List[MarketData] = None,
                               position_size: float = 0.0) -> float:
        """Assess price volatility risk."""
        try:
            if not historical_data or len(historical_data) < 10:
                return 0.5
            
            prices = [d.price for d in historical_data[-10:]]
            volatility = np.std(prices) / np.mean(prices)
            
            # Normalize to [0, 1] risk score
            risk_score = min(volatility * 10, 1.0)
            return risk_score
            
        except Exception as e:
            logger.error(f"Error assessing price volatility: {e}")
            return 0.5
    
    def _assess_volume_volatility(self, market_data: MarketData, 
                                historical_data: List[MarketData] = None,
                                position_size: float = 0.0) -> float:
        """Assess volume volatility risk."""
        try:
            if not historical_data or len(historical_data) < 10:
                return 0.5
            
            volumes = [d.volume for d in historical_data[-10:]]
            volatility = np.std(volumes) / np.mean(volumes)
            
            # Normalize to [0, 1] risk score
            risk_score = min(volatility * 5, 1.0)
            return risk_score
            
        except Exception as e:
            logger.error(f"Error assessing volume volatility: {e}")
            return 0.5
    
    def _assess_market_conditions(self, market_data: MarketData, 
                                historical_data: List[MarketData] = None,
                                position_size: float = 0.0) -> float:
        """Assess market conditions risk."""
        try:
            # Simplified market conditions assessment
            # In practice, this would use market indicators, news, etc.
            
            risk_score = 0.5  # Default moderate risk
            
            if historical_data and len(historical_data) >= 5:
                # Check for extreme price movements
                prices = [d.price for d in historical_data[-5:]]
                price_changes = np.diff(prices)
                max_change = np.max(np.abs(price_changes))
                avg_price = np.mean(prices)
                
                if avg_price > 0:
                    relative_change = max_change / avg_price
                    if relative_change > 0.1:  # 10% change
                        risk_score = min(relative_change * 5, 1.0)
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return 0.5
    
    def _assess_position_size(self, market_data: MarketData, 
                            historical_data: List[MarketData] = None,
                            position_size: float = 0.0) -> float:
        """Assess position size risk."""
        try:
            # Position size risk increases with larger positions
            risk_score = min(position_size * 2, 1.0)
            return risk_score
            
        except Exception as e:
            logger.error(f"Error assessing position size: {e}")
            return 0.5
    
    def _assess_liquidity(self, market_data: MarketData, 
                         historical_data: List[MarketData] = None,
                         position_size: float = 0.0) -> float:
        """Assess liquidity risk."""
        try:
            # Simplified liquidity assessment based on spread
            spread_ratio = market_data.spread / market_data.price
            risk_score = min(spread_ratio * 100, 1.0)  # Higher spread = higher risk
            return risk_score
            
        except Exception as e:
            logger.error(f"Error assessing liquidity: {e}")
            return 0.5

class DecisionEngine:
    """Main decision engine."""
    
    def __init__(self, config: DecisionConfig):
        self.config = config
        self.factor_calculator = DecisionFactorCalculator()
        self.risk_assessor = RiskAssessor(config)
        self.decision_history: deque = deque(maxlen=config.decision_history_size)
        self.is_initialized = False
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the decision engine."""
        try:
            self.is_initialized = True
            logger.info("Decision engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing decision engine: {e}")
    
    def make_decision(self, market_data: MarketData, 
                     historical_data: List[MarketData] = None,
                     current_position: float = 0.0) -> DecisionResult:
        """Make a trading decision."""
        try:
            if not self.is_initialized:
                logger.error("Decision engine not initialized")
                return self._create_default_decision(market_data)
            
            # Calculate decision factors
            factors = self.factor_calculator.calculate_factors(market_data, historical_data)
            
            # Calculate decision score
            decision_score = self._calculate_decision_score(factors)
            
            # Assess risk
            risk_score, risk_level = self.risk_assessor.assess_risk(
                market_data, historical_data, current_position
            )
            
            # Calculate confidence
            confidence_score = self._calculate_confidence(decision_score, risk_score)
            confidence_level = self._get_confidence_level(confidence_score)
            
            # Determine decision type
            decision_type = self._determine_decision_type(decision_score, confidence_score, risk_score)
            
            # Calculate position size
            position_size = self._calculate_position_size(decision_score, confidence_score, risk_score)
            
            # Create decision result
            decision_result = DecisionResult(
                decision_id=f"decision_{int(time.time() * 1000)}",
                timestamp=market_data.timestamp,
                decision_type=decision_type,
                confidence_level=confidence_level,
                risk_level=risk_level,
                decision_score=decision_score,
                risk_score=risk_score,
                confidence_score=confidence_score,
                factors=factors,
                recommended_action=self._get_recommended_action(decision_type, position_size),
                position_size=position_size,
                metadata={
                    'current_position': current_position,
                    'market_price': market_data.price,
                    'market_volume': market_data.volume
                }
            )
            
            # Record decision
            self.decision_history.append(decision_result)
            
            return decision_result
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return self._create_default_decision(market_data)
    
    def _calculate_decision_score(self, factors: List[DecisionFactor]) -> float:
        """Calculate overall decision score."""
        try:
            if not factors:
                return 0.0
            
            # Weighted sum of factor contributions
            decision_score = sum(f.contribution for f in factors)
            
            # Normalize to [-1, 1] range
            decision_score = np.tanh(decision_score * 2)
            
            return float(decision_score)
            
        except Exception as e:
            logger.error(f"Error calculating decision score: {e}")
            return 0.0
    
    def _calculate_confidence(self, decision_score: float, risk_score: float) -> float:
        """Calculate confidence score."""
        try:
            # Confidence increases with decision strength and decreases with risk
            confidence = abs(decision_score) * (1 - risk_score)
            
            # Apply sigmoid function for smooth scaling
            confidence = 1 / (1 + np.exp(-confidence * 5))
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _get_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Get confidence level from score."""
        try:
            if confidence_score < 0.2:
                return ConfidenceLevel.VERY_LOW
            elif confidence_score < 0.4:
                return ConfidenceLevel.LOW
            elif confidence_score < 0.6:
                return ConfidenceLevel.MEDIUM
            elif confidence_score < 0.8:
                return ConfidenceLevel.HIGH
            else:
                return ConfidenceLevel.VERY_HIGH
                
        except Exception as e:
            logger.error(f"Error getting confidence level: {e}")
            return ConfidenceLevel.MEDIUM
    
    def _determine_decision_type(self, decision_score: float, 
                               confidence_score: float, risk_score: float) -> DecisionType:
        """Determine decision type based on scores."""
        try:
            # Check if confidence is too low
            if confidence_score < self.config.confidence_threshold:
                return DecisionType.WAIT
            
            # Check if risk is too high
            if risk_score > self.config.risk_threshold:
                return DecisionType.EXIT
            
            # Determine action based on decision score
            if decision_score > self.config.decision_threshold:
                return DecisionType.BUY
            elif decision_score < -self.config.decision_threshold:
                return DecisionType.SELL
            else:
                return DecisionType.HOLD
                
        except Exception as e:
            logger.error(f"Error determining decision type: {e}")
            return DecisionType.HOLD
    
    def _calculate_position_size(self, decision_score: float, 
                               confidence_score: float, risk_score: float) -> float:
        """Calculate recommended position size."""
        try:
            # Base position size on decision strength and confidence
            base_size = abs(decision_score) * confidence_score
            
            # Adjust for risk
            risk_adjustment = 1 - risk_score
            position_size = base_size * risk_adjustment
            
            # Apply limits
            position_size = max(self.config.min_position_size, 
                              min(self.config.max_position_size, position_size))
            
            return float(position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.config.min_position_size
    
    def _get_recommended_action(self, decision_type: DecisionType, 
                              position_size: float) -> str:
        """Get recommended action description."""
        try:
            actions = {
                DecisionType.BUY: f"BUY with {position_size:.2%} position size",
                DecisionType.SELL: f"SELL with {position_size:.2%} position size",
                DecisionType.HOLD: "HOLD current position",
                DecisionType.WAIT: "WAIT for better conditions",
                DecisionType.EXIT: "EXIT all positions due to high risk"
            }
            
            return actions.get(decision_type, "UNKNOWN")
            
        except Exception as e:
            logger.error(f"Error getting recommended action: {e}")
            return "ERROR"
    
    def _create_default_decision(self, market_data: MarketData) -> DecisionResult:
        """Create a default decision when processing fails."""
        try:
            return DecisionResult(
                decision_id=f"default_{int(time.time() * 1000)}",
                timestamp=market_data.timestamp,
                decision_type=DecisionType.WAIT,
                confidence_level=ConfidenceLevel.LOW,
                risk_level=RiskLevel.MEDIUM,
                decision_score=0.0,
                risk_score=0.5,
                confidence_score=0.0,
                factors=[],
                recommended_action="WAIT - Default decision due to processing error",
                position_size=0.0,
                metadata={'error': 'Default decision created due to processing error'}
            )
            
        except Exception as e:
            logger.error(f"Error creating default decision: {e}")
            return None
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get decision engine statistics."""
        try:
            if not self.decision_history:
                return {'total_decisions': 0}
            
            decisions = list(self.decision_history)
            
            # Decision type distribution
            decision_counts = defaultdict(int)
            for decision in decisions:
                decision_counts[decision.decision_type.value] += 1
            
            # Confidence level distribution
            confidence_counts = defaultdict(int)
            for decision in decisions:
                confidence_counts[decision.confidence_level.value] += 1
            
            # Risk level distribution
            risk_counts = defaultdict(int)
            for decision in decisions:
                risk_counts[decision.risk_level.value] += 1
            
            # Performance metrics
            decision_scores = [d.decision_score for d in decisions]
            confidence_scores = [d.confidence_score for d in decisions]
            risk_scores = [d.risk_score for d in decisions]
            
            stats = {
                'total_decisions': len(decisions),
                'decision_distribution': dict(decision_counts),
                'confidence_distribution': dict(confidence_counts),
                'risk_distribution': dict(risk_counts),
                'avg_decision_score': float(np.mean(decision_scores)),
                'avg_confidence_score': float(np.mean(confidence_scores)),
                'avg_risk_score': float(np.mean(risk_scores)),
                'max_decision_score': float(np.max(decision_scores)),
                'min_decision_score': float(np.min(decision_scores))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting decision statistics: {e}")
            return {'total_decisions': 0}

def main():
    """Main function for testing."""
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create decision engine configuration
        config = DecisionConfig(
            decision_threshold=0.3,
            risk_threshold=0.7,
            confidence_threshold=0.6,
            max_position_size=0.5,
            min_position_size=0.1
        )
        
        # Create decision engine
        engine = DecisionEngine(config)
        
        # Create test market data
        base_price = 50000.0
        historical_data = []
        
        for i in range(20):
            timestamp = datetime.now() - timedelta(minutes=20-i)
            price = base_price + np.random.normal(0, 100)
            volume = np.random.uniform(50, 200)
            
            market_data = MarketData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                bid=price - 0.5,
                ask=price + 0.5,
                spread=1.0,
                indicators={'sma_20': price + np.random.normal(0, 50)}
            )
            
            historical_data.append(market_data)
        
        # Make decisions
        for i in range(5):
            current_data = historical_data[-(i+1)]
            decision = engine.make_decision(current_data, historical_data, 0.0)
            
            print(f"Decision {i+1}:")
            print(f"  Type: {decision.decision_type.value}")
            print(f"  Confidence: {decision.confidence_level.value}")
            print(f"  Risk: {decision.risk_level.value}")
            print(f"  Score: {decision.decision_score:.3f}")
            print(f"  Position Size: {decision.position_size:.3f}")
            print(f"  Action: {decision.recommended_action}")
            print("-" * 50)
        
        # Get engine statistics
        stats = engine.get_decision_statistics()
        print("Decision Engine Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 