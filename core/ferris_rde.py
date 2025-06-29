#!/usr/bin/env python3
"""
Ferris RDE (Recursive Dualistic Engine) Module
==============================================

Modular Ferris wheel engine with tick, pivot, ascent, descent logic
for Schwabot v0.05. Provides cyclical trading pattern management.
"""

import time
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class FerrisPhase(Enum):
    """Ferris wheel phase enumeration."""
    TICK = "tick"
    PIVOT = "pivot"
    ASCENT = "ascent"
    DESCENT = "descent"


class FerrisState(Enum):
    """Ferris wheel state enumeration."""
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    OPTIMIZING = "optimizing"
    RESETTING = "resetting"


@dataclass
class FerrisCycle:
    """Ferris wheel cycle data."""
    cycle_id: str
    phase: FerrisPhase
    start_time: float
    end_time: Optional[float] = None
    duration: float = 0.0
    performance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FerrisMetrics:
    """Ferris wheel metrics."""
    timestamp: float
    current_phase: FerrisPhase
    cycle_position: float  # 0.0 to 1.0
    momentum: float
    volatility: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FerrisSignal:
    """Ferris wheel trading signal."""
    signal_id: str
    timestamp: float
    phase: FerrisPhase
    signal_type: str  # "buy", "sell", "hold", "scale_in", "scale_out"
    strength: float
    confidence: float
    cycle_context: FerrisCycle
    metadata: Dict[str, Any] = field(default_factory=dict)


class FerrisRDE:
    """
    Ferris RDE (Recursive Dualistic Engine) for Schwabot v0.05.
    
    Provides modular Ferris wheel engine with tick, pivot, ascent, descent logic
    for cyclical trading pattern management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Ferris RDE."""
        self.config = config or self._default_config()
        
        # Cycle management
        self.current_cycle: Optional[FerrisCycle] = None
        self.cycle_history: List[FerrisCycle] = []
        self.max_history_size = self.config.get('max_history_size', 100)
        
        # Phase tracking
        self.current_phase = FerrisPhase.TICK
        self.phase_durations = self.config.get('phase_durations', {})
        self.phase_transitions = self.config.get('phase_transitions', {})
        
        # Performance tracking
        self.total_cycles = 0
        self.successful_cycles = 0
        self.failed_cycles = 0
        self.total_signals = 0
        
        # State management
        self.state = FerrisState.ACTIVE
        self.last_update = time.time()
        self.cycle_start_time = time.time()
        
        # Metrics tracking
        self.metrics_history: List[FerrisMetrics] = []
        self.signal_history: List[FerrisSignal] = []
        
        # Initialize phase durations
        self._initialize_phase_durations()
        
        logger.info("🎡 Ferris RDE initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_history_size': 100,
            'phase_durations': {
                'tick': 300,      # 5 minutes
                'pivot': 600,     # 10 minutes
                'ascent': 1800,   # 30 minutes
                'descent': 1800   # 30 minutes
            },
            'phase_transitions': {
                'tick_to_pivot': 0.8,
                'pivot_to_ascent': 0.7,
                'ascent_to_descent': 0.6,
                'descent_to_tick': 0.9
            },
            'momentum_threshold': 0.5,
            'volatility_threshold': 0.3,
            'confidence_threshold': 0.6,
            'cycle_timeout': 7200,  # 2 hours
            'auto_reset_enabled': True
        }
    
    def _initialize_phase_durations(self):
        """Initialize phase durations from config."""
        self.phase_durations = self.config.get('phase_durations', {
            'tick': 300,
            'pivot': 600,
            'ascent': 1800,
            'descent': 1800
        })
        
        self.phase_transitions = self.config.get('phase_transitions', {
            'tick_to_pivot': 0.8,
            'pivot_to_ascent': 0.7,
            'ascent_to_descent': 0.6,
            'descent_to_tick': 0.9
        })
    
    def start_cycle(self, cycle_id: Optional[str] = None) -> FerrisCycle:
        """
        Start a new Ferris wheel cycle.
        
        Args:
            cycle_id: Optional cycle ID, auto-generated if None
            
        Returns:
            New Ferris cycle
        """
        try:
            if cycle_id is None:
                cycle_id = f"ferris_cycle_{int(time.time() * 1000)}"
            
            # End current cycle if exists
            if self.current_cycle:
                self.end_cycle()
            
            # Create new cycle
            self.current_cycle = FerrisCycle(
                cycle_id=cycle_id,
                phase=FerrisPhase.TICK,
                start_time=time.time()
            )
            
            self.current_phase = FerrisPhase.TICK
            self.cycle_start_time = time.time()
            self.state = FerrisState.ACTIVE
            
            logger.info(f"Started Ferris cycle: {cycle_id}")
            return self.current_cycle
            
        except Exception as e:
            logger.error(f"Error starting cycle: {e}")
            return self._create_default_cycle()
    
    def end_cycle(self) -> Optional[FerrisCycle]:
        """End the current cycle."""
        try:
            if not self.current_cycle:
                return None
            
            # Calculate cycle metrics
            end_time = time.time()
            duration = end_time - self.current_cycle.start_time
            performance = self._calculate_cycle_performance()
            
            # Update cycle
            self.current_cycle.end_time = end_time
            self.current_cycle.duration = duration
            self.current_cycle.performance = performance
            
            # Add to history
            self.cycle_history.append(self.current_cycle)
            if len(self.cycle_history) > self.max_history_size:
                self.cycle_history.pop(0)
            
            # Update metrics
            self.total_cycles += 1
            if performance > 0:
                self.successful_cycles += 1
            else:
                self.failed_cycles += 1
            
            # Reset state
            completed_cycle = self.current_cycle
            self.current_cycle = None
            self.current_phase = FerrisPhase.TICK
            
            logger.info(f"Ended Ferris cycle: {completed_cycle.cycle_id} (performance: {performance:.2f})")
            return completed_cycle
            
        except Exception as e:
            logger.error(f"Error ending cycle: {e}")
            return None
    
    def _create_default_cycle(self) -> FerrisCycle:
        """Create default cycle."""
        return FerrisCycle(
            cycle_id="default_cycle",
            phase=FerrisPhase.TICK,
            start_time=time.time()
        )
    
    def _calculate_cycle_performance(self) -> float:
        """Calculate cycle performance."""
        try:
            if not self.signal_history:
                return 0.0
            
            # Calculate performance based on signals
            recent_signals = [s for s in self.signal_history 
                            if s.timestamp >= self.cycle_start_time]
            
            if not recent_signals:
                return 0.0
            
            # Simple performance calculation
            buy_signals = [s for s in recent_signals if s.signal_type == "buy"]
            sell_signals = [s for s in recent_signals if s.signal_type == "sell"]
            
            total_confidence = sum(s.confidence for s in recent_signals)
            if total_confidence == 0:
                return 0.0
            
            # Weighted performance
            performance = (len(buy_signals) - len(sell_signals)) / len(recent_signals)
            performance *= (total_confidence / len(recent_signals))
            
            return np.clip(performance, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating cycle performance: {e}")
            return 0.0
    
    def update_phase(self, market_data: Dict[str, Any]) -> FerrisPhase:
        """
        Update the current phase based on market data.
        
        Args:
            market_data: Current market data
            
        Returns:
            Updated phase
        """
        try:
            if not self.current_cycle:
                return self.current_phase
            
            # Calculate phase metrics
            metrics = self._calculate_phase_metrics(market_data)
            
            # Check for phase transition
            new_phase = self._check_phase_transition(metrics)
            
            if new_phase != self.current_phase:
                self._transition_phase(new_phase)
            
            # Update metrics history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            self.last_update = time.time()
            return self.current_phase
            
        except Exception as e:
            logger.error(f"Error updating phase: {e}")
            return self.current_phase
    
    def _calculate_phase_metrics(self, market_data: Dict[str, Any]) -> FerrisMetrics:
        """Calculate phase metrics from market data."""
        try:
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            volatility = market_data.get('volatility', 0)
            
            # Calculate cycle position (0.0 to 1.0)
            cycle_duration = time.time() - self.cycle_start_time
            expected_duration = self.phase_durations.get(self.current_phase.value, 300)
            cycle_position = min(cycle_duration / expected_duration, 1.0)
            
            # Calculate momentum
            momentum = self._calculate_momentum(market_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(market_data)
            
            metrics = FerrisMetrics(
                timestamp=time.time(),
                current_phase=self.current_phase,
                cycle_position=cycle_position,
                momentum=momentum,
                volatility=volatility,
                confidence=confidence
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating phase metrics: {e}")
            return self._create_default_metrics()
    
    def _create_default_metrics(self) -> FerrisMetrics:
        """Create default metrics."""
        return FerrisMetrics(
            timestamp=time.time(),
            current_phase=self.current_phase,
            cycle_position=0.0,
            momentum=0.0,
            volatility=0.0,
            confidence=0.5
        )
    
    def _calculate_momentum(self, market_data: Dict[str, Any]) -> float:
        """Calculate momentum from market data."""
        try:
            # Simple momentum calculation
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            
            # Normalize values
            price_momentum = min(price / 100000, 1.0) if price > 0 else 0.0
            volume_momentum = min(volume / 1000000, 1.0) if volume > 0 else 0.0
            
            # Combined momentum
            momentum = (price_momentum + volume_momentum) / 2
            return np.clip(momentum, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def _calculate_confidence(self, market_data: Dict[str, Any]) -> float:
        """Calculate confidence from market data."""
        try:
            # Base confidence on volatility and data quality
            volatility = market_data.get('volatility', 0)
            price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            
            # Higher confidence for lower volatility and valid data
            volatility_confidence = 1.0 - min(volatility, 1.0)
            data_confidence = 1.0 if price > 0 and volume > 0 else 0.5
            
            confidence = (volatility_confidence + data_confidence) / 2
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _check_phase_transition(self, metrics: FerrisMetrics) -> FerrisPhase:
        """Check if phase transition is needed."""
        try:
            current_phase = metrics.current_phase
            cycle_position = metrics.cycle_position
            momentum = metrics.momentum
            confidence = metrics.confidence
            
            # Get transition threshold
            transition_key = self._get_transition_key(current_phase)
            threshold = self.phase_transitions.get(transition_key, 0.8)
            
            # Check if transition conditions are met
            if cycle_position >= threshold and confidence >= self.config['confidence_threshold']:
                return self._get_next_phase(current_phase)
            
            return current_phase
            
        except Exception as e:
            logger.error(f"Error checking phase transition: {e}")
            return self.current_phase
    
    def _get_transition_key(self, phase: FerrisPhase) -> str:
        """Get transition key for phase."""
        transitions = {
            FerrisPhase.TICK: "tick_to_pivot",
            FerrisPhase.PIVOT: "pivot_to_ascent",
            FerrisPhase.ASCENT: "ascent_to_descent",
            FerrisPhase.DESCENT: "descent_to_tick"
        }
        return transitions.get(phase, "tick_to_pivot")
    
    def _get_next_phase(self, current_phase: FerrisPhase) -> FerrisPhase:
        """Get next phase in cycle."""
        next_phases = {
            FerrisPhase.TICK: FerrisPhase.PIVOT,
            FerrisPhase.PIVOT: FerrisPhase.ASCENT,
            FerrisPhase.ASCENT: FerrisPhase.DESCENT,
            FerrisPhase.DESCENT: FerrisPhase.TICK
        }
        return next_phases.get(current_phase, FerrisPhase.TICK)
    
    def _transition_phase(self, new_phase: FerrisPhase):
        """Transition to new phase."""
        try:
            old_phase = self.current_phase
            self.current_phase = new_phase
            
            if self.current_cycle:
                self.current_cycle.phase = new_phase
            
            logger.info(f"Phase transition: {old_phase.value} -> {new_phase.value}")
            
        except Exception as e:
            logger.error(f"Error transitioning phase: {e}")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[FerrisSignal]:
        """
        Generate trading signal based on current phase.
        
        Args:
            market_data: Current market data
            
        Returns:
            Ferris trading signal
        """
        try:
            if not self.current_cycle:
                return None
            
            # Update phase
            self.update_phase(market_data)
            
            # Generate signal based on phase
            signal_type, strength = self._generate_phase_signal(market_data)
            
            if signal_type == "hold":
                return None
            
            # Calculate confidence
            confidence = self._calculate_signal_confidence(market_data)
            
            if confidence < self.config['confidence_threshold']:
                return None
            
            # Create signal
            signal = FerrisSignal(
                signal_id=f"ferris_signal_{int(time.time() * 1000)}",
                timestamp=time.time(),
                phase=self.current_phase,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                cycle_context=self.current_cycle
            )
            
            # Update signal history
            self.signal_history.append(signal)
            if len(self.signal_history) > self.max_history_size:
                self.signal_history.pop(0)
            
            self.total_signals += 1
            
            logger.info(f"Generated Ferris signal: {signal_type} (phase: {self.current_phase.value}, confidence: {confidence:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _generate_phase_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Generate signal based on current phase."""
        try:
            if self.current_phase == FerrisPhase.TICK:
                return self._generate_tick_signal(market_data)
            elif self.current_phase == FerrisPhase.PIVOT:
                return self._generate_pivot_signal(market_data)
            elif self.current_phase == FerrisPhase.ASCENT:
                return self._generate_ascent_signal(market_data)
            elif self.current_phase == FerrisPhase.DESCENT:
                return self._generate_descent_signal(market_data)
            else:
                return "hold", 0.0
                
        except Exception as e:
            logger.error(f"Error generating phase signal: {e}")
            return "hold", 0.0
    
    def _generate_tick_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Generate signal for tick phase."""
        # Tick phase: Look for small opportunities
        volatility = market_data.get('volatility', 0)
        if volatility < 0.02:  # Low volatility
            return "buy", 0.3
        return "hold", 0.0
    
    def _generate_pivot_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Generate signal for pivot phase."""
        # Pivot phase: Look for trend changes
        momentum = self._calculate_momentum(market_data)
        if momentum > 0.6:
            return "buy", 0.5
        elif momentum < 0.4:
            return "sell", 0.5
        return "hold", 0.0
    
    def _generate_ascent_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Generate signal for ascent phase."""
        # Ascent phase: Bullish momentum
        return "buy", 0.7
    
    def _generate_descent_signal(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Generate signal for descent phase."""
        # Descent phase: Bearish momentum
        return "sell", 0.7
    
    def _calculate_signal_confidence(self, market_data: Dict[str, Any]) -> float:
        """Calculate signal confidence."""
        try:
            # Base confidence on phase and market conditions
            base_confidence = 0.5
            
            # Phase-specific adjustments
            if self.current_phase == FerrisPhase.ASCENT:
                base_confidence += 0.2
            elif self.current_phase == FerrisPhase.DESCENT:
                base_confidence += 0.2
            elif self.current_phase == FerrisPhase.PIVOT:
                base_confidence += 0.1
            
            # Market condition adjustments
            volatility = market_data.get('volatility', 0)
            if volatility < 0.05:  # Low volatility
                base_confidence += 0.1
            
            return np.clip(base_confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating signal confidence: {e}")
            return 0.5
    
    def get_ferris_summary(self) -> Dict[str, Any]:
        """Get summary of Ferris RDE."""
        return {
            "current_phase": self.current_phase.value if self.current_phase else None,
            "current_cycle": self.current_cycle.cycle_id if self.current_cycle else None,
            "state": self.state.value,
            "total_cycles": self.total_cycles,
            "successful_cycles": self.successful_cycles,
            "failed_cycles": self.failed_cycles,
            "total_signals": self.total_signals,
            "cycle_history_size": len(self.cycle_history),
            "metrics_history_size": len(self.metrics_history),
            "signal_history_size": len(self.signal_history),
            "last_update": self.last_update
        }
    
    def get_recent_signals(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent Ferris signals."""
        recent_signals = self.signal_history[-count:]
        return [
            {
                "signal_id": signal.signal_id,
                "timestamp": signal.timestamp,
                "phase": signal.phase.value,
                "signal_type": signal.signal_type,
                "strength": signal.strength,
                "confidence": signal.confidence
            }
            for signal in recent_signals
        ]
    
    def get_recent_cycles(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent Ferris cycles."""
        recent_cycles = self.cycle_history[-count:]
        return [
            {
                "cycle_id": cycle.cycle_id,
                "phase": cycle.phase.value,
                "start_time": cycle.start_time,
                "end_time": cycle.end_time,
                "duration": cycle.duration,
                "performance": cycle.performance
            }
            for cycle in recent_cycles
        ] 