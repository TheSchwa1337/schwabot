#!/usr/bin/env python3
"""
Glyph VM (Virtual Machine) Module
=================================

Glyph drift visualizer/debug terminal output for Schwabot v0.05.
Provides real-time visualization of system state and drift patterns.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
import os
import sys

logger = logging.getLogger(__name__)


class GlyphType(Enum):
    """Glyph type enumeration."""
    SYSTEM = "system"
    TRADING = "trading"
    STRATEGY = "strategy"
    PERFORMANCE = "performance"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class GlyphState(Enum):
    """Glyph state enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    WARNING = "warning"
    ERROR = "error"
    DRIFTING = "drifting"
    STABLE = "stable"


@dataclass
class GlyphData:
    """Glyph data structure."""
    glyph_id: str
    glyph_type: GlyphType
    state: GlyphState
    value: float
    timestamp: float
    drift_factor: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlyphPattern:
    """Glyph pattern for visualization."""
    pattern_id: str
    pattern_type: str
    glyphs: List[GlyphData]
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class GlyphVM:
    """
    Glyph VM (Virtual Machine) for Schwabot v0.05.
    
    Provides glyph drift visualizer/debug terminal output
    for real-time system state visualization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Glyph VM."""
        self.config = config or self._default_config()
        
        # Glyph management
        self.glyphs: Dict[str, GlyphData] = {}
        self.glyph_history: List[GlyphData] = []
        self.max_history_size = self.config.get('max_history_size', 1000)
        
        # Pattern recognition
        self.patterns: List[GlyphPattern] = []
        self.max_patterns = self.config.get('max_patterns', 100)
        
        # Visualization settings
        self.display_enabled = self.config.get('display_enabled', True)
        self.terminal_width = self.config.get('terminal_width', 80)
        self.update_interval = self.config.get('update_interval', 1.0)
        
        # Performance tracking
        self.total_glyphs = 0
        self.total_patterns = 0
        self.drift_detections = 0
        
        # State management
        self.last_update = time.time()
        self.last_display_update = time.time()
        
        # Initialize glyphs
        self._initialize_default_glyphs()
        
        logger.info("🔮 Glyph VM initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_history_size': 1000,
            'max_patterns': 100,
            'display_enabled': True,
            'terminal_width': 80,
            'update_interval': 1.0,
            'drift_threshold': 0.1,
            'pattern_confidence_threshold': 0.7,
            'glyph_symbols': {
                'system': '⚙️',
                'trading': '💰',
                'strategy': '🎯',
                'performance': '📊',
                'error': '❌',
                'warning': '⚠️',
                'info': 'ℹ️'
            },
            'state_colors': {
                'active': 'green',
                'inactive': 'gray',
                'warning': 'yellow',
                'error': 'red',
                'drifting': 'cyan',
                'stable': 'blue'
            }
        }
    
    def _initialize_default_glyphs(self):
        """Initialize default glyphs."""
        default_glyphs = [
            ("system_health", GlyphType.SYSTEM, GlyphState.ACTIVE, 1.0),
            ("trading_performance", GlyphType.TRADING, GlyphState.STABLE, 0.5),
            ("strategy_confidence", GlyphType.STRATEGY, GlyphState.ACTIVE, 0.7),
            ("profit_margin", GlyphType.PERFORMANCE, GlyphState.STABLE, 0.0),
            ("error_rate", GlyphType.ERROR, GlyphState.INACTIVE, 0.0),
            ("warning_level", GlyphType.WARNING, GlyphState.INACTIVE, 0.0)
        ]
        
        for glyph_id, glyph_type, state, value in default_glyphs:
            self.add_glyph(glyph_id, glyph_type, state, value)
    
    def add_glyph(self, glyph_id: str, glyph_type: GlyphType, 
                  state: GlyphState, value: float) -> GlyphData:
        """
        Add a new glyph.
        
        Args:
            glyph_id: Unique glyph identifier
            glyph_type: Type of glyph
            state: Current state
            value: Current value
            
        Returns:
            Created glyph data
        """
        try:
            glyph = GlyphData(
                glyph_id=glyph_id,
                glyph_type=glyph_type,
                state=state,
                value=value,
                timestamp=time.time()
            )
            
            self.glyphs[glyph_id] = glyph
            self.total_glyphs += 1
            
            logger.debug(f"Added glyph: {glyph_id} ({glyph_type.value}, {state.value})")
            return glyph
            
        except Exception as e:
            logger.error(f"Error adding glyph {glyph_id}: {e}")
            return self._create_default_glyph()
    
    def _create_default_glyph(self) -> GlyphData:
        """Create default glyph."""
        return GlyphData(
            glyph_id="default",
            glyph_type=GlyphType.INFO,
            state=GlyphState.INACTIVE,
            value=0.0,
            timestamp=time.time()
        )
    
    def update_glyph(self, glyph_id: str, value: float, 
                    state: Optional[GlyphState] = None) -> bool:
        """
        Update glyph value and state.
        
        Args:
            glyph_id: Glyph identifier
            value: New value
            state: New state (optional)
            
        Returns:
            True if update was successful
        """
        try:
            if glyph_id not in self.glyphs:
                logger.error(f"Glyph {glyph_id} not found")
                return False
            
            glyph = self.glyphs[glyph_id]
            old_value = glyph.value
            
            # Calculate drift factor
            drift_factor = abs(value - old_value)
            glyph.drift_factor = drift_factor
            
            # Update glyph
            glyph.value = value
            glyph.timestamp = time.time()
            
            if state:
                glyph.state = state
            
            # Check for drift
            if drift_factor > self.config.get('drift_threshold', 0.1):
                glyph.state = GlyphState.DRIFTING
                self.drift_detections += 1
                logger.debug(f"Glyph drift detected: {glyph_id} (drift: {drift_factor:.3f})")
            
            # Add to history
            self.glyph_history.append(glyph)
            if len(self.glyph_history) > self.max_history_size:
                self.glyph_history.pop(0)
            
            self.last_update = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error updating glyph {glyph_id}: {e}")
            return False
    
    def detect_patterns(self) -> List[GlyphPattern]:
        """
        Detect patterns in glyph data.
        
        Returns:
            List of detected patterns
        """
        try:
            patterns = []
            
            # Simple pattern detection (can be enhanced with ML)
            recent_glyphs = self.glyph_history[-50:]  # Last 50 glyphs
            
            if len(recent_glyphs) < 10:
                return patterns
            
            # Detect drift patterns
            drift_pattern = self._detect_drift_pattern(recent_glyphs)
            if drift_pattern:
                patterns.append(drift_pattern)
            
            # Detect state transition patterns
            transition_pattern = self._detect_state_transition_pattern(recent_glyphs)
            if transition_pattern:
                patterns.append(transition_pattern)
            
            # Detect value correlation patterns
            correlation_pattern = self._detect_correlation_pattern(recent_glyphs)
            if correlation_pattern:
                patterns.append(correlation_pattern)
            
            # Update pattern history
            for pattern in patterns:
                self.patterns.append(pattern)
                if len(self.patterns) > self.max_patterns:
                    self.patterns.pop(0)
            
            self.total_patterns += len(patterns)
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []
    
    def _detect_drift_pattern(self, glyphs: List[GlyphData]) -> Optional[GlyphPattern]:
        """Detect drift patterns in glyphs."""
        try:
            drifting_glyphs = [g for g in glyphs if g.state == GlyphState.DRIFTING]
            
            if len(drifting_glyphs) >= 3:
                confidence = min(len(drifting_glyphs) / 10.0, 1.0)
                
                return GlyphPattern(
                    pattern_id=f"drift_pattern_{int(time.time() * 1000)}",
                    pattern_type="drift",
                    glyphs=drifting_glyphs,
                    confidence=confidence,
                    timestamp=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting drift pattern: {e}")
            return None
    
    def _detect_state_transition_pattern(self, glyphs: List[GlyphData]) -> Optional[GlyphPattern]:
        """Detect state transition patterns."""
        try:
            transitions = []
            for i in range(1, len(glyphs)):
                if glyphs[i].state != glyphs[i-1].state:
                    transitions.append(glyphs[i])
            
            if len(transitions) >= 2:
                confidence = min(len(transitions) / 5.0, 1.0)
                
                return GlyphPattern(
                    pattern_id=f"transition_pattern_{int(time.time() * 1000)}",
                    pattern_type="state_transition",
                    glyphs=transitions,
                    confidence=confidence,
                    timestamp=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting state transition pattern: {e}")
            return None
    
    def _detect_correlation_pattern(self, glyphs: List[GlyphData]) -> Optional[GlyphPattern]:
        """Detect value correlation patterns."""
        try:
            # Simple correlation detection
            values = [g.value for g in glyphs]
            if len(values) < 5:
                return None
            
            # Check for trend
            trend = np.polyfit(range(len(values)), values, 1)[0]
            confidence = min(abs(trend) * 10, 1.0)
            
            if confidence > 0.3:
                return GlyphPattern(
                    pattern_id=f"correlation_pattern_{int(time.time() * 1000)}",
                    pattern_type="value_correlation",
                    glyphs=glyphs[-5:],  # Last 5 glyphs
                    confidence=confidence,
                    timestamp=time.time()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting correlation pattern: {e}")
            return None
    
    def render_display(self) -> str:
        """
        Render the glyph display for terminal output.
        
        Returns:
            Formatted display string
        """
        try:
            if not self.display_enabled:
                return ""
            
            # Check if it's time to update display
            if time.time() - self.last_display_update < self.update_interval:
                return ""
            
            self.last_display_update = time.time()
            
            # Build display
            display_lines = []
            
            # Header
            display_lines.append("=" * self.terminal_width)
            display_lines.append("🔮 GLYPH VM - SCHWABOT v0.05")
            display_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            display_lines.append("=" * self.terminal_width)
            
            # Glyph status
            display_lines.append("📊 GLYPH STATUS:")
            display_lines.append("-" * 40)
            
            for glyph_id, glyph in self.glyphs.items():
                symbol = self.config['glyph_symbols'].get(glyph.glyph_type.value, '❓')
                color = self.config['state_colors'].get(glyph.state.value, 'white')
                
                # Create status bar
                bar_length = 20
                filled_length = int(glyph.value * bar_length)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                line = f"{symbol} {glyph_id:<20} [{bar}] {glyph.value:.3f} ({glyph.state.value})"
                if glyph.drift_factor > 0:
                    line += f" [drift: {glyph.drift_factor:.3f}]"
                
                display_lines.append(line)
            
            # Pattern summary
            display_lines.append("")
            display_lines.append("🎯 PATTERN SUMMARY:")
            display_lines.append("-" * 40)
            
            recent_patterns = self.patterns[-5:]  # Last 5 patterns
            if recent_patterns:
                for pattern in recent_patterns:
                    line = f"• {pattern.pattern_type} (confidence: {pattern.confidence:.2f})"
                    display_lines.append(line)
            else:
                display_lines.append("No recent patterns detected")
            
            # System metrics
            display_lines.append("")
            display_lines.append("📈 SYSTEM METRICS:")
            display_lines.append("-" * 40)
            
            metrics = [
                f"Total Glyphs: {self.total_glyphs}",
                f"Total Patterns: {self.total_patterns}",
                f"Drift Detections: {self.drift_detections}",
                f"History Size: {len(self.glyph_history)}",
                f"Active Glyphs: {len([g for g in self.glyphs.values() if g.state == GlyphState.ACTIVE])}"
            ]
            
            for metric in metrics:
                display_lines.append(metric)
            
            # Footer
            display_lines.append("")
            display_lines.append("=" * self.terminal_width)
            
            return "\n".join(display_lines)
            
        except Exception as e:
            logger.error(f"Error rendering display: {e}")
            return f"Error rendering display: {e}"
    
    def print_display(self):
        """Print the glyph display to terminal."""
        try:
            display = self.render_display()
            if display:
                # Clear screen (works on most terminals)
                os.system('cls' if os.name == 'nt' else 'clear')
                print(display)
                
        except Exception as e:
            logger.error(f"Error printing display: {e}")
    
    def get_glyph_summary(self) -> Dict[str, Any]:
        """Get summary of glyph VM."""
        return {
            "total_glyphs": self.total_glyphs,
            "total_patterns": self.total_patterns,
            "drift_detections": self.drift_detections,
            "active_glyphs": len([g for g in self.glyphs.values() if g.state == GlyphState.ACTIVE]),
            "drifting_glyphs": len([g for g in self.glyphs.values() if g.state == GlyphState.DRIFTING]),
            "history_size": len(self.glyph_history),
            "recent_patterns": len(self.patterns[-10:]),  # Last 10 patterns
            "last_update": self.last_update
        }
    
    def get_glyph_status(self) -> List[Dict[str, Any]]:
        """Get current status of all glyphs."""
        return [
            {
                "glyph_id": glyph.glyph_id,
                "glyph_type": glyph.glyph_type.value,
                "state": glyph.state.value,
                "value": glyph.value,
                "drift_factor": glyph.drift_factor,
                "timestamp": glyph.timestamp
            }
            for glyph in self.glyphs.values()
        ]
    
    def get_recent_patterns(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent detected patterns."""
        recent_patterns = self.patterns[-count:]
        return [
            {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "timestamp": pattern.timestamp,
                "glyphs_count": len(pattern.glyphs)
            }
            for pattern in recent_patterns
        ]
    
    def export_glyph_data(self, filepath: str) -> bool:
        """
        Export glyph data to file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export was successful
        """
        try:
            import json
            
            data = {
                "export_timestamp": time.time(),
                "glyph_summary": self.get_glyph_summary(),
                "glyph_status": self.get_glyph_status(),
                "recent_patterns": self.get_recent_patterns(50)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported glyph data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting glyph data: {e}")
            return False
    
    def start_display_loop(self, interval: Optional[float] = None):
        """
        Start continuous display loop.
        
        Args:
            interval: Update interval in seconds (uses config default if None)
        """
        try:
            update_interval = interval or self.update_interval
            
            logger.info(f"Starting glyph display loop (interval: {update_interval}s)")
            
            while True:
                try:
                    # Update glyphs (simulate some changes)
                    self._simulate_glyph_updates()
                    
                    # Detect patterns
                    self.detect_patterns()
                    
                    # Print display
                    self.print_display()
                    
                    # Wait for next update
                    time.sleep(update_interval)
                    
                except KeyboardInterrupt:
                    logger.info("Display loop interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in display loop: {e}")
                    time.sleep(update_interval)
                    
        except Exception as e:
            logger.error(f"Error starting display loop: {e}")
    
    def _simulate_glyph_updates(self):
        """Simulate glyph updates for demo purposes."""
        try:
            # Simulate some random changes
            for glyph_id in self.glyphs:
                if np.random.random() < 0.3:  # 30% chance of update
                    new_value = np.clip(self.glyphs[glyph_id].value + np.random.normal(0, 0.1), 0, 1)
                    self.update_glyph(glyph_id, new_value)
                    
        except Exception as e:
            logger.error(f"Error simulating glyph updates: {e}") 