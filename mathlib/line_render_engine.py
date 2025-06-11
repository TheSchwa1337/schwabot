"""
Line Render Engine

Handles the rendering and processing of matrix paths for visualization.
Provides functionality for calculating scores and processing ticks.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from core.config import ConfigLoader, ConfigError

@dataclass
class LineState:
    """Represents the state of a line in the render engine."""
    path: List[Tuple[float, float]]
    score: float
    active: bool
    last_update: float

class LineRenderEngine:
    """Engine for rendering and processing matrix paths."""
    
    def __init__(self, config_path: str = "config/line_render.yaml"):
        """Initialize the line render engine.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_loader = ConfigLoader()
        try:
            self.config = self.config_loader.load_yaml(config_path)
        except ConfigError as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            self.config = self.config_loader.load_yaml("config/defaults.yaml")
        
        self.threshold = self.config.get("threshold", 0.15)
        self.max_path_length = self.config.get("max_path_length", 100)
        self.active_lines: Dict[str, LineState] = {}
        
    def load_matrix_paths(self, paths: Dict[str, List[Tuple[float, float]]]) -> None:
        """Load matrix paths into the render engine.
        
        Args:
            paths: Dictionary mapping line IDs to their paths
        """
        for line_id, path in paths.items():
            if len(path) > self.max_path_length:
                print(f"Warning: Path for {line_id} exceeds max length, truncating")
                path = path[-self.max_path_length:]
            
            self.active_lines[line_id] = LineState(
                path=path,
                score=0.0,
                active=True,
                last_update=0.0
            )
    
    def calculate_score(self, line_id: str) -> float:
        """Calculate the score for a given line.
        
        Args:
            line_id: ID of the line to calculate score for
            
        Returns:
            float: Calculated score
        """
        if line_id not in self.active_lines:
            raise ValueError(f"Line {line_id} not found")
            
        line = self.active_lines[line_id]
        if not line.active:
            return 0.0
            
        # Calculate score based on path characteristics
        path = np.array(line.path)
        if len(path) < 2:
            return 0.0
            
        # Calculate velocity and acceleration
        velocities = np.diff(path, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Score based on smoothness and consistency
        velocity_magnitude = np.linalg.norm(velocities, axis=1)
        acceleration_magnitude = np.linalg.norm(accelerations, axis=1)
        
        smoothness = 1.0 / (1.0 + np.mean(acceleration_magnitude))
        consistency = 1.0 / (1.0 + np.std(velocity_magnitude))
        
        return smoothness * consistency
    
    def process_tick(self, tick_data: Dict) -> Dict[str, float]:
        """Process a new tick of data.
        
        Args:
            tick_data: Dictionary containing tick data
            
        Returns:
            Dict[str, float]: Updated scores for each line
        """
        scores = {}
        for line_id, line in self.active_lines.items():
            try:
                if not line.active:
                    continue
                    
                # Update line state based on tick data
                if line_id in tick_data:
                    line.last_update = tick_data[line_id].get("timestamp", 0.0)
                    
                # Calculate new score
                score = self.calculate_score(line_id)
                line.score = score
                scores[line_id] = score
                
                # Check if line should be deactivated
                if score < self.threshold:
                    line.active = False
                    print(f"Line {line_id} deactivated due to low score")
                    
            except Exception as e:
                print(f"Error processing tick for line {line_id}: {e}")
                continue
                
        return scores
    
    def get_active_lines(self) -> Dict[str, LineState]:
        """Get all currently active lines.
        
        Returns:
            Dict[str, LineState]: Dictionary of active lines
        """
        return {k: v for k, v in self.active_lines.items() if v.active}
    
    def reset(self) -> None:
        """Reset the render engine state."""
        self.active_lines.clear() 