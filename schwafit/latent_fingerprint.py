"""
Latent Fingerprint Module

Stores and manages fingerprints of profitable strategy patterns, including
Oracle signatures and execution outcomes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from schwabot.utils.oracle_guard import safe_log_fingerprint

logger = logging.getLogger("latent_fingerprint")

class LatentFingerprint:
    """
    Manages storage and retrieval of strategy fingerprints with Oracle signatures.
    """
    
    def __init__(self, storage_path: str = "data/fingerprints"):
        """
        Initialize the fingerprint storage.
        
        Args:
            storage_path: Path to store fingerprint data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        self._load_existing()
        
    def _load_existing(self) -> None:
        """Load existing fingerprints from storage."""
        try:
            fingerprint_file = self.storage_path / "fingerprints.json"
            if fingerprint_file.exists():
                with open(fingerprint_file, 'r') as f:
                    self.entries = json.load(f)
                logger.info(f"Loaded {len(self.entries)} existing fingerprints")
        except Exception as e:
            logger.error(f"Failed to load existing fingerprints: {e}")
            self.entries = []
    
    def _save(self) -> None:
        """Save current fingerprints to storage."""
        try:
            fingerprint_file = self.storage_path / "fingerprints.json"
            with open(fingerprint_file, 'w') as f:
                json.dump(self.entries, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save fingerprints: {e}")
    
    def log(self, strategy_name: str, oracle_signature: str, profit_delta: float) -> None:
        """
        Log a new strategy fingerprint.
        
        Args:
            strategy_name: Name of the strategy
            oracle_signature: Oracle signature string
            profit_delta: Profit delta value
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "oracle": oracle_signature,
            "profit": profit_delta
        }
        
        self.entries.append(entry)
        self._save()
        logger.info(f"Logged fingerprint for strategy: {strategy_name}")
    
    def get_similar_fingerprints(self, oracle_signature: str, 
                               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar fingerprints based on Oracle signature.
        
        Args:
            oracle_signature: Oracle signature to compare against
            top_k: Number of similar fingerprints to return
            
        Returns:
            List of similar fingerprints
        """
        # TODO: Implement proper similarity matching
        # For now, return most recent entries
        return sorted(
            self.entries,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:top_k]
    
    def get_profitable_patterns(self, min_profit: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve fingerprints of profitable patterns.
        
        Args:
            min_profit: Minimum profit threshold
            
        Returns:
            List of profitable fingerprints
        """
        return [
            entry for entry in self.entries
            if entry["profit"] >= min_profit
        ] 