"""
Tests for ScalarLaws class
"""

import unittest
from datetime import datetime, timedelta
import time
from schwabot.core.scalar_laws import ScalarLaws, ChunkProfile

class TestScalarLaws(unittest.TestCase):
    """Test cases for ScalarLaws"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scalar_laws = ScalarLaws()
        
    def test_initialization(self):
        """Test ScalarLaws initialization"""
        self.assertEqual(len(self.scalar_laws.chunk_profiles), 5)
        self.assertEqual(self.scalar_laws.tick_window, 225.0)
        self.assertIsInstance(self.scalar_laws.current_tick_start, datetime)
        self.assertEqual(len(self.scalar_laws.chunk_usage_log), 0)
        
    def test_chunk_profiles(self):
        """Test chunk profile characteristics"""
        profiles = self.scalar_laws.chunk_profiles
        
        # Test 128 chunk profile
        self.assertEqual(profiles[128].size, 128)
        self.assertEqual(profiles[128].parse_time, 1.55)
        self.assertEqual(profiles[128].max_runs_per_tick, 145)
        self.assertEqual(profiles[128].confidence_threshold, 0.65)
        
        # Test 2048 chunk profile
        self.assertEqual(profiles[2048].size, 2048)
        self.assertEqual(profiles[2048].parse_time, 1.72)
        self.assertEqual(profiles[2048].max_runs_per_tick, 130)
        self.assertEqual(profiles[2048].confidence_threshold, 0.85)
        
    def test_allocate_chunk_bandwidth(self):
        """Test chunk bandwidth allocation"""
        # Test high confidence scenario
        chunks = self.scalar_laws.allocate_chunk_bandwidth(
            profit_likelihood=0.9,
            entropy_score=0.8,
            memkey_confidence=0.85
        )
        self.assertTrue(all(c == 2048 for c in chunks))
        
        # Test medium confidence scenario
        chunks = self.scalar_laws.allocate_chunk_bandwidth(
            profit_likelihood=0.7,
            entropy_score=0.6,
            memkey_confidence=0.65
        )
        self.assertTrue(all(c == 512 for c in chunks))
        
        # Test low confidence scenario
        chunks = self.scalar_laws.allocate_chunk_bandwidth(
            profit_likelihood=0.4,
            entropy_score=0.3,
            memkey_confidence=0.5
        )
        self.assertTrue(all(c == 128 for c in chunks))
        
    def test_chunk_metrics(self):
        """Test chunk usage metrics"""
        # Generate some allocations
        self.scalar_laws.allocate_chunk_bandwidth(0.9, 0.8, 0.85)  # 2048
        self.scalar_laws.allocate_chunk_bandwidth(0.7, 0.6, 0.65)  # 512
        self.scalar_laws.allocate_chunk_bandwidth(0.4, 0.3, 0.5)   # 128
        
        metrics = self.scalar_laws.get_chunk_metrics()
        
        self.assertEqual(metrics["total_allocations"], 3)
        self.assertIn("chunk_distribution", metrics)
        self.assertIn("avg_confidence", metrics)
        self.assertIn("avg_profit_likelihood", metrics)
        
    def test_tick_window_management(self):
        """Test tick window timer management"""
        # Test initial remaining time
        remaining = self.scalar_laws.get_remaining_tick_time()
        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, 225.0)
        
        # Test reset
        self.scalar_laws.reset_tick_window()
        new_remaining = self.scalar_laws.get_remaining_tick_time()
        self.assertGreater(new_remaining, 0)
        self.assertLessEqual(new_remaining, 225.0)
        
    def test_optimal_chunk_selection(self):
        """Test optimal chunk size selection logic"""
        # Test boundary conditions
        self.assertEqual(
            self.scalar_laws._get_optimal_chunk_size(0.9),
            2048
        )
        self.assertEqual(
            self.scalar_laws._get_optimal_chunk_size(0.75),
            512
        )
        self.assertEqual(
            self.scalar_laws._get_optimal_chunk_size(0.5),
            128
        )
        
if __name__ == '__main__':
    unittest.main() 