"""
Tests for AlephUnitizer class
"""

import unittest
import time
import numpy as np
from pathlib import Path
import tempfile
import os

from schwabot.core.aleph_unitizer import AlephUnitizer, UnitizerState

class TestAlephUnitizer(unittest.TestCase):
    """Test cases for AlephUnitizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.unitizer = AlephUnitizer(
            cache_size=100,
            max_depth=2,
            min_price_change=0.001,
            parallel_processing=True
        )
        
    def test_initialization(self):
        """Test AlephUnitizer initialization"""
        self.assertEqual(self.unitizer.cache_size, 100)
        self.assertEqual(self.unitizer.max_depth, 2)
        self.assertEqual(self.unitizer.min_price_change, 0.001)
        self.assertTrue(self.unitizer.parallel_processing)
        self.assertIsNone(self.unitizer.last_price)
        self.assertIsNone(self.unitizer.last_hash_tree)
        
    def test_generate_root_hash(self):
        """Test root hash generation"""
        price = 50000.0
        timestamp = time.time()
        hash_value = self.unitizer.generate_root_hash(price, timestamp)
        
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)  # SHA256 produces 64 hex chars
        
    def test_unitize_price(self):
        """Test price unitization"""
        price = 50000.0
        result = self.unitizer.unitize_price(price)
        
        # Check result structure
        self.assertIn("root_hash", result)
        self.assertIn("tree", result)
        self.assertIn("entropy_score", result)
        self.assertIn("pattern_vector", result)
        self.assertIn("generation_time", result)
        self.assertIn("tree_depth", result)
        
        # Check types
        self.assertIsInstance(result["root_hash"], str)
        self.assertIsInstance(result["tree"], dict)
        self.assertIsInstance(result["entropy_score"], float)
        self.assertIsInstance(result["pattern_vector"], list)
        self.assertIsInstance(result["generation_time"], float)
        self.assertIsInstance(result["tree_depth"], int)
        
        # Check pattern vector
        self.assertEqual(len(result["pattern_vector"]), 8)
        
    def test_cache_functionality(self):
        """Test caching mechanism"""
        price = 50000.0
        
        # First call should miss cache
        result1 = self.unitizer.unitize_price(price)
        self.assertEqual(self.unitizer.state.cache_hits, 0)
        self.assertEqual(self.unitizer.state.cache_misses, 1)
        
        # Second call should hit cache
        result2 = self.unitizer.unitize_price(price)
        self.assertEqual(self.unitizer.state.cache_hits, 1)
        self.assertEqual(self.unitizer.state.cache_misses, 1)
        
        # Results should be identical
        self.assertEqual(result1, result2)
        
    def test_tree_depth_adaptation(self):
        """Test adaptive tree depth based on price change"""
        # First price
        price1 = 50000.0
        result1 = self.unitizer.unitize_price(price1)
        self.assertEqual(result1["tree_depth"], 2)  # Full depth for first price
        
        # Small price change
        price2 = 50000.1  # 0.0002% change
        result2 = self.unitizer.unitize_price(price2)
        self.assertEqual(result2["tree_depth"], 1)  # Reduced depth for small change
        
        # Large price change
        price3 = 51000.0  # 2% change
        result3 = self.unitizer.unitize_price(price3)
        self.assertEqual(result3["tree_depth"], 2)  # Full depth for large change
        
    def test_entropy_calculation(self):
        """Test entropy score calculation"""
        price = 50000.0
        result = self.unitizer.unitize_price(price)
        
        # Entropy should be between 0 and 1
        self.assertGreaterEqual(result["entropy_score"], 0.0)
        self.assertLessEqual(result["entropy_score"], 1.0)
        
    def test_state_management(self):
        """Test state saving and loading"""
        # Generate some state
        price = 50000.0
        self.unitizer.unitize_price(price)
        
        # Save state
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            self.unitizer.save_state(tmp.name)
            
            # Create new unitizer and load state
            new_unitizer = AlephUnitizer()
            new_unitizer.load_state(tmp.name)
            
            # Compare states
            self.assertEqual(
                self.unitizer.state.root_hash,
                new_unitizer.state.root_hash
            )
            self.assertEqual(
                self.unitizer.state.entropy_score,
                new_unitizer.state.entropy_score
            )
            self.assertEqual(
                self.unitizer.state.pattern_vector,
                new_unitizer.state.pattern_vector
            )
            
        # Cleanup
        os.unlink(tmp.name)
        
    def test_performance(self):
        """Test performance with multiple prices"""
        prices = np.linspace(50000.0, 51000.0, 10)
        start_time = time.time()
        
        for price in prices:
            result = self.unitizer.unitize_price(price)
            
        total_time = time.time() - start_time
        
        # Average time per price should be reasonable
        avg_time = total_time / len(prices)
        self.assertLess(avg_time, 1.0)  # Less than 1 second per price
        
    def test_parallel_processing(self):
        """Test parallel processing functionality"""
        # Create unitizer with parallel processing
        parallel_unitizer = AlephUnitizer(parallel_processing=True)
        
        # Create unitizer without parallel processing
        serial_unitizer = AlephUnitizer(parallel_processing=False)
        
        # Compare performance
        price = 50000.0
        start_parallel = time.time()
        parallel_result = parallel_unitizer.unitize_price(price)
        parallel_time = time.time() - start_parallel
        
        start_serial = time.time()
        serial_result = serial_unitizer.unitize_price(price)
        serial_time = time.time() - start_serial
        
        # Results should be identical
        self.assertEqual(parallel_result["root_hash"], serial_result["root_hash"])
        
        # Parallel should be faster (though this might not always be true due to overhead)
        self.assertLess(parallel_time, serial_time * 1.5)
        
if __name__ == '__main__':
    unittest.main() 