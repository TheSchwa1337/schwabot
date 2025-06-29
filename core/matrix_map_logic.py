#!/usr/bin/env python3
"""
Matrix Map Logic Module
=======================

Logic hash selection based on matrix similarity for Schwabot v0.05.
Provides intelligent matrix-based decision making and pattern matching.
"""

import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MatrixType(Enum):
    """Matrix type enumeration."""
    FEATURE = "feature"
    SIMILARITY = "similarity"
    CORRELATION = "correlation"
    TRANSITION = "transition"
    PATTERN = "pattern"


class LogicHashType(Enum):
    """Logic hash type enumeration."""
    STRATEGY = "strategy"
    DECISION = "decision"
    PATTERN = "pattern"
    SIGNAL = "signal"
    RULE = "rule"


@dataclass
class MatrixData:
    """Matrix data structure."""
    matrix_id: str
    matrix_type: MatrixType
    data: np.ndarray
    dimensions: Tuple[int, int]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicHash:
    """Logic hash structure."""
    hash_id: str
    hash_type: LogicHashType
    hash_value: str
    matrix_reference: str
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatrixMatch:
    """Matrix match result."""
    match_id: str
    source_matrix: str
    target_matrix: str
    similarity_score: float
    matched_hashes: List[str]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MatrixMapLogic:
    """
    Matrix Map Logic for Schwabot v0.05.
    
    Provides logic hash selection based on matrix similarity
    for intelligent decision making and pattern matching.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the matrix map logic."""
        self.config = config or self._default_config()
        
        # Matrix management
        self.matrices: Dict[str, MatrixData] = {}
        self.matrix_history: List[MatrixData] = []
        self.max_matrix_history = self.config.get('max_matrix_history', 100)
        
        # Logic hash management
        self.logic_hashes: Dict[str, LogicHash] = {}
        self.hash_history: List[LogicHash] = []
        self.max_hash_history = self.config.get('max_hash_history', 1000)
        
        # Match tracking
        self.matches: List[MatrixMatch] = []
        self.max_matches = self.config.get('max_matches', 100)
        
        # Performance tracking
        self.total_matrices = 0
        self.total_hashes = 0
        self.total_matches = 0
        self.similarity_calculations = 0
        
        # State management
        self.last_update = time.time()
        
        # Initialize default matrices
        self._initialize_default_matrices()
        
        logger.info("🧮 Matrix Map Logic initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'max_matrix_history': 100,
            'max_hash_history': 1000,
            'max_matches': 100,
            'similarity_threshold': 0.7,
            'hash_confidence_threshold': 0.6,
            'matrix_dimensions': (10, 10),
            'feature_scaling': True,
            'similarity_method': 'cosine',  # 'cosine', 'euclidean', 'correlation'
            'hash_algorithm': 'sha256',
            'auto_similarity_calculation': True
        }
    
    def _initialize_default_matrices(self):
        """Initialize default matrices."""
        # Feature matrix for market data
        feature_matrix = np.random.rand(10, 10)
        self.add_matrix("market_features", MatrixType.FEATURE, feature_matrix)
        
        # Similarity matrix
        similarity_matrix = np.eye(10)  # Identity matrix as base
        self.add_matrix("base_similarity", MatrixType.SIMILARITY, similarity_matrix)
        
        # Correlation matrix
        correlation_matrix = np.random.rand(10, 10)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1
        self.add_matrix("market_correlation", MatrixType.CORRELATION, correlation_matrix)
    
    def add_matrix(self, matrix_id: str, matrix_type: MatrixType, 
                   data: np.ndarray) -> MatrixData:
        """
        Add a new matrix.
        
        Args:
            matrix_id: Unique matrix identifier
            matrix_type: Type of matrix
            data: Matrix data as numpy array
            
        Returns:
            Created matrix data
        """
        try:
            # Validate matrix data
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            if data.ndim != 2:
                raise ValueError("Matrix data must be 2-dimensional")
            
            matrix = MatrixData(
                matrix_id=matrix_id,
                matrix_type=matrix_type,
                data=data,
                dimensions=data.shape,
                timestamp=time.time()
            )
            
            self.matrices[matrix_id] = matrix
            self.total_matrices += 1
            
            # Add to history
            self.matrix_history.append(matrix)
            if len(self.matrix_history) > self.max_matrix_history:
                self.matrix_history.pop(0)
            
            logger.info(f"Added matrix: {matrix_id} ({matrix_type.value}, {data.shape})")
            return matrix
            
        except Exception as e:
            logger.error(f"Error adding matrix {matrix_id}: {e}")
            return self._create_default_matrix()
    
    def _create_default_matrix(self) -> MatrixData:
        """Create default matrix."""
        return MatrixData(
            matrix_id="default",
            matrix_type=MatrixType.FEATURE,
            data=np.zeros((5, 5)),
            dimensions=(5, 5),
            timestamp=time.time()
        )
    
    def update_matrix(self, matrix_id: str, new_data: np.ndarray) -> bool:
        """
        Update matrix data.
        
        Args:
            matrix_id: Matrix identifier
            new_data: New matrix data
            
        Returns:
            True if update was successful
        """
        try:
            if matrix_id not in self.matrices:
                logger.error(f"Matrix {matrix_id} not found")
                return False
            
            matrix = self.matrices[matrix_id]
            
            # Validate new data
            if not isinstance(new_data, np.ndarray):
                new_data = np.array(new_data)
            
            if new_data.shape != matrix.dimensions:
                logger.warning(f"Matrix shape mismatch: expected {matrix.dimensions}, got {new_data.shape}")
                # Resize if possible
                if new_data.size == matrix.data.size:
                    new_data = new_data.reshape(matrix.dimensions)
                else:
                    return False
            
            # Update matrix
            matrix.data = new_data
            matrix.timestamp = time.time()
            
            self.last_update = time.time()
            logger.debug(f"Updated matrix: {matrix_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating matrix {matrix_id}: {e}")
            return False
    
    def create_logic_hash(self, hash_type: LogicHashType, matrix_id: str,
                         data: Any, confidence: float = 1.0) -> LogicHash:
        """
        Create a logic hash from data.
        
        Args:
            hash_type: Type of logic hash
            matrix_id: Reference matrix ID
            data: Data to hash
            confidence: Hash confidence
            
        Returns:
            Created logic hash
        """
        try:
            # Generate hash value
            hash_algorithm = self.config.get('hash_algorithm', 'sha256')
            
            if isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            elif isinstance(data, (list, tuple)):
                data_bytes = str(data).encode()
            else:
                data_bytes = str(data).encode()
            
            if hash_algorithm == 'sha256':
                hash_value = hashlib.sha256(data_bytes).hexdigest()
            elif hash_algorithm == 'md5':
                hash_value = hashlib.md5(data_bytes).hexdigest()
            else:
                hash_value = hashlib.sha256(data_bytes).hexdigest()
            
            # Create logic hash
            hash_id = f"{hash_type.value}_{hash_value[:8]}_{int(time.time() * 1000)}"
            
            logic_hash = LogicHash(
                hash_id=hash_id,
                hash_type=hash_type,
                hash_value=hash_value,
                matrix_reference=matrix_id,
                confidence=confidence,
                timestamp=time.time()
            )
            
            self.logic_hashes[hash_id] = logic_hash
            
            # Add to history
            self.hash_history.append(logic_hash)
            if len(self.hash_history) > self.max_hash_history:
                self.hash_history.pop(0)
            
            self.total_hashes += 1
            
            logger.debug(f"Created logic hash: {hash_id} ({hash_type.value})")
            return logic_hash
            
        except Exception as e:
            logger.error(f"Error creating logic hash: {e}")
            return self._create_default_hash()
    
    def _create_default_hash(self) -> LogicHash:
        """Create default logic hash."""
        return LogicHash(
            hash_id="default",
            hash_type=LogicHashType.STRATEGY,
            hash_value="0" * 64,
            matrix_reference="default",
            confidence=0.0,
            timestamp=time.time()
        )
    
    def calculate_similarity(self, matrix_id_1: str, matrix_id_2: str) -> float:
        """
        Calculate similarity between two matrices.
        
        Args:
            matrix_id_1: First matrix ID
            matrix_id_2: Second matrix ID
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            if matrix_id_1 not in self.matrices or matrix_id_2 not in self.matrices:
                logger.error(f"One or both matrices not found: {matrix_id_1}, {matrix_id_2}")
                return 0.0
            
            matrix_1 = self.matrices[matrix_id_1]
            matrix_2 = self.matrices[matrix_id_2]
            
            # Ensure matrices have same dimensions
            if matrix_1.dimensions != matrix_2.dimensions:
                # Resize to smaller dimension
                min_rows = min(matrix_1.dimensions[0], matrix_2.dimensions[0])
                min_cols = min(matrix_1.dimensions[1], matrix_2.dimensions[1])
                
                data_1 = matrix_1.data[:min_rows, :min_cols]
                data_2 = matrix_2.data[:min_rows, :min_cols]
            else:
                data_1 = matrix_1.data
                data_2 = matrix_2.data
            
            # Flatten matrices for similarity calculation
            flat_1 = data_1.flatten().reshape(1, -1)
            flat_2 = data_2.flatten().reshape(1, -1)
            
            # Apply feature scaling if enabled
            if self.config.get('feature_scaling', True):
                scaler = StandardScaler()
                flat_1 = scaler.fit_transform(flat_1)
                flat_2 = scaler.transform(flat_2)
            
            # Calculate similarity
            similarity_method = self.config.get('similarity_method', 'cosine')
            
            if similarity_method == 'cosine':
                similarity = cosine_similarity(flat_1, flat_2)[0, 0]
            elif similarity_method == 'euclidean':
                distance = np.linalg.norm(flat_1 - flat_2)
                similarity = 1.0 / (1.0 + distance)
            elif similarity_method == 'correlation':
                correlation = np.corrcoef(flat_1.flatten(), flat_2.flatten())[0, 1]
                similarity = (correlation + 1) / 2  # Convert from [-1, 1] to [0, 1]
            else:
                similarity = cosine_similarity(flat_1, flat_2)[0, 0]
            
            # Ensure similarity is in [0, 1] range
            similarity = np.clip(similarity, 0.0, 1.0)
            
            self.similarity_calculations += 1
            
            logger.debug(f"Similarity between {matrix_id_1} and {matrix_id_2}: {similarity:.3f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_matrices(self, target_matrix_id: str, 
                            threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Find matrices similar to target matrix.
        
        Args:
            target_matrix_id: Target matrix ID
            threshold: Similarity threshold (uses config default if None)
            
        Returns:
            List of (matrix_id, similarity_score) tuples
        """
        try:
            if target_matrix_id not in self.matrices:
                logger.error(f"Target matrix {target_matrix_id} not found")
                return []
            
            threshold = threshold or self.config.get('similarity_threshold', 0.7)
            similar_matrices = []
            
            for matrix_id in self.matrices:
                if matrix_id == target_matrix_id:
                    continue
                
                similarity = self.calculate_similarity(target_matrix_id, matrix_id)
                if similarity >= threshold:
                    similar_matrices.append((matrix_id, similarity))
            
            # Sort by similarity score (descending)
            similar_matrices.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Found {len(similar_matrices)} similar matrices for {target_matrix_id}")
            return similar_matrices
            
        except Exception as e:
            logger.error(f"Error finding similar matrices: {e}")
            return []
    
    def select_logic_hash(self, matrix_id: str, hash_type: Optional[LogicHashType] = None,
                         min_confidence: Optional[float] = None) -> Optional[LogicHash]:
        """
        Select logic hash based on matrix similarity.
        
        Args:
            matrix_id: Reference matrix ID
            hash_type: Optional hash type filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            Selected logic hash
        """
        try:
            if matrix_id not in self.matrices:
                logger.error(f"Matrix {matrix_id} not found")
                return None
            
            min_confidence = min_confidence or self.config.get('hash_confidence_threshold', 0.6)
            
            # Find hashes associated with similar matrices
            similar_matrices = self.find_similar_matrices(matrix_id)
            
            candidate_hashes = []
            for similar_matrix_id, similarity in similar_matrices:
                # Get hashes for this matrix
                matrix_hashes = [
                    h for h in self.logic_hashes.values()
                    if h.matrix_reference == similar_matrix_id
                ]
                
                # Apply filters
                for hash_obj in matrix_hashes:
                    if hash_type and hash_obj.hash_type != hash_type:
                        continue
                    
                    if hash_obj.confidence < min_confidence:
                        continue
                    
                    # Calculate weighted score
                    weighted_score = similarity * hash_obj.confidence
                    candidate_hashes.append((hash_obj, weighted_score))
            
            if not candidate_hashes:
                logger.warning(f"No suitable logic hashes found for matrix {matrix_id}")
                return None
            
            # Select hash with highest weighted score
            selected_hash, score = max(candidate_hashes, key=lambda x: x[1])
            
            logger.info(f"Selected logic hash: {selected_hash.hash_id} (score: {score:.3f})")
            return selected_hash
            
        except Exception as e:
            logger.error(f"Error selecting logic hash: {e}")
            return None
    
    def create_matrix_match(self, source_matrix_id: str, target_matrix_id: str,
                          matched_hashes: List[str]) -> MatrixMatch:
        """
        Create a matrix match record.
        
        Args:
            source_matrix_id: Source matrix ID
            target_matrix_id: Target matrix ID
            matched_hashes: List of matched hash IDs
            
        Returns:
            Created matrix match
        """
        try:
            similarity_score = self.calculate_similarity(source_matrix_id, target_matrix_id)
            
            match = MatrixMatch(
                match_id=f"match_{int(time.time() * 1000)}",
                source_matrix=source_matrix_id,
                target_matrix=target_matrix_id,
                similarity_score=similarity_score,
                matched_hashes=matched_hashes,
                timestamp=time.time()
            )
            
            self.matches.append(match)
            if len(self.matches) > self.max_matches:
                self.matches.pop(0)
            
            self.total_matches += 1
            
            logger.info(f"Created matrix match: {match.match_id} (similarity: {similarity_score:.3f})")
            return match
            
        except Exception as e:
            logger.error(f"Error creating matrix match: {e}")
            return self._create_default_match()
    
    def _create_default_match(self) -> MatrixMatch:
        """Create default matrix match."""
        return MatrixMatch(
            match_id="default",
            source_matrix="default",
            target_matrix="default",
            similarity_score=0.0,
            matched_hashes=[],
            timestamp=time.time()
        )
    
    def get_matrix_summary(self) -> Dict[str, Any]:
        """Get summary of matrix map logic."""
        return {
            "total_matrices": self.total_matrices,
            "total_hashes": self.total_hashes,
            "total_matches": self.total_matches,
            "similarity_calculations": self.similarity_calculations,
            "active_matrices": len(self.matrices),
            "active_hashes": len(self.logic_hashes),
            "recent_matches": len(self.matches[-10:]),  # Last 10 matches
            "last_update": self.last_update
        }
    
    def get_matrix_status(self) -> List[Dict[str, Any]]:
        """Get current status of all matrices."""
        return [
            {
                "matrix_id": matrix.matrix_id,
                "matrix_type": matrix.matrix_type.value,
                "dimensions": matrix.dimensions,
                "timestamp": matrix.timestamp
            }
            for matrix in self.matrices.values()
        ]
    
    def get_hash_status(self) -> List[Dict[str, Any]]:
        """Get current status of all logic hashes."""
        return [
            {
                "hash_id": hash_obj.hash_id,
                "hash_type": hash_obj.hash_type.value,
                "matrix_reference": hash_obj.matrix_reference,
                "confidence": hash_obj.confidence,
                "timestamp": hash_obj.timestamp
            }
            for hash_obj in self.logic_hashes.values()
        ]
    
    def get_recent_matches(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent matrix matches."""
        recent_matches = self.matches[-count:]
        return [
            {
                "match_id": match.match_id,
                "source_matrix": match.source_matrix,
                "target_matrix": match.target_matrix,
                "similarity_score": match.similarity_score,
                "matched_hashes_count": len(match.matched_hashes),
                "timestamp": match.timestamp
            }
            for match in recent_matches
        ]
    
    def export_matrix_data(self, filepath: str) -> bool:
        """
        Export matrix data to file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export was successful
        """
        try:
            import json
            
            data = {
                "export_timestamp": time.time(),
                "matrix_summary": self.get_matrix_summary(),
                "matrix_status": self.get_matrix_status(),
                "hash_status": self.get_hash_status(),
                "recent_matches": self.get_recent_matches(50)
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Exported matrix data to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting matrix data: {e}")
            return False 