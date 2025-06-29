#!/usr/bin/env python3
""""""
Aleph Overlay Mapper Module
===========================

Implements NDArray hash similarity projection using cosine similarity
and phase alignment for Schwabot v0.5.
""""""

import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class OverlayType(Enum):
    """Overlay type enumeration."""
    HASH_SIMILARITY = "hash_similarity"
    PHASE_ALIGNMENT = "phase_alignment"
    COSINE_PROJECTION = "cosine_projection"
    MATRIX_MAPPING = "matrix_mapping"


@dataclass
class OverlayMap:
    """Overlay map data."""
    map_id: str
    overlay_type: OverlayType
    hash_signal: str
    similarity_matrix: NDArray
    phase_alignment: NDArray
    confidence_score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HashVector:
    """Hash vector data."""
    vector_id: str
    hash_value: str
    vector_representation: NDArray
    phase_components: NDArray
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlephOverlayMapper:
    """"""
    Aleph Overlay Mapper for Schwabot v0.5.

    Implements NDArray hash similarity projection using cosine similarity
    and phase alignment.
    """"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the aleph overlay mapper."""
        self.config = config or self._default_config()

        # Overlay tracking
        self.overlay_maps: List[OverlayMap] = []
        self.max_map_history = self.config.get('max_map_history', 100)

        # Hash vectors
        self.hash_vectors: List[HashVector] = []
        self.max_vector_history = self.config.get('max_vector_history', 100)

        # Similarity parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.phase_alignment_threshold = self.config.get('phase_alignment_threshold', 0.7)
        self.vector_dimension = self.config.get('vector_dimension', 256)

        # Performance tracking
        self.total_mappings = 0
        self.successful_projections = 0

        logger.info("🔮 Aleph Overlay Mapper initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {}
            'max_map_history': 100,
                'max_vector_history': 100,
                    'similarity_threshold': 0.8,
                    'phase_alignment_threshold': 0.7,
                    'vector_dimension': 256,
                    'hash_length': 64,
                    'cosine_similarity_weight': 0.6,
                    'phase_alignment_weight': 0.4,
                    'projection_resolution': 128
}
    def map_hash_to_overlay(self, hash_signal: str) -> OverlayMap:
        """"""
        Map hash signal to overlay using similarity projection.

        Args:
            hash_signal: Input hash signal string

        Returns:
            Overlay map
        """"""
        try:
            # Validate hash signal
            if not hash_signal or len(hash_signal) < 8:
                logger.warning(f"Invalid hash signal: {hash_signal}")
                return self._create_default_overlay_map()

            # Generate hash vector
            hash_vector = self._generate_hash_vector(hash_signal)

            # Calculate similarity matrix
            similarity_matrix = self._calculate_similarity_matrix(hash_vector)

            # Calculate phase alignment
            phase_alignment = self._calculate_phase_alignment(hash_vector)

            # Calculate confidence score
            confidence_score = self._calculate_overlay_confidence(similarity_matrix, phase_alignment)

            # Determine overlay type
            overlay_type = self._determine_overlay_type(similarity_matrix, phase_alignment)

            # Create overlay map
            overlay_map = OverlayMap()
                map_id=f"overlay_{int(time.time() * 1000)}",
                    overlay_type=overlay_type,
                        hash_signal=hash_signal,
                        similarity_matrix=similarity_matrix,
                        phase_alignment=phase_alignment,
                        confidence_score=confidence_score,
                        timestamp=time.time()
            )

            # Add to history
            self.overlay_maps.append(overlay_map)
            if len(self.overlay_maps) > self.max_map_history:
                self.overlay_maps.pop(0)

            # Store hash vector
            self.hash_vectors.append(hash_vector)
            if len(self.hash_vectors) > self.max_vector_history:
                self.hash_vectors.pop(0)

            self.total_mappings += 1
            if confidence_score > self.similarity_threshold:
                self.successful_projections += 1

            logger.debug(f"Mapped hash to overlay: {overlay_type.value} (confidence: {confidence_score:.3f})")

            return overlay_map

        except Exception as e:
            logger.error(f"Error mapping hash to overlay: {e}")
            return self._create_default_overlay_map()

    def calculate_overlay_confidence(self, matrix: NDArray) -> float:
        """"""
        Calculate overlay confidence from similarity matrix.

        Args:
            matrix: Similarity matrix

        Returns:
            Confidence score between 0 and 1
        """"""
        try:
            if matrix.size == 0:
                return 0.0

            # Calculate matrix properties
            matrix_mean = np.mean(matrix)
            matrix_std = np.std(matrix)
            matrix_max = np.max(matrix)

            # Calculate confidence based on matrix characteristics
            mean_confidence = matrix_mean
            std_confidence = 1.0 - min(1.0, matrix_std)
            max_confidence = matrix_max

            # Weighted combination
            confidence = (0.4 * mean_confidence + 0.3 * std_confidence + 0.3 * max_confidence)

            return min(1.0, max(0.0, confidence))

        except Exception as e:
            logger.error(f"Error calculating overlay confidence: {e}")
            return 0.0

    def _generate_hash_vector(self, hash_signal: str) -> HashVector:
        """Generate hash vector from hash signal."""
        try:
            # Create hash vector representation
            hash_bytes = hash_signal.encode('utf-8')
            hash_digest = hashlib.sha256(hash_bytes).hexdigest()

            # Convert to numerical vector
            vector_data = []
            for i in range(0, len(hash_digest), 2):
                hex_pair = hash_digest[i:i+2]
                vector_data.append(int(hex_pair, 16) / 255.0)

            # Pad or truncate to target dimension
            target_dim = self.config['vector_dimension']
            if len(vector_data) < target_dim:
                # Pad with zeros
                vector_data.extend([0.0] * (target_dim - len(vector_data)))
            else:
                # Truncate
                vector_data = vector_data[:target_dim]

            vector_representation = np.array(vector_data)

            # Calculate phase components using FFT
            fft_result = np.fft.fft(vector_representation)
            phase_components = np.angle(fft_result)

            # Create hash vector
            hash_vector = HashVector()
                vector_id=f"hash_{int(time.time() * 1000)}",
                    hash_value=hash_signal,
                        vector_representation=vector_representation,
                        phase_components=phase_components,
                        timestamp=time.time()
            )

            return hash_vector

        except Exception as e:
            logger.error(f"Error generating hash vector: {e}")
            return self._create_default_hash_vector()

    def _calculate_similarity_matrix(self, hash_vector: HashVector) -> NDArray:
        """Calculate similarity matrix from hash vector."""
        try:
            # Get vector representation
            vector = hash_vector.vector_representation

            # Create similarity matrix with historical vectors
            if not self.hash_vectors:
                # Create identity matrix if no history
                matrix_size = min(len(vector), self.config['projection_resolution'])
                return np.eye(matrix_size)

            # Calculate similarities with historical vectors
            similarities = []
            for hist_vector in self.hash_vectors[-10:]:  # Last 10 vectors
                similarity = 1.0 - cosine(vector, hist_vector.vector_representation)
                similarities.append(similarity)

            # Create similarity matrix
            matrix_size = min(len(similarities), self.config['projection_resolution'])
            similarity_matrix = np.zeros((matrix_size, matrix_size))

            for i in range(matrix_size):
                for j in range(matrix_size):
                    if i < len(similarities) and j < len(similarities):
                        similarity_matrix[i, j] = similarities[i] * similarities[j]
                    else:
                        similarity_matrix[i, j] = 0.0

            return similarity_matrix

        except Exception as e:
            logger.error(f"Error calculating similarity matrix: {e}")
            return np.eye(min(len(hash_vector.vector_representation), 10))

    def _calculate_phase_alignment(self, hash_vector: HashVector) -> NDArray:
        """Calculate phase alignment from hash vector."""
        try:
            # Get phase components
            phase_components = hash_vector.phase_components

            # Create phase alignment matrix
            matrix_size = min(len(phase_components), self.config['projection_resolution'])
            phase_alignment = np.zeros((matrix_size, matrix_size))

            for i in range(matrix_size):
                for j in range(matrix_size):
                    if i < len(phase_components) and j < len(phase_components):
                        # Calculate phase difference
                        phase_diff = abs(phase_components[i] - phase_components[j])
                        # Normalize to [0, 1]
                        alignment = 1.0 - (phase_diff / (2 * np.pi))
                        phase_alignment[i, j] = alignment
                    else:
                        phase_alignment[i, j] = 0.0

            return phase_alignment

        except Exception as e:
            logger.error(f"Error calculating phase alignment: {e}")
            return np.eye(min(len(hash_vector.phase_components), 10))

    def _calculate_overlay_confidence(self, similarity_matrix: NDArray, )
                                    phase_alignment: NDArray) -> float:
        """Calculate overall overlay confidence."""
        try:
            # Calculate similarity confidence
            similarity_confidence = self.calculate_overlay_confidence(similarity_matrix)

            # Calculate phase alignment confidence
            phase_confidence = self.calculate_overlay_confidence(phase_alignment)

            # Weighted combination
            cosine_weight = self.config['cosine_similarity_weight']
            phase_weight = self.config['phase_alignment_weight']

            overall_confidence = (cosine_weight * similarity_confidence + )
                                phase_weight * phase_confidence)

            return min(1.0, max(0.0, overall_confidence))

        except Exception as e:
            logger.error(f"Error calculating overlay confidence: {e}")
            return 0.0

    def _determine_overlay_type(self, similarity_matrix: NDArray, )
                              phase_alignment: NDArray) -> OverlayType:
        """Determine overlay type based on matrix characteristics."""
        try:
            # Calculate matrix properties
            sim_mean = np.mean(similarity_matrix)
            phase_mean = np.mean(phase_alignment)

            # Determine type based on characteristics
            if sim_mean > 0.8 and phase_mean > 0.8:
                return OverlayType.HASH_SIMILARITY
            elif phase_mean > 0.7:
                return OverlayType.PHASE_ALIGNMENT
            elif sim_mean > 0.7:
                return OverlayType.COSINE_PROJECTION
            else:
                return OverlayType.MATRIX_MAPPING

        except Exception as e:
            logger.error(f"Error determining overlay type: {e}")
            return OverlayType.MATRIX_MAPPING

    def _create_default_overlay_map(self) -> OverlayMap:
        """Create default overlay map."""
        return OverlayMap()
            map_id="default",
                overlay_type=OverlayType.MATRIX_MAPPING,
                    hash_signal="default",
                    similarity_matrix=np.eye(10),
                    phase_alignment=np.eye(10),
                    confidence_score=0.0,
                    timestamp=time.time()
        )

    def _create_default_hash_vector(self) -> HashVector:
        """Create default hash vector."""
        return HashVector()
            vector_id="default",
                hash_value="default",
                    vector_representation=np.zeros(self.config['vector_dimension']),
                    phase_components=np.zeros(self.config['vector_dimension']),
                    timestamp=time.time()
        )

    def get_overlay_summary(self) -> Dict[str, Any]:
        """Get overlay analysis summary."""
        try:
            if not self.overlay_maps:
                return {}
                    "total_mappings": 0,
                        "successful_projections": 0,
                            "average_confidence": 0.0,
                            "most_common_type": "matrix_mapping",
                            "average_similarity": 0.0
}
            # Calculate statistics
            confidences = [m.confidence_score for m in self.overlay_maps]
            similarities = [np.mean(m.similarity_matrix) for m in self.overlay_maps]

            # Count overlay types
            type_counts = {}
            for overlay_map in self.overlay_maps:
                overlay_type = overlay_map.overlay_type.value
                type_counts[overlay_type] = type_counts.get(overlay_type, 0) + 1

            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "matrix_mapping"

            return {}
                "total_mappings": self.total_mappings,
                    "successful_projections": self.successful_projections,
                        "average_confidence": np.mean(confidences),
                        "most_common_type": most_common_type,
                        "average_similarity": np.mean(similarities),
                        "projection_rate": self.successful_projections / self.total_mappings if self.total_mappings > 0 else 0.0
}
        except Exception as e:
            logger.error(f"Error getting overlay summary: {e}")
            return {}

    def get_recent_overlays(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent overlay maps."""
        recent_overlays = self.overlay_maps[-count:]
        return []
            {}
                "map_id": m.map_id,
                    "timestamp": m.timestamp,
                        "overlay_type": m.overlay_type.value,
                        "hash_signal": m.hash_signal[:16] + "...",
                        "confidence_score": m.confidence_score,
                        "similarity_mean": np.mean(m.similarity_matrix)
}
            for m in recent_overlays
]
    def export_overlay_data(self, filepath: str) -> bool:
        """"""
        Export overlay data to JSON file.

        Args:
            filepath: Output file path

        Returns:
            True if export was successful
        """"""
        try:
            import json

            data = {
                "export_timestamp": time.time(),
                "config": self.config,
                "summary": self.get_overlay_summary(),
                "recent_overlays": self.get_recent_overlays(20)
}
}
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported overlay data to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting overlay data: {e}")
            return False