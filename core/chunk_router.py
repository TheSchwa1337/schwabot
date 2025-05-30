"""
Chunk Router Core for Schwabot
Detects file type/subtype/format, determines optimal chunk size, and routes chunks for profit mapping.
"""

import os
import hashlib
import mimetypes
from typing import Dict, List, Optional, Tuple
import numpy as np

# Example file map for type/subtype/format logic
file_map = {
    "Resume": {
        "format": ["pdf", "docx"],
        "subtypes": ["Technical Resume", "Creative Resume"],
        "chunk_size": 512
    },
    "Report": {
        "format": ["pdf", "docx"],
        "subtypes": ["Research report", "Security Policy"],
        "chunk_size": 1024
    },
    "Spreadsheet": {
        "format": ["xlsx", "csv"],
        "subtypes": ["Survey", "Financial"],
        "chunk_size": 256
    }
}

def detect_file_type(filepath: str) -> Tuple[str, str, str]:
    """
    Detect file type, subtype, and format from filename and extension.
    Returns (type, subtype, format)
    """
    filename = os.path.basename(filepath)
    ext = filename.split('.')[-1].lower()
    for ftype, meta in file_map.items():
        if ext in meta["format"]:
            # Try to guess subtype from filename
            for subtype in meta["subtypes"]:
                if subtype.lower().replace(' ', '') in filename.lower().replace(' ', ''):
                    return ftype, subtype, ext
            return ftype, meta["subtypes"][0], ext
    return "Unknown", "Unknown", ext

def get_chunk_size(file_type: str, file_format: str, file_entropy: Optional[float] = None) -> int:
    base_size = file_map.get(file_type, {}).get("chunk_size", 512)
    if file_entropy is not None:
        if file_entropy > 0.9:
            return max(64, base_size // 2)
        elif file_entropy < 0.3:
            return min(4096, base_size * 2)
    if file_format in ['pdf', 'png']:
        return int(base_size * 1.5)
    elif file_format in ['txt', 'csv']:
        return int(base_size * 0.75)
    return base_size

def calculate_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    values = np.frombuffer(data, dtype=np.uint8)
    hist, _ = np.histogram(values, bins=256, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def chunk_and_hash(file_bytes: bytes, chunk_size: int) -> List[Dict]:
    chunks = [file_bytes[i:i + chunk_size] for i in range(0, len(file_bytes), chunk_size)]
    hash_chunks = []
    for idx, chunk in enumerate(chunks):
        chunk_hash = hashlib.sha256(chunk).hexdigest()
        entropy = calculate_entropy(chunk)
        hash_chunks.append({
            "index": idx,
            "hash": chunk_hash,
            "entropy": entropy,
            "size": len(chunk)
        })
    return hash_chunks

def route_file(filepath: str) -> List[Dict]:
    """
    Main entry: routes file through chunking and hashing pipeline.
    Returns list of chunk meta dicts.
    """
    file_type, subtype, file_format = detect_file_type(filepath)
    with open(filepath, 'rb') as f:
        file_bytes = f.read()
    file_entropy = calculate_entropy(file_bytes)
    chunk_size = get_chunk_size(file_type, file_format, file_entropy)
    chunk_meta = chunk_and_hash(file_bytes, chunk_size)
    # Attach routing info
    for meta in chunk_meta:
        meta.update({
            "file_type": file_type,
            "subtype": subtype,
            "format": file_format,
            "file_entropy": file_entropy,
            "chunk_size": chunk_size
        })
    return chunk_meta

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chunk_router.py <file_path>")
    else:
        meta = route_file(sys.argv[1])
        for m in meta:
            print(m) 