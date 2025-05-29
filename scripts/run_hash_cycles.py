#!/usr/bin/env python3
"""
CLI script for running hash cycles with throttling
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scaling.hash_dispatcher import HashDispatcher
from scaling.throttle_manager import SystemState

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run hash cycles with throttling")
    
    parser.add_argument(
        "--min-workers",
        type=int,
        default=1,
        help="Minimum number of worker threads"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of worker threads (default: CPU count)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for hash operations"
    )
    
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Interval between throttle checks (seconds)"
    )
    
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input file to hash (default: stdin)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Size of chunks to read from input"
    )
    
    return parser.parse_args()

def read_chunks(file_path: str, chunk_size: int) -> List[bytes]:
    """Read chunks from file"""
    chunks = []
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
    return chunks

def hash_callback(result: str):
    """Callback for hash results"""
    print(f"Hash: {result}")

def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize dispatcher
    dispatcher = HashDispatcher(
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval
    )
    
    try:
        # Read input
        if args.input_file:
            chunks = read_chunks(args.input_file, args.chunk_size)
        else:
            # Read from stdin
            chunks = [sys.stdin.buffer.read(args.chunk_size)]
            
        # Run hash cycles
        start_time = time.time()
        results = dispatcher.dispatch_hashes(chunks, hash_callback)
        end_time = time.time()
        
        # Print summary
        print(f"\nProcessed {len(results)} chunks in {end_time - start_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        dispatcher.shutdown()

if __name__ == "__main__":
    main() 