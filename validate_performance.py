#!/usr/bin/env python3
"""Quick performance validation of HolographicMemory C++ implementation."""

import time
import tempfile
from pathlib import Path
import sys

# Add holographic-fs to path
sys.path.insert(0, 'holographic-fs')

from holographicfs.memory import HoloFS as HolographicFS

def test_performance():
    """Test real C++ performance."""
    print("🔬 Testing HolographicMemory Performance...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize
        fs = HolographicFS(Path(tmpdir), grid_size=64)
        
        # Create test file
        test_file = Path(tmpdir) / "test.txt"
        test_content = b"This is a test of holographic memory performance!" * 100
        test_file.write_bytes(test_content)
        
        # Test store performance
        print(f"\n📦 Testing STORE performance...")
        start = time.perf_counter()
        doc_id = fs.store(test_file)
        store_time = (time.perf_counter() - start) * 1000  # Convert to ms
        
        print(f"  ✅ Store time: {store_time:.3f}ms")
        print(f"  ✅ Document ID: {doc_id}")
        
        # Test search performance
        print(f"\n🔍 Testing SEARCH performance...")
        start = time.perf_counter()
        results = fs.search("holographic")
        search_time = (time.perf_counter() - start) * 1000
        
        print(f"  ✅ Search time: {search_time:.3f}ms")
        print(f"  ✅ Results found: {len(results)}")
        
        # Get stats
        stats = fs.stats()
        print(f"\n📊 System Statistics:")
        print(f"  ✅ Backend: {stats.get('backend', 'Unknown')}")
        print(f"  ✅ Files indexed: {stats.get('files_indexed', 0)}")
        print(f"  ✅ Compression: {stats.get('compression_x', 'N/A')}")
        
        # Validate performance claims
        print(f"\n🏆 Performance Validation:")
        if store_time < 1.0:  # Sub-millisecond
            print(f"  ✅ VERIFIED: Store time {store_time:.3f}ms < 1ms target")
        else:
            print(f"  ⚠️  Store time {store_time:.3f}ms exceeds 1ms target")
            
        if search_time < 1.0:
            print(f"  ✅ VERIFIED: Search time {search_time:.3f}ms < 1ms target")
        else:
            print(f"  ⚠️  Search time {search_time:.3f}ms exceeds 1ms target")
            
        # Check if using C++ backend
        if "C++" in stats.get('backend', ''):
            print(f"  ✅ VERIFIED: Using real C++ backend")
        else:
            print(f"  ❌ WARNING: Not using C++ backend!")
            
        return store_time < 1.0 and search_time < 1.0 and "C++" in stats.get('backend', '')

if __name__ == "__main__":
    success = test_performance()
    sys.exit(0 if success else 1)
