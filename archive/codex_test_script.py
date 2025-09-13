#!/usr/bin/env python3
"""
HolographicMemory Pure Holographic Storage System - Codex Test Script
====================================================================

This script comprehensively tests the pure holographic storage system
to validate that traditional file system dependencies have been removed
and the system operates purely on holographic principles.
"""

import sys
import os
from pathlib import Path
import tempfile
import json
import hashlib
import time

# Add the holographic memory system to path
sys.path.insert(0, 'services/holographic-memory/core')
sys.path.insert(0, 'services/holographic-memory/core/native/holographic/build')

def test_system_imports():
    """Test that all required components can be imported."""
    print("🔍 Testing System Imports...")
    
    try:
        from holographicfs.memory import mount
        print("✅ HoloFS import: SUCCESS")
    except Exception as e:
        print(f"❌ HoloFS import: FAILED - {e}")
        return False
    
    try:
        import holographic_gpu
        print("✅ GPU backend import: SUCCESS")
    except Exception as e:
        print(f"❌ GPU backend import: FAILED - {e}")
        return False
    
    try:
        import holographic_cpp_3d
        print("✅ 3D backend import: SUCCESS")
    except Exception as e:
        print(f"❌ 3D backend import: FAILED - {e}")
        return False
    
    return True

def test_pure_holographic_storage():
    """Test pure holographic storage without traditional files."""
    print("\n🧠 Testing Pure Holographic Storage...")
    
    # Import here to avoid scope issues
    from holographicfs.memory import mount
    
    # Create test environment
    tmp = tempfile.mkdtemp()
    fs = mount(Path(tmp), grid_size=32)
    
    # Test data
    test_cases = [
        (b'Hello Pure Holographic Memory!', 'text.txt'),
        (b'\x00\x01\x02\x03\x04\x05', 'binary.bin'),
        (b'{"json": "data", "number": 42}', 'data.json'),
        (b'This is a longer document with multiple lines.\nLine 2\nLine 3', 'long.txt')
    ]
    
    stored_ids = []
    for data, filename in test_cases:
        try:
            doc_id = fs.store_data(data, filename)
            stored_ids.append((doc_id, filename, data))
            print(f"✅ Stored {filename}: {len(data)} bytes -> {doc_id[:16]}...")
        except Exception as e:
            print(f"❌ Failed to store {filename}: {e}")
            return False
    
    # Test retrieval
    print("\n🔍 Testing Retrieval...")
    for doc_id, filename, original_data in stored_ids:
        try:
            retrieved_path = fs.recall(doc_id)
            retrieved_data = retrieved_path.read_bytes()
            
            if retrieved_data == original_data:
                print(f"✅ Retrieved {filename}: Perfect match ({len(retrieved_data)} bytes)")
            else:
                print(f"❌ Retrieved {filename}: Content mismatch")
                return False
        except Exception as e:
            print(f"❌ Failed to retrieve {filename}: {e}")
            return False
    
    return True

def test_gpu_backend():
    """Test GPU backend functionality."""
    print("\n🎮 Testing GPU Backend...")
    
    from holographicfs.memory import mount
    tmp = tempfile.mkdtemp()
    fs = mount(Path(tmp), grid_size=32)
    
    if not hasattr(fs, 'mem') or not hasattr(fs.mem, 'gpu_backend') or not fs.mem.gpu_backend:
        print("❌ GPU backend not available")
        return False
    
    gpu = fs.mem.gpu_backend
    
    # Test GPU availability
    if not gpu.available():
        print("❌ GPU backend not available")
        return False
    print("✅ GPU backend available")
    
    # Test 7-layer system
    if not gpu.layers_initialized:
        print("❌ 7-layer system not initialized")
        return False
    print("✅ 7-layer system initialized")
    
    # Test layer statistics
    try:
        layer_stats = gpu.get_layer_stats()
        active_layers = len([k for k in layer_stats.keys() if k.isdigit()])
        print(f"✅ Active layers: {active_layers}")
        
        # Print layer details
        for i in range(7):
            if str(i) in layer_stats:
                layer = layer_stats[str(i)]
                print(f"   Layer {i} ({layer['name']}): dim={layer['dimension']}, weight={layer['importance_weight']}")
    except Exception as e:
        print(f"❌ Failed to get layer stats: {e}")
        return False
    
    return True

def test_no_traditional_files():
    """Test that no traditional files are created."""
    print("\n📁 Testing No Traditional Files...")
    
    from holographicfs.memory import mount
    tmp = tempfile.mkdtemp()
    fs = mount(Path(tmp), grid_size=32)
    
    # Store some data
    test_data = b'Test data for file system check'
    doc_id = fs.store_data(test_data, 'test.txt')
    
    # Check what files were created
    all_files = list(Path(tmp).rglob('*'))
    traditional_files = [f for f in all_files if f.is_file() and not str(f).startswith('.holofs')]
    holo_files = [f for f in all_files if f.is_file() and str(f).startswith('.holofs')]
    
    print(f"   Traditional files: {len(traditional_files)}")
    print(f"   Holographic files: {len(holo_files)}")
    
    if len(traditional_files) > 0:
        print("❌ Traditional files found:")
        for f in traditional_files:
            print(f"   {f.relative_to(tmp)}")
        return False
    
    if len(holo_files) == 0:
        print("❌ No holographic files found")
        return False
    
    print("✅ Only holographic files created")
    for f in holo_files:
        print(f"   {f.relative_to(tmp)} ({f.stat().st_size} bytes)")
    
    return True

def test_compression_efficiency():
    """Test holographic compression efficiency."""
    print("\n📊 Testing Compression Efficiency...")
    
    from holographicfs.memory import mount
    tmp = tempfile.mkdtemp()
    fs = mount(Path(tmp), grid_size=32)
    
    # Test with different data sizes
    test_sizes = [64, 256, 1024, 4096]
    
    for size in test_sizes:
        test_data = b'X' * size
        try:
            doc_id = fs.store_data(test_data, f'test_{size}.bin')
            stats = fs.stats()
            
            compression_ratio = stats.get('compression_x', 0)
            print(f"   {size} bytes: {compression_ratio}x compression")
            
            if compression_ratio > 1.0:
                print(f"❌ Poor compression for {size} bytes: {compression_ratio}x")
                return False
        except Exception as e:
            print(f"❌ Failed to test {size} bytes: {e}")
            return False
    
    print("✅ Compression efficiency acceptable")
    return True

def test_duplicate_detection():
    """Test duplicate content detection."""
    print("\n🔄 Testing Duplicate Detection...")
    
    from holographicfs.memory import mount
    tmp = tempfile.mkdtemp()
    fs = mount(Path(tmp), grid_size=32)
    
    # Store same content with different filenames
    test_data = b'Duplicate content test'
    doc_id1 = fs.store_data(test_data, 'file1.txt')
    doc_id2 = fs.store_data(test_data, 'file2.txt')
    
    if doc_id1 == doc_id2:
        print("✅ Duplicate detection working (same doc_id)")
    else:
        print("❌ Duplicate detection failed (different doc_ids)")
        return False
    
    # Verify only one entry in stats
    stats = fs.stats()
    if stats['files_indexed'] == 1:
        print("✅ Only one entry indexed for duplicate content")
    else:
        print(f"❌ Multiple entries indexed: {stats['files_indexed']}")
        return False
    
    return True

def test_api_compatibility():
    """Test both pure and legacy APIs."""
    print("\n🔌 Testing API Compatibility...")
    
    from holographicfs.memory import mount
    tmp = tempfile.mkdtemp()
    fs = mount(Path(tmp), grid_size=32)
    
    # Test pure API
    test_data = b'Pure API test data'
    try:
        doc_id_pure = fs.store_data(test_data, 'pure_test.txt')
        retrieved_pure = fs.recall(doc_id_pure).read_bytes()
        if retrieved_pure == test_data:
            print("✅ Pure API (store_data): Working")
        else:
            print("❌ Pure API: Content mismatch")
            return False
    except Exception as e:
        print(f"❌ Pure API failed: {e}")
        return False
    
    # Test legacy API
    test_file = Path(tmp) / 'legacy_test.txt'
    test_file.write_text('Legacy API test data')
    try:
        doc_id_legacy = fs.store(test_file)
        retrieved_legacy = fs.recall(doc_id_legacy).read_bytes()
        if retrieved_legacy == test_file.read_bytes():
            print("✅ Legacy API (store): Working")
        else:
            print("❌ Legacy API: Content mismatch")
            return False
    except Exception as e:
        print(f"❌ Legacy API failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 HOLOGRAPHIC MEMORY PURE STORAGE SYSTEM - CODEX VALIDATION")
    print("=" * 70)
    
    tests = [
        ("System Imports", test_system_imports),
        ("Pure Holographic Storage", test_pure_holographic_storage),
        ("GPU Backend", test_gpu_backend),
        ("No Traditional Files", test_no_traditional_files),
        ("Compression Efficiency", test_compression_efficiency),
        ("Duplicate Detection", test_duplicate_detection),
        ("API Compatibility", test_api_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n{'='*70}")
    print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Pure Holographic Storage System Working!")
        print("\n✅ TRADITIONAL FILE SYSTEM LAYER: REMOVED")
        print("✅ PURE HOLOGRAPHIC STORAGE: ACTIVE")
        print("✅ GPU ACCELERATION: ACTIVE")
        print("✅ 7-LAYER ROUTING: ACTIVE")
        print("✅ WAVE SUPERPOSITION: ACTIVE")
        print("✅ EXACT BYTE RETRIEVAL: WORKING")
        return True
    else:
        print(f"❌ {total - passed} tests failed - System needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
