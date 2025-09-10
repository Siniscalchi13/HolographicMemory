#!/usr/bin/env python3
"""
Test script for the enhanced file processing system.
"""

import sys
import os
sys.path.insert(0, 'services/holographic-memory/api')

from file_processor import file_processor
import tempfile
from pathlib import Path

def test_text_file():
    """Test processing of a text file."""
    print("🧪 Testing text file processing...")
    
    content = b"""This is a test document for holographic memory processing.
    
It contains multiple paragraphs and various types of content:
- Bullet points
- Numbers: 1, 2, 3
- Special characters: @#$%^&*()

The document has about 50 words and should be processed correctly.
"""
    
    info = file_processor.get_file_info("test.txt", content)
    
    print(f"✅ Text file processed:")
    print(f"   - Word count: {info['word_count']}")
    print(f"   - Language: {info['language']}")
    print(f"   - Content preview: {info['text_content'][:100]}...")
    
    return info

def test_pdf_file():
    """Test processing of a PDF file (if available)."""
    print("\n🧪 Testing PDF file processing...")
    
    # Create a simple PDF content (this is just a placeholder)
    # In a real test, you'd use an actual PDF file
    content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    
    info = file_processor.get_file_info("test.pdf", content)
    
    if info.get('error'):
        print(f"⚠️  PDF processing not available: {info['error']}")
    else:
        print(f"✅ PDF file processed:")
        print(f"   - Pages: {info['pages']}")
        print(f"   - Word count: {info['word_count']}")
    
    return info

def test_csv_file():
    """Test processing of a CSV file."""
    print("\n🧪 Testing CSV file processing...")
    
    content = b"""name,age,city
John,25,New York
Jane,30,Los Angeles
Bob,35,Chicago
Alice,28,San Francisco
"""
    
    info = file_processor.get_file_info("test.csv", content)
    
    print(f"✅ CSV file processed:")
    print(f"   - Word count: {info['word_count']}")
    print(f"   - Metadata: {info['metadata']}")
    
    return info

def test_unsupported_file():
    """Test processing of an unsupported file type."""
    print("\n🧪 Testing unsupported file processing...")
    
    content = b"Binary content that cannot be processed as text"
    
    info = file_processor.get_file_info("test.bin", content)
    
    print(f"✅ Unsupported file processed:")
    print(f"   - Supported: {info['supported']}")
    print(f"   - Content type: {info['content_type']}")
    
    return info

def test_processing_stats():
    """Test processing statistics."""
    print("\n🧪 Testing processing statistics...")
    
    stats = file_processor.get_processing_stats()
    
    print(f"✅ Processing statistics:")
    print(f"   - Supported formats: {list(stats['supported_formats'].keys())}")
    print(f"   - Libraries available: {stats['libraries_available']}")
    print(f"   - Total supported: {stats['total_supported']}")

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced File Processing System")
    print("=" * 50)
    
    try:
        # Test various file types
        test_text_file()
        test_pdf_file()
        test_csv_file()
        test_unsupported_file()
        test_processing_stats()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
