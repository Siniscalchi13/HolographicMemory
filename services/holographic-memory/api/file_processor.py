#!/usr/bin/env python3
"""
Advanced File Processing System for Holographic Memory
Supports PDF, DOCX, TXT, and other document formats with content extraction and analysis.
"""

import io
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import hashlib
import base64

# Document processing libraries
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    fitz = None

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    DocxDocument = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

logger = logging.getLogger(__name__)


class FileProcessor:
    """
    Advanced file processor for holographic memory system.
    Extracts text content, metadata, and generates thumbnails for various file formats.
    """
    
    def __init__(self):
        self.supported_formats = {
            'pdf': PDF_AVAILABLE,
            'docx': DOCX_AVAILABLE,
            'txt': True,
            'md': True,
            'csv': PANDAS_AVAILABLE,
            'json': True,
            'xml': True,
            'html': True,
            'rtf': True,
        }
        
        # Image formats for thumbnail generation
        self.image_formats = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
        
    def is_supported(self, filename: str) -> bool:
        """Check if file format is supported for processing."""
        ext = Path(filename).suffix.lower().lstrip('.')
        return ext in self.supported_formats and self.supported_formats[ext]
    
    def get_file_info(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Get comprehensive file information including metadata and content preview."""
        ext = Path(filename).suffix.lower().lstrip('.')
        
        info = {
            'filename': filename,
            'extension': ext,
            'size': len(content),
            'mime_type': mimetypes.guess_type(filename)[0] or 'application/octet-stream',
            'hash': hashlib.sha256(content).hexdigest(),
            'supported': self.is_supported(filename),
            'content_type': self._get_content_type(ext),
            'metadata': {},
            'text_content': '',
            'thumbnail': None,
            'pages': 0,
            'word_count': 0,
            'language': 'unknown'
        }
        
        if info['supported']:
            try:
                # Extract content based on file type
                if ext == 'pdf':
                    info.update(self._process_pdf(content))
                elif ext == 'docx':
                    info.update(self._process_docx(content))
                elif ext in {'txt', 'md', 'json', 'xml', 'html', 'rtf'}:
                    info.update(self._process_text(content, ext))
                elif ext == 'csv':
                    info.update(self._process_csv(content))
                else:
                    info.update(self._process_generic(content, ext))
                    
                # Generate thumbnail if possible
                info['thumbnail'] = self._generate_thumbnail(content, ext)
                
            except Exception as e:
                logger.warning(f"Error processing file {filename}: {e}")
                info['error'] = str(e)
        
        return info
    
    def _get_content_type(self, ext: str) -> str:
        """Determine content type category for routing."""
        if ext in {'pdf', 'docx', 'txt', 'md', 'rtf'}:
            return 'document'
        elif ext in {'csv', 'json', 'xml'}:
            return 'data'
        elif ext in self.image_formats:
            return 'image'
        else:
            return 'binary'
    
    def _process_pdf(self, content: bytes) -> Dict[str, Any]:
        """Process PDF files and extract text, metadata, and thumbnails."""
        if not PDF_AVAILABLE:
            return {'error': 'PDF processing not available'}
        
        result = {
            'text_content': '',
            'pages': 0,
            'metadata': {},
            'word_count': 0,
            'language': 'unknown'
        }
        
        try:
            # Open PDF from bytes
            doc = fitz.open(stream=content, filetype="pdf")
            result['pages'] = len(doc)
            
            # Extract metadata
            metadata = doc.metadata
            result['metadata'] = {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', '')
            }
            
            # Extract text from all pages
            text_parts = []
            for page_num in range(min(result['pages'], 10)):  # Limit to first 10 pages
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            
            result['text_content'] = '\n\n'.join(text_parts)
            result['word_count'] = len(result['text_content'].split())
            
            # Detect language (simple heuristic)
            result['language'] = self._detect_language(result['text_content'])
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            result['error'] = str(e)
        
        return result
    
    def _process_docx(self, content: bytes) -> Dict[str, Any]:
        """Process DOCX files and extract text and metadata."""
        if not DOCX_AVAILABLE:
            return {'error': 'DOCX processing not available'}
        
        result = {
            'text_content': '',
            'pages': 0,
            'metadata': {},
            'word_count': 0,
            'language': 'unknown'
        }
        
        try:
            # Open DOCX from bytes
            doc = DocxDocument(io.BytesIO(content))
            
            # Extract text from paragraphs
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            result['text_content'] = '\n'.join(text_parts)
            result['word_count'] = len(result['text_content'].split())
            
            # Extract metadata
            core_props = doc.core_properties
            result['metadata'] = {
                'title': core_props.title or '',
                'author': core_props.author or '',
                'subject': core_props.subject or '',
                'keywords': core_props.keywords or '',
                'created': str(core_props.created) if core_props.created else '',
                'modified': str(core_props.modified) if core_props.modified else ''
            }
            
            # Estimate pages (rough calculation)
            result['pages'] = max(1, result['word_count'] // 250)  # ~250 words per page
            
            # Detect language
            result['language'] = self._detect_language(result['text_content'])
            
        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            result['error'] = str(e)
        
        return result
    
    def _process_text(self, content: bytes, ext: str) -> Dict[str, Any]:
        """Process plain text files."""
        result = {
            'text_content': '',
            'pages': 0,
            'metadata': {},
            'word_count': 0,
            'language': 'unknown'
        }
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            text_content = None
            
            for encoding in encodings:
                try:
                    text_content = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                # Fallback to utf-8 with error handling
                text_content = content.decode('utf-8', errors='replace')
            
            result['text_content'] = text_content
            result['word_count'] = len(text_content.split())
            result['pages'] = max(1, result['word_count'] // 250)
            result['language'] = self._detect_language(text_content)
            
            # Extract basic metadata for text files
            lines = text_content.split('\n')
            result['metadata'] = {
                'lines': len(lines),
                'characters': len(text_content),
                'non_empty_lines': len([line for line in lines if line.strip()])
            }
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            result['error'] = str(e)
        
        return result
    
    def _process_csv(self, content: bytes) -> Dict[str, Any]:
        """Process CSV files and extract structure information."""
        if not PANDAS_AVAILABLE:
            return self._process_text(content, 'csv')  # Fallback to text processing
        
        result = {
            'text_content': '',
            'pages': 0,
            'metadata': {},
            'word_count': 0,
            'language': 'data'
        }
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=encoding, nrows=1000)  # Limit rows for performance
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            
            if df is not None:
                result['metadata'] = {
                    'columns': list(df.columns),
                    'rows': len(df),
                    'data_types': df.dtypes.to_dict(),
                    'sample_data': df.head(5).to_dict('records')
                }
                
                # Create text representation
                result['text_content'] = f"CSV with {len(df)} rows and {len(df.columns)} columns\n"
                result['text_content'] += f"Columns: {', '.join(df.columns)}\n\n"
                result['text_content'] += df.head(10).to_string()
                result['word_count'] = len(result['text_content'].split())
                result['pages'] = max(1, result['word_count'] // 250)
            else:
                # Fallback to text processing
                return self._process_text(content, 'csv')
                
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            result['error'] = str(e)
        
        return result
    
    def _process_generic(self, content: bytes, ext: str) -> Dict[str, Any]:
        """Process generic files with basic information."""
        return {
            'text_content': f"Binary file of type {ext}",
            'pages': 1,
            'metadata': {'type': 'binary', 'extension': ext},
            'word_count': 0,
            'language': 'binary'
        }
    
    def _generate_thumbnail(self, content: bytes, ext: str) -> Optional[str]:
        """Generate thumbnail for supported file types."""
        try:
            if ext in self.image_formats and PIL_AVAILABLE:
                # Image thumbnail
                img = Image.open(io.BytesIO(content))
                img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                return base64.b64encode(buf.getvalue()).decode('ascii')
            
            elif ext == 'pdf' and PDF_AVAILABLE:
                # PDF first page thumbnail
                doc = fitz.open(stream=content, filetype="pdf")
                if len(doc) > 0:
                    page = doc[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(200/page.rect.width, 200/page.rect.height))
                    png_data = pix.tobytes("png")
                    doc.close()
                    return base64.b64encode(png_data).decode('ascii')
                doc.close()
                
        except Exception as e:
            logger.warning(f"Error generating thumbnail: {e}")
        
        return None
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns."""
        if not text:
            return 'unknown'
        
        # Simple heuristics
        text_lower = text.lower()
        
        # English indicators
        if any(word in text_lower for word in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']):
            return 'en'
        
        # Spanish indicators
        if any(word in text_lower for word in ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le']):
            return 'es'
        
        # French indicators
        if any(word in text_lower for word in ['le', 'la', 'de', 'et', 'à', 'un', 'il', 'que', 'ne', 'se', 'ce', 'pas', 'tout', 'plus']):
            return 'fr'
        
        # German indicators
        if any(word in text_lower for word in ['der', 'die', 'das', 'und', 'in', 'den', 'von', 'zu', 'dem', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'an', 'als', 'eine', 'als']):
            return 'de'
        
        return 'unknown'
    
    def extract_text_for_holographic_processing(self, content: bytes, filename: str) -> str:
        """Extract text content optimized for holographic memory processing."""
        info = self.get_file_info(filename, content)
        
        if info.get('error'):
            return f"Error processing {filename}: {info['error']}"
        
        # Return structured text for holographic processing
        text_parts = []
        
        # Add metadata as structured text
        if info.get('metadata'):
            text_parts.append(f"METADATA: {info['metadata']}")
        
        # Add main content
        if info.get('text_content'):
            text_parts.append(f"CONTENT: {info['text_content']}")
        
        # Add file information
        text_parts.append(f"FILE_INFO: {filename}, {info['size']} bytes, {info['word_count']} words, {info['pages']} pages")
        
        return '\n\n'.join(text_parts)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about file processing capabilities."""
        return {
            'supported_formats': {fmt: available for fmt, available in self.supported_formats.items() if available},
            'libraries_available': {
                'pdf': PDF_AVAILABLE,
                'docx': DOCX_AVAILABLE,
                'pandas': PANDAS_AVAILABLE,
                'pil': PIL_AVAILABLE
            },
            'total_supported': sum(1 for available in self.supported_formats.values() if available)
        }


# Global instance
file_processor = FileProcessor()
