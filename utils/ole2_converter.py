"""OLE2 .doc to DOCX conversion utility for enhanced image extraction support.

Uses LibreOffice subprocess (when available) for reliable conversion, with pure Python
fallback for unsupported environments. LibreOffice is preferred because it properly
handles the complex Word 97-2003 binary format.
"""
from __future__ import annotations

import os
import io
import subprocess
import shutil
from typing import Optional

from .logger import ProcessLogger, get_logger

LOGGER = get_logger(__name__)


class OLE2Converter:
    """
    Detects and converts OLE2 .doc files to modern DOCX format.
    
    Docling can extract images from DOCX and PDF, but not from legacy OLE2 .doc files.
    This converter programmatically converts OLE2 to DOCX using pure Python (olefile + python-docx).
    """
    
    @staticmethod
    def detect_ole2_format(file_path: str) -> bool:
        """
        Detect if a file is in OLE2 binary format (Word 97-2003 .doc).
        
        OLE2 files start with magic bytes: 0xD0 0xCF 0x11 0xE0
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if file is OLE2 format, False otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'\xd0\xcf\x11\xe0'
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_FETCH,
                f"Failed to detect OLE2 format for {file_path}: {e}"
            )
            return False
    
    @staticmethod
    def _extract_text_from_ole2(ole: object) -> str:
        """Extract text content from OLE2 document using olefile.
        
        WARNING: Word 97-2003 binary format is complex and not fully parseable
        without understanding the complete format specification. This extraction
        is LIMITED and may miss content or be incomplete.
        
        Args:
            ole: olefile.OleFileIO object
            
        Returns:
            Extracted text content (may be empty or incomplete)
        """
        try:
            text_content = []
            
            # Try to find text in common Word stream locations
            # Note: Word 97-2003 uses complex binary encoding that can't be fully
            # parsed without implementing the complete format specification
            
            # Attempt 1: Try "WordDocument" stream - but this is mainly metadata
            if ole.exists('WordDocument'):
                try:
                    data = ole.openstream('WordDocument').read()
                    # WordDocument stream structure: first 28 bytes are header
                    # Word text is usually in the "1Table" stream, not WordDocument
                    # Try to find any readable ASCII text as fallback
                    decoded = data.decode('utf-8', errors='ignore')
                    # Only keep lines with mostly readable content
                    lines = [line.strip() for line in decoded.split('\n') if line.strip()]
                    # Filter: keep lines that have more ASCII than garbage
                    readable_lines = [
                        line for line in lines 
                        if len([c for c in line if 32 <= ord(c) < 127]) > len(line) * 0.3
                    ]
                    text_content.extend(readable_lines[:10])  # Limit to avoid garbage
                except:
                    pass
            
            return '\n'.join(text_content).strip() if text_content else ''
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_FETCH,
                f"Error extracting text from OLE2 (pure Python fallback): {str(e)[:60]}"
            )
            return ''

    
    @staticmethod
    def _extract_embedded_images(ole: object) -> list:
        """Extract embedded images/objects from OLE2 document.
        
        Args:
            ole: olefile.OleFileIO object
            
        Returns:
            List of (image_name, image_data) tuples
        """
        images = []
        try:
            # Look for embedded objects in OLE2
            for entry in ole.listdir():
                entry_path = '/'.join(entry)
                
                # Check for OLE object streams
                if 'Object' in entry_path or 'ole10Native' in entry_path:
                    try:
                        data = ole.openstream(entry).read()
                        if data:
                            images.append((entry_path, data))
                    except:
                        pass
            
            return images
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_FETCH,
                f"Error extracting images from OLE2: {e}"
            )
            return []
    
    @staticmethod
    def convert_ole2_to_docx(file_path: str, timeout: int = 120) -> str:
        """
        Convert OLE2 .doc file to DOCX format using LibreOffice subprocess or fallback.
        
        Strategy (in order of preference):
        1. **LibreOffice subprocess (preferred)**: Uses 'soffice --headless --convert-to docx'
           - Available in the target container runtime
           - Properly handles Word 97-2003 binary format
           - Produces valid, readable DOCX files
        
        2. **Pure Python fallback (local testing only)**: Uses olefile + python-docx
           - For environments where LibreOffice is not available
           - Limited text extraction from binary streams
           - May produce incomplete but valid DOCX
        
        Args:
            file_path: Path to OLE2 .doc file
            timeout: Timeout in seconds (default 120)
            
        Returns:
            Path to converted DOCX file, or original file if conversion fails.
        """
        # Check if file is OLE2
        if not OLE2Converter.detect_ole2_format(file_path):
            return file_path  # Not OLE2, return original
        
        # Check file extension
        if not file_path.lower().endswith('.doc'):
            return file_path  # Not a .doc file
        
        try:
            base_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(base_dir, f"{base_name_no_ext}_converted.docx")
            
            # Try LibreOffice subprocess first (preferred method)
            if OLE2Converter._try_libreoffice_conversion(file_path, output_path, timeout):
                return output_path
            
            # Fall back to pure Python if LibreOffice not available
            LOGGER.log_stage(
                ProcessLogger.STAGE_FETCH,
                "LibreOffice not available, attempting pure Python OLE2 conversion (limited)"
            )
            return OLE2Converter._convert_ole2_pure_python(file_path, output_path)
            
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_FETCH,
                f"OLE2 conversion failed ({type(e).__name__}: {str(e)[:100]}), using original file"
            )
            return file_path
    
    @staticmethod
    def _try_libreoffice_conversion(file_path: str, output_path: str, timeout: int) -> bool:
        """
        Attempt to convert OLE2 to DOCX using LibreOffice subprocess.
        
        Args:
            file_path: Input .doc file path
            output_path: Output .docx file path
            timeout: Subprocess timeout in seconds
            
        Returns:
            True if conversion succeeded, False otherwise
        """
        # Check if soffice is available
        soffice_path = shutil.which('soffice') or shutil.which('libreoffice')
        if not soffice_path:
            return False
        
        try:
            LOGGER.log_stage(
                ProcessLogger.STAGE_FETCH,
                f"Converting OLE2 .doc to DOCX using LibreOffice: {os.path.basename(file_path)}"
            )
            
            # Use LibreOffice headless mode to convert
            # soffice --headless --convert-to docx file.doc --outdir /tmp/
            output_dir = os.path.dirname(output_path)
            cmd = [
                soffice_path,
                '--headless',
                '--convert-to', 'docx',
                '--outdir', output_dir,
                file_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True
            )
            
            if result.returncode != 0:
                LOGGER.log_warning(
                    ProcessLogger.STAGE_FETCH,
                    f"LibreOffice conversion failed: {result.stderr[:100]}"
                )
                return False
            
            # LibreOffice creates .docx with same base name in the output directory
            # We need to rename it to match our expected output_path
            expected_name = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.docx')
            if os.path.exists(expected_name) and expected_name != output_path:
                shutil.move(expected_name, output_path)
            
            if not os.path.exists(output_path):
                LOGGER.log_warning(
                    ProcessLogger.STAGE_FETCH,
                    f"LibreOffice conversion did not produce output file"
                )
                return False
            
            original_size = os.path.getsize(file_path)
            converted_size = os.path.getsize(output_path)
            
            LOGGER.log_stage(
                ProcessLogger.STAGE_FETCH,
                f"✅ LibreOffice conversion succeeded: {original_size:,} → {converted_size:,} bytes"
            )
            return True
            
        except subprocess.TimeoutExpired:
            LOGGER.log_warning(
                ProcessLogger.STAGE_FETCH,
                f"LibreOffice conversion timed out after {timeout}s"
            )
            return False
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_FETCH,
                f"LibreOffice conversion error: {str(e)[:100]}"
            )
            return False
    
    @staticmethod
    def _convert_ole2_pure_python(file_path: str, output_path: str) -> str:
        """
        Pure Python fallback for OLE2 to DOCX conversion.
        
        WARNING: This is a limited implementation for environments where LibreOffice
        is not available. Pure Python cannot fully parse the Word 97-2003 binary format,
        so text extraction may be incomplete or corrupted.
        
        Args:
            file_path: Input .doc file path
            output_path: Output .docx file path
            
        Returns:
            Path to converted DOCX file, or original file if conversion fails
        """
        try:
            import olefile
            from docx import Document
            
            LOGGER.log_stage(
                ProcessLogger.STAGE_FETCH,
                f"Converting OLE2 .doc to DOCX (pure Python, limited): {os.path.basename(file_path)}"
            )
            
            # Open OLE2 file
            if not olefile.isOleFile(file_path):
                LOGGER.log_warning(
                    ProcessLogger.STAGE_FETCH,
                    f"File is not valid OLE2 format, returning original"
                )
                return file_path
            
            ole = olefile.OleFileIO(file_path)
            
            # Extract text from OLE2 streams
            text_content = OLE2Converter._extract_text_from_ole2(ole)
            
            ole.close()
            
            # Create new DOCX file
            doc = Document()
            
            # Add extracted text to document
            if text_content.strip():
                # Split by paragraphs and add to document
                paragraphs = text_content.split('\n')
                for para_text in paragraphs:
                    para_text = para_text.strip()
                    if para_text:
                        try:
                            doc.add_paragraph(para_text)
                        except Exception as para_error:
                            LOGGER.log_warning(
                                ProcessLogger.STAGE_FETCH,
                                f"Error adding paragraph to DOCX: {str(para_error)[:60]}"
                            )
            else:
                # If no text extracted, add a note
                doc.add_paragraph('[OLE2 document: text extraction limited in pure Python mode]')
            
            doc.save(output_path)
            
            original_size = os.path.getsize(file_path)
            converted_size = os.path.getsize(output_path)
            
            LOGGER.log_stage(
                ProcessLogger.STAGE_FETCH,
                f"⚠️  Pure Python conversion completed (may be incomplete): {original_size:,} → {converted_size:,} bytes"
            )
            return output_path
            
        except ImportError as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_FETCH,
                f"Required library not found ({e}). Install: pip install olefile"
            )
            return file_path
        except Exception as e:
            LOGGER.log_warning(
                ProcessLogger.STAGE_FETCH,
                f"Pure Python OLE2 conversion failed ({type(e).__name__}: {str(e)[:100]}), using original file"
            )
            return file_path

