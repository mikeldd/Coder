"""
PDF text extraction module for medical records.
Uses PyMuPDF for primary extraction and Tesseract OCR for scanned documents.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available, falling back to other methods")

try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract OCR not available")

from ..core.file_manager import FileManager
from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Result of PDF text extraction."""
    text: str
    pages: List[str]
    metadata: Dict[str, Any]
    extraction_method: str
    confidence: float
    page_count: int
    is_scanned: bool
    images_extracted: int

class PDFExtractor:
    """Extracts text from medical record PDFs using multiple methods."""

    def __init__(self, config: Config, file_manager: FileManager):
        """Initialize PDF extractor."""
        self.config = config
        self.file_manager = file_manager
        self.ocr_language = config.get("processing.ocr_language", "eng")

        # Configure Tesseract if available
        if TESSERACT_AVAILABLE:
            try:
                # Try to find Tesseract path
                import os
                if os.name == 'nt':  # Windows
                    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                    if os.path.exists(tesseract_path):
                        pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info("Tesseract OCR initialized")
            except Exception as e:
                logger.warning(f"Tesseract initialization warning: {e}")

    def extract_text(self, secure_file_path: str) -> ExtractionResult:
        """
        Extract text from PDF using hybrid approach.
        Primary: PyMuPDF, Fallback: Tesseract OCR
        """
        try:
            # Get file content from secure storage
            pdf_content = self.file_manager.get_file_content(secure_file_path)

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_path = temp_file.name

            try:
                # Try PyMuPDF first (fastest and most accurate for text-based PDFs)
                if PYMUPDF_AVAILABLE:
                    result = self._extract_with_pymupdf(temp_path)
                    if result.confidence > 0.7:  # Good confidence
                        logger.info(f"Successfully extracted text with PyMuPDF: {result.page_count} pages")
                        return result

                # Fallback to OCR if PyMuPDF failed or low confidence
                if TESSERACT_AVAILABLE:
                    logger.info("PyMuPDF extraction insufficient, using OCR fallback")
                    result = self._extract_with_ocr(temp_path)
                    if result.confidence > 0.5:
                        logger.info(f"Successfully extracted text with OCR: {result.page_count} pages")
                        return result

                # Last resort: try basic text extraction
                result = self._extract_basic_text(temp_path)
                logger.warning("Used basic text extraction (low quality)")
                return result

            finally:
                # Clean up temporary file
                try:
                    Path(temp_path).unlink()
                except:
                    pass

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ExtractionResult(
                text="", pages=[], metadata={}, extraction_method="failed",
                confidence=0.0, page_count=0, is_scanned=False, images_extracted=0
            )

    def _extract_with_pymupdf(self, file_path: str) -> ExtractionResult:
        """Extract text using PyMuPDF with layout preservation."""
        try:
            doc = fitz.open(file_path)
            pages_text = []
            full_text = ""
            images_extracted = 0
            total_confidence = 0.0

            # Extract metadata
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'pdf_version': doc.pdf_version
            }

            # Check if PDF is likely scanned by looking at images
            is_scanned = self._is_scanned_pdf(doc)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text with layout information
                text = page.get_text("text")  # Simple text extraction
                if text.strip():
                    # Get text with formatting for better structure
                    dict_text = page.get_text("dict")
                    structured_text = self._format_structured_text(dict_text)
                    pages_text.append(structured_text)
                    full_text += structured_text + "\n\n"

                    # Calculate page confidence based on text density
                    page_area = page.rect
                    text_area = page.get_text("words")
                    confidence = min(len(text_area) / 50, 1.0)  # Simple confidence metric
                    total_confidence += confidence
                else:
                    # No text found, might be image-based
                    pages_text.append("")
                    total_confidence += 0.1  # Low confidence for blank pages

                # Count images on page
                images = page.get_images()
                images_extracted += len(images)

            doc.close()

            # Calculate overall confidence
            avg_confidence = total_confidence / len(doc) if doc else 0.0

            # If no text found but images exist, mark as scanned
            if not full_text.strip() and images_extracted > 0:
                is_scanned = True

            metadata['is_scanned'] = is_scanned
            metadata['total_images'] = images_extracted

            return ExtractionResult(
                text=full_text.strip(),
                pages=pages_text,
                metadata=metadata,
                extraction_method="pymupdf",
                confidence=avg_confidence,
                page_count=len(doc),
                is_scanned=is_scanned,
                images_extracted=images_extracted
            )

        except Exception as e:
            logger.error(f"PyMuPDF extraction error: {e}")
            raise

    def _extract_with_ocr(self, file_path: str) -> ExtractionResult:
        """Extract text using Tesseract OCR for scanned PDFs."""
        try:
            doc = fitz.open(file_path)
            pages_text = []
            full_text = ""
            total_confidence = 0.0
            images_extracted = 0

            metadata = {
                'extraction_method': 'tesseract_ocr',
                'ocr_language': self.ocr_language,
                'preprocessing_applied': True
            }

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # Higher resolution for better OCR
                pix = page.get_pixmap(matrix=mat)

                # Convert PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # Preprocess image for better OCR
                processed_img = self._preprocess_image_for_ocr(img)

                # Perform OCR
                try:
                    ocr_text = pytesseract.image_to_string(
                        processed_img,
                        lang=self.ocr_language,
                        config='--psm 6 --oem 3'  # Assume uniform block of text, LSTM OCR engine
                    )

                    # Get confidence data
                    data = pytesseract.image_to_data(
                        processed_img,
                        lang=self.ocr_language,
                        output_type=pytesseract.Output.DICT,
                        config='--psm 6 --oem 3'
                    )

                    # Calculate average confidence for words
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0
                    page_confidence = avg_conf / 100.0

                    pages_text.append(ocr_text.strip())
                    full_text += ocr_text.strip() + "\n\n"
                    total_confidence += page_confidence
                    images_extracted += 1

                except Exception as e:
                    logger.warning(f"OCR failed on page {page_num + 1}: {e}")
                    pages_text.append("")
                    total_confidence += 0.0

            doc.close()

            avg_confidence = total_confidence / len(doc) if doc else 0.0

            return ExtractionResult(
                text=full_text.strip(),
                pages=pages_text,
                metadata=metadata,
                extraction_method="tesseract_ocr",
                confidence=avg_confidence,
                page_count=len(doc),
                is_scanned=True,
                images_extracted=images_extracted
            )

        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            raise

    def _extract_basic_text(self, file_path: str) -> ExtractionResult:
        """Basic text extraction as last resort."""
        try:
            # This is a simplified fallback
            doc = fitz.open(file_path)
            full_text = ""

            for page in doc:
                text = page.get_text()
                full_text += text + "\n"

            doc.close()

            return ExtractionResult(
                text=full_text.strip(),
                pages=full_text.strip().split("\n\n"),
                metadata={'method': 'basic_fallback'},
                extraction_method="basic",
                confidence=0.3,  # Low confidence
                page_count=len(doc),
                is_scanned=False,
                images_extracted=0
            )

        except Exception as e:
            logger.error(f"Basic text extraction error: {e}")
            raise

    def _is_scanned_pdf(self, doc) -> bool:
        """Determine if PDF is likely scanned based on text vs image ratio."""
        try:
            total_pages = len(doc)
            text_pages = 0
            image_pages = 0

            for page in doc:
                text = page.get_text().strip()
                images = len(page.get_images())

                if len(text) > 50:  # Significant text found
                    text_pages += 1
                elif images > 0:
                    image_pages += 1

            # If most pages are image-based with little text, consider it scanned
            scanned_ratio = image_pages / total_pages if total_pages > 0 else 0
            return scanned_ratio > 0.6

        except Exception as e:
            logger.warning(f"Error determining if PDF is scanned: {e}")
            return False

    def _format_structured_text(self, dict_text: Dict) -> str:
        """Format structured text from PyMuPDF dict output."""
        try:
            formatted_lines = []

            if 'blocks' in dict_text:
                for block in dict_text['blocks']:
                    if 'lines' in block:
                        for line in block['lines']:
                            if 'spans' in line:
                                line_text = ""
                                for span in line['spans']:
                                    if 'text' in span:
                                        line_text += span['text']
                                if line_text.strip():
                                    formatted_lines.append(line_text.strip())

            return '\n'.join(formatted_lines)

        except Exception as e:
            logger.warning(f"Error formatting structured text: {e}")
            return ""

    def _preprocess_image_for_ocr(self, img: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy for medical documents."""
        try:
            # Convert to OpenCV format
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # Apply noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)

            # Thresholding for better text contrast
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Convert back to PIL Image
            return Image.fromarray(processed)

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return img  # Return original if preprocessing fails

    def extract_medical_metadata(self, extraction_result: ExtractionResult) -> Dict[str, Any]:
        """Extract medical-specific metadata from the text."""
        metadata = extraction_result.metadata.copy()
        text = extraction_result.text.lower()

        # Look for medical record indicators
        medical_indicators = {
            'patient_name': self._find_patient_name(text),
            'medical_record': any(term in text for term in ['medical record', 'patient record', 'clinical note']),
            'consultation': any(term in text for term in ['consultation', 'referral', 'specialist']),
            'soap_note': any(term in text for term in ['subjective:', 'objective:', 'assessment:', 'plan:']),
            'has_pharmacy': any(term in text for term in ['pharmacy', 'medication', 'prescription']),
            'has_labs': any(term in text for term in ['lab', 'laboratory', 'test result']),
            'has_vitals': any(term in text for term in ['vital signs', 'blood pressure', 'heart rate'])
        }

        metadata.update(medical_indicators)
        return metadata

    def _find_patient_name(self, text: str) -> Optional[str]:
        """Simple patient name extraction (basic pattern matching)."""
        # This is a simplified implementation
        # In production, you'd use more sophisticated NLP
        import re

        # Look for "Patient:" patterns
        patient_patterns = [
            r'patient[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'name[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'mr[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'mrs[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        ]

        for pattern in patient_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def get_extraction_quality_score(self, result: ExtractionResult) -> float:
        """Calculate quality score for extracted text."""
        score = 0.0

        # Base confidence from extraction method
        score += result.confidence * 0.4

        # Text length score (longer is usually better for medical records)
        text_length = len(result.text)
        length_score = min(text_length / 5000, 1.0)  # Normalize to 0-1
        score += length_score * 0.3

        # Medical content indicators
        medical_terms = ['patient', 'diagnosis', 'treatment', 'medication', 'symptom', 'examination']
        medical_count = sum(1 for term in medical_terms if term in result.text.lower())
        medical_score = min(medical_count / len(medical_terms), 1.0)
        score += medical_score * 0.2

        # Structure score (multiple pages usually indicate comprehensive records)
        structure_score = min(result.page_count / 5, 1.0)  # Normalize to 0-1, assume 5 pages is good
        score += structure_score * 0.1

        return min(score, 1.0)