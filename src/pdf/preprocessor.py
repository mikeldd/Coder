"""
Clinical note preprocessor for medical records.
Handles text cleaning, medical entity recognition, and structure identification.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# Try to import spaCy for medical NLP
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, using basic text processing")

from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    category: str

@dataclass
class StructuredNote:
    """Represents a structured medical note."""
    original_text: str
    cleaned_text: str
    sections: Dict[str, str]
    entities: List[MedicalEntity]
    document_type: str
    quality_score: float
    metadata: Dict[str, Any]

class ClinicalPreprocessor:
    """Preprocesses clinical notes for CPT coding analysis."""

    def __init__(self, config: Config):
        """Initialize clinical preprocessor."""
        self.config = config
        self.nlp = None
        self.document_types = {
            'consultation': ['consultation', 'referral', 'specialist opinion', 'second opinion'],
            'progress_note': ['progress note', 'follow-up', 'update', 'review'],
            'initial_eval': ['initial evaluation', 'intake', 'first visit', 'new patient'],
            'discharge_summary': ['discharge', 'summary', 'final report'],
            'operative_report': ['operative', 'surgery', 'procedure', 'operation'],
            'lab_result': ['lab result', 'laboratory', 'test result', 'pathology'],
            'imaging': ['x-ray', 'mri', 'ct scan', 'ultrasound', 'radiology']
        }

        # Medical abbreviation dictionary
        self.abbreviations = {
            'HTN': 'hypertension',
            'DM': 'diabetes mellitus',
            'CAD': 'coronary artery disease',
            'COPD': 'chronic obstructive pulmonary disease',
            'CHF': 'congestive heart failure',
            'MI': 'myocardial infarction',
            'CVA': 'cerebrovascular accident',
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'RR': 'respiratory rate',
            'T': 'temperature',
            'HT': 'height',
            'WT': 'weight',
            'BMI': 'body mass index',
            'BID': 'twice daily',
            'TID': 'three times daily',
            'QID': 'four times daily',
            'PRN': 'as needed',
            'STAT': 'immediately',
            'NPO': 'nothing by mouth',
            'PO': 'by mouth',
            'IV': 'intravenous',
            'IM': 'intramuscular',
            'SQ': 'subcutaneous',
            'c/o': 'complains of',
            'h/o': 'history of',
            's/p': 'status post',
            'r/o': 'rule out',
            'w/i': 'within',
            'w/o': 'without'
        }

        # Initialize spaCy model if available
        self._init_nlp_model()

    def _init_nlp_model(self):
        """Initialize spaCy model for medical NLP."""
        if not SPACY_AVAILABLE:
            return

        try:
            # Try to load medical model first
            try:
                self.nlp = spacy.load("en_core_med_sci_lg")
                logger.info("Loaded medical spaCy model: en_core_med_sci_lg")
            except OSError:
                # Fallback to base model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded base spaCy model: en_core_web_sm")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            self.nlp = None

    def preprocess_text(self, raw_text: str) -> StructuredNote:
        """
        Main preprocessing pipeline for clinical notes.
        """
        try:
            logger.info("Starting clinical note preprocessing")

            # Step 1: Basic text cleaning
            cleaned_text = self._clean_text(raw_text)

            # Step 2: Identify document type
            document_type = self._identify_document_type(cleaned_text)

            # Step 3: Extract sections (SOAP format, etc.)
            sections = self._extract_sections(cleaned_text, document_type)

            # Step 4: Medical entity recognition
            entities = self._extract_medical_entities(cleaned_text)

            # Step 5: Calculate quality score
            quality_score = self._calculate_quality_score(cleaned_text, sections, entities)

            # Step 6: Generate metadata
            metadata = self._generate_metadata(cleaned_text, sections, entities, document_type)

            structured_note = StructuredNote(
                original_text=raw_text,
                cleaned_text=cleaned_text,
                sections=sections,
                entities=entities,
                document_type=document_type,
                quality_score=quality_score,
                metadata=metadata
            )

            logger.info(f"Preprocessing completed. Document type: {document_type}, Quality: {quality_score:.2f}")
            return structured_note

        except Exception as e:
            logger.error(f"Error in clinical preprocessing: {e}")
            # Return minimal structure on error
            return StructuredNote(
                original_text=raw_text,
                cleaned_text=raw_text,
                sections={},
                entities=[],
                document_type="unknown",
                quality_score=0.0,
                metadata={"error": str(e)}
            )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common OCR errors
        text = text.replace('O', '0').replace('l', '1').replace('I', '1')

        # Remove excessive line breaks but keep paragraph structure
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Ensure double newlines for paragraphs
        text = re.sub(r'\n(?=[A-Z])', ' ', text)  # Join lines that start with capital (likely sentences)

        # Remove page numbers and headers/footers
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone numbers at end of lines

        # Remove artifacts but keep medical symbols
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]/\+\=\%\<\>\°]', ' ', text)

        # Expand common medical abbreviations
        text = self._expand_abbreviations(text)

        # Final cleanup
        text = ' '.join(text.split())  # Normalize spaces

        return text.strip()

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations."""
        # Create pattern for word boundaries to avoid partial matches
        for abbr, expansion in self.abbreviations.items():
            # Pattern matches abbreviation as whole word, case insensitive
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)

        return text

    def _identify_document_type(self, text: str) -> str:
        """Identify the type of medical document."""
        text_lower = text.lower()

        # Score each document type
        type_scores = {}
        for doc_type, keywords in self.document_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[doc_type] = score

        # Return the type with highest score
        if type_scores:
            max_score = max(type_scores.values())
            if max_score > 0:
                return max(type_scores, key=type_scores.get)

        return "general_note"

    def _extract_sections(self, text: str, document_type: str) -> Dict[str, str]:
        """Extract structured sections from medical notes."""
        sections = {}

        if document_type in ['consultation', 'initial_eval', 'progress_note']:
            # Try SOAP format first
            soap_sections = self._extract_soap_sections(text)
            if soap_sections:
                sections.update(soap_sections)

        # Extract common sections regardless of document type
        common_sections = {
            'history': self._extract_section_by_keywords(
                text, ['history', 'hpi', 'history of present illness', 'chief complaint']
            ),
            'examination': self._extract_section_by_keywords(
                text, ['examination', 'physical exam', 'pe', 'exam']
            ),
            'assessment': self._extract_section_by_keywords(
                text, ['assessment', 'impression', 'diagnosis', 'conclusion']
            ),
            'plan': self._extract_section_by_keywords(
                text, ['plan', 'treatment', 'recommendations', 'follow-up']
            ),
            'medications': self._extract_section_by_keywords(
                text, ['medications', 'meds', 'drugs', 'prescriptions']
            ),
            'vitals': self._extract_section_by_keywords(
                text, ['vitals', 'vital signs', 'bp', 'blood pressure', 'temperature']
            ),
            'labs': self._extract_section_by_keywords(
                text, ['lab', 'laboratory', 'test results', 'studies']
            )
        }

        # Add non-empty sections
        for section_name, content in common_sections.items():
            if content and len(content.strip()) > 20:  # Minimum content threshold
                sections[section_name] = content.strip()

        return sections

    def _extract_soap_sections(self, text: str) -> Dict[str, str]:
        """Extract SOAP format sections (Subjective, Objective, Assessment, Plan)."""
        soap_sections = {}

        # SOAP section patterns
        patterns = {
            'subjective': [r'subjective[:\s]*(.*?)(?=objective|$)', r's[:\s]*(.*?)(?=o[:\s]|objective|$)'],
            'objective': [r'objective[:\s]*(.*?)(?=assessment|a[:\s]|plan|$)', r'o[:\s]*(.*?)(?=a[:\s]|assessment|plan|$)'],
            'assessment': [r'assessment[:\s]*(.*?)(?=plan|p[:\s]|$)', r'a[:\s]*(.*?)(?=p[:\s]|plan|$)'],
            'plan': [r'plan[:\s]*(.*?)(?=$)', r'p[:\s]*(.*?)(?=$)']
        }

        for section, section_patterns in patterns.items():
            for pattern in section_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    if content:
                        soap_sections[section] = content
                        break

        return soap_sections

    def _extract_section_by_keywords(self, text: str, keywords: List[str]) -> Optional[str]:
        """Extract text section following specific keywords."""
        for keyword in keywords:
            # Look for keyword followed by content
            patterns = [
                rf'{re.escape(keyword)}[:\s\n]+(.*?)(?=\n\s*[A-Z][a-z]+\s*[:\n]|\Z)',
                rf'{re.escape(keyword)}\s*\n+(.*?)(?=\n\s*[A-Z][a-z]+\s*[:\n]|\Z)'
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    if len(content) > 20:  # Minimum content threshold
                        return content

        return None

    def _extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities using spaCy or rule-based methods."""
        entities = []

        if self.nlp:
            entities.extend(self._extract_entities_with_spacy(text))

        # Always add rule-based entities as backup/enhancement
        entities.extend(self._extract_entities_rule_based(text))

        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda e: e.start)

        return entities

    def _extract_entities_with_spacy(self, text: str) -> List[MedicalEntity]:
        """Extract entities using spaCy NLP model."""
        entities = []

        try:
            doc = self.nlp(text)

            for ent in doc.ents:
                # Map spaCy labels to medical categories
                medical_label = self._map_to_medical_label(ent.label_)

                if medical_label:
                    entity = MedicalEntity(
                        text=ent.text,
                        label=medical_label,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8,  # Default confidence for spaCy
                        category=self._get_entity_category(medical_label)
                    )
                    entities.append(entity)

        except Exception as e:
            logger.warning(f"spaCy entity extraction error: {e}")

        return entities

    def _extract_entities_rule_based(self, text: str) -> List[MedicalEntity]:
        """Extract entities using rule-based patterns."""
        entities = []

        # Medical procedure patterns
        procedure_patterns = [
            r'\b(examination|exam|x-ray|mri|ct scan|ultrasound|echocardiogram|ecg|ekg)\b',
            r'\b(biopsy|endoscopy|colonoscopy|bronchoscopy|laparoscopy)\b',
            r'\b(injection|vaccination|immunization|infusion|transfusion)\b',
            r'\b(surgery|operation|procedure|resection|repair|replacement)\b'
        ]

        # Medication patterns
        medication_patterns = [
            r'\b(aspirin|ibuprofen|acetaminophen|lipitor|metformin|insulin|lisinopril)\b',
            r'\b(antibiotic|antidepressant|antihypertensive|diuretic)\b'
        ]

        # Condition/diagnosis patterns
        condition_patterns = [
            r'\b(hypertension|diabetes|pneumonia|asthma|arthritis|depression)\b',
            r'\b(fracture|infection|inflammation|tumor|cancer)\b'
        ]

        # Extract entities for each pattern type
        pattern_types = [
            (procedure_patterns, 'PROCEDURE', 'procedure'),
            (medication_patterns, 'MEDICATION', 'medication'),
            (condition_patterns, 'CONDITION', 'diagnosis')
        ]

        for patterns, label, category in pattern_types:
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = MedicalEntity(
                        text=match.group(),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.6,  # Lower confidence for rule-based
                        category=category
                    )
                    entities.append(entity)

        return entities

    def _map_to_medical_label(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to medical categories."""
        mapping = {
            'PERSON': 'PATIENT',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'CARDINAL': 'NUMBER',
            'ORDINAL': 'ORDINAL',
            'MONEY': 'COST',
            'QUANTITY': 'MEASUREMENT',
            'EVENT': 'PROCEDURE'
        }

        return mapping.get(spacy_label)

    def _get_entity_category(self, label: str) -> str:
        """Get category for entity type."""
        categories = {
            'PATIENT': 'demographic',
            'PROCEDURE': 'procedure',
            'MEDICATION': 'medication',
            'CONDITION': 'diagnosis',
            'DATE': 'temporal',
            'TIME': 'temporal',
            'NUMBER': 'measurement',
            'MEASUREMENT': 'measurement',
            'LOCATION': 'location'
        }

        return categories.get(label, 'other')

    def _deduplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove duplicate entities, keeping highest confidence."""
        if not entities:
            return []

        # Sort by confidence (descending) then by start position
        sorted_entities = sorted(entities, key=lambda e: (e.confidence, e.start), reverse=True)

        unique_entities = []
        for entity in sorted_entities:
            # Check if entity overlaps with existing entities
            is_duplicate = False
            for existing in unique_entities:
                if (entity.text.lower() == existing.text.lower() or
                    (entity.start >= existing.start and entity.end <= existing.end)):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_entities.append(entity)

        return unique_entities

    def _calculate_quality_score(self, text: str, sections: Dict[str, str],
                                entities: List[MedicalEntity]) -> float:
        """Calculate quality score for processed medical note."""
        score = 0.0

        # Text length score (comprehensive notes are better)
        text_length = len(text)
        length_score = min(text_length / 2000, 1.0)  # Normalize to 0-1
        score += length_score * 0.2

        # Section completeness score
        expected_sections = ['history', 'examination', 'assessment', 'plan']
        section_score = sum(1 for section in expected_sections if section in sections) / len(expected_sections)
        score += section_score * 0.3

        # Entity density score (more medical entities = better content)
        entity_density = len(entities) / max(text_length / 100, 1)  # Entities per 100 chars
        entity_score = min(entity_density / 2, 1.0)  # Normalize to 0-1
        score += entity_score * 0.3

        # Medical terminology score
        medical_terms = ['patient', 'diagnosis', 'treatment', 'medication', 'examination', 'therapy']
        medical_count = sum(1 for term in medical_terms if term in text.lower())
        medical_score = min(medical_count / len(medical_terms), 1.0)
        score += medical_score * 0.2

        return min(score, 1.0)

    def _generate_metadata(self, text: str, sections: Dict[str, str],
                          entities: List[MedicalEntity], document_type: str) -> Dict[str, Any]:
        """Generate metadata for the processed note."""
        metadata = {
            'document_type': document_type,
            'total_sections': len(sections),
            'section_names': list(sections.keys()),
            'total_entities': len(entities),
            'entity_types': list(set(e.category for e in entities)),
            'text_length': len(text),
            'word_count': len(text.split()),
            'processing_timestamp': datetime.now().isoformat()
        }

        # Add section-specific metadata
        if 'vitals' in sections:
            metadata['has_vitals'] = True
            vitals_text = sections['vitals'].lower()
            metadata['vitals_found'] = {
                'blood_pressure': 'bp' in vitals_text or 'blood pressure' in vitals_text,
                'heart_rate': 'hr' in vitals_text or 'heart rate' in vitals_text,
                'temperature': 'temp' in vitals_text or 'temperature' in vitals_text,
                'respiratory_rate': 'rr' in vitals_text or 'respiratory' in vitals_text
            }

        # Add entity counts by category
        entity_counts = {}
        for entity in entities:
            entity_counts[entity.category] = entity_counts.get(entity.category, 0) + 1
        metadata['entity_counts'] = entity_counts

        return metadata