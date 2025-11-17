"""
PDF validator for medical records.
Validates that documents contain legitimate medical record content.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .preprocessor import ClinicalPreprocessor, StructuredNote
from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of medical record validation."""
    is_valid: bool
    confidence_score: float
    validation_factors: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]

class MedicalRecordValidator:
    """Validates medical record content and structure."""

    def __init__(self, config: Config):
        """Initialize medical record validator."""
        self.config = config
        self.preprocessor = ClinicalPreprocessor(config)

        # Medical record indicators
        self.medical_indicators = {
            'patient_identifiers': [
                r'\bpatient\s*name\b', r'\bname\s*:\s*[A-Z][a-z]+\s+[A-Z][a-z]+',
                r'\bmr[n]?\s*\d+', r'\bmedical\s*record\s*#\d+',
                r'\bpatient\s*id\b', r'\bchart\s*#\d+',
                r'\bdob\s*[:\s]', r'\bdate\s*of\s*birth'
            ],
            'provider_identifiers': [
                r'\bphysician\s*name\b', r'\bdoctor\s*[:\s]', r'\bdr\s*\.?\s*[A-Z]',
                r'\bprovider\s*name\b', r'\bsignature\s*[:\s]',
                r'\bnurse\s*name\b', r'\bclinician\s*[:\s]'
            ],
            'medical_content': [
                r'\bchief\s*complaint\b', r'\bsymptoms?\b', r'\bpresent\s*illness\b',
                r'\bpast\s*medical\s*history\b', r'\bpmh\b',
                r'\bmedications?\b', r'\bdrugs?\b', r'\bprescriptions?\b',
                r'\ballerg(y|ies)\b', r'\badverse\s*reactions?\b',
                r'\bvital\s*signs?\b', r'\bbp\b', r'\bblood\s*pressure\b',
                r'\btemperature\b', r'\bheart\s*rate\b', r'\brespiratory\b',
                r'\bexamination\b', r'\bexam\b', r'\bphysical\s*exam\b',
                r'\bassessment\b', r'\bdiagnosis\b', r'\bimpression\b',
                r'\btreatment\s*plan\b', r'\bplan\b', r'\btherapy\b',
                r'\bprocedure\b', r'\bsurgery\b', r'\boperation\b',
                r'\blab(oratory)?\s*results?\b', r'\bstudies?\b', r'\btests?\b'
            ],
            'document_structure': [
                r'\bs(?:ubjective)?[:\s]', r'\bo(?:bjective)?[:\s]',
                r'\ba(?:ssessment)?[:\s]', r'\bp(?:lan)?[:\s]',
                r'\bsoap\b', r'\bhpi\b', r'\bhistory\s*of\s*present\s*illness\b'
            ]
        }

        # Non-medical indicators (red flags)
        self.non_medical_indicators = [
            r'\breceipt\b', r'\binvoice\b', r'\bbill\b', r'\bpayment\b',
            r'\bcontract\b', r'\bagreement\b', r'\blegal\s*document\b',
            r'\bfinancial\s*statement\b', r'\bbank\s*account\b',
            r'\bcredit\s*card\b', r'\bssn\b', r'\bsocial\s*security',
            r'\bdriver\'s\s*license\b', r'\bpassport\b'
        ]

        # Medical specialties and contexts
        self.medical_specialties = [
            'cardiology', 'dermatology', 'endocrinology', 'gastroenterology',
            'neurology', 'oncology', 'pediatrics', 'psychiatry',
            'radiology', 'surgery', 'urology', 'orthopedics',
            'family medicine', 'internal medicine', 'emergency medicine'
        ]

    def validate_medical_record(self, text: str) -> ValidationResult:
        """
        Validate that text appears to be a legitimate medical record.
        """
        try:
            logger.info("Starting medical record validation")

            # Initialize validation result
            validation_factors = {}
            warnings = []
            errors = []
            recommendations = []

            # Step 1: Basic content validation
            content_score, content_issues = self._validate_medical_content(text)
            validation_factors['content_score'] = content_score
            warnings.extend(content_issues)

            # Step 2: Structure validation
            structure_score, structure_issues = self._validate_document_structure(text)
            validation_factors['structure_score'] = structure_score
            warnings.extend(structure_issues)

            # Step 3: Entity validation
            entity_score, entity_issues = self._validate_medical_entities(text)
            validation_factors['entity_score'] = entity_score
            warnings.extend(entity_issues)

            # Step 4: Red flag detection
            red_flag_score, red_flag_errors = self._detect_red_flags(text)
            validation_factors['red_flag_score'] = red_flag_score
            errors.extend(red_flag_errors)

            # Step 5: Quality assessment
            quality_score, quality_issues = self._assess_document_quality(text)
            validation_factors['quality_score'] = quality_score
            warnings.extend(quality_issues)

            # Calculate overall confidence
            overall_score = self._calculate_overall_score(validation_factors)

            # Determine if document is valid
            is_valid = overall_score >= 0.5 and len(errors) == 0

            # Generate recommendations
            recommendations = self._generate_recommendations(validation_factors, warnings, errors)

            result = ValidationResult(
                is_valid=is_valid,
                confidence_score=overall_score,
                validation_factors=validation_factors,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations
            )

            logger.info(f"Validation completed. Valid: {is_valid}, Score: {overall_score:.2f}")
            return result

        except Exception as e:
            logger.error(f"Error in medical record validation: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_factors={},
                warnings=[],
                errors=[f"Validation error: {str(e)}"],
                recommendations=["Document could not be validated due to processing error"]
            )

    def _validate_medical_content(self, text: str) -> Tuple[float, List[str]]:
        """Validate medical content indicators."""
        score = 0.0
        issues = []
        text_lower = text.lower()

        # Check for patient identifiers
        patient_found = any(re.search(pattern, text, re.IGNORECASE)
                          for pattern in self.medical_indicators['patient_identifiers'])
        if patient_found:
            score += 0.2
        else:
            issues.append("No clear patient identifiers found")

        # Check for provider identifiers
        provider_found = any(re.search(pattern, text, re.IGNORECASE)
                           for pattern in self.medical_indicators['provider_identifiers'])
        if provider_found:
            score += 0.15
        else:
            issues.append("No clear provider information found")

        # Check for medical terminology
        medical_terms_found = 0
        for pattern in self.medical_indicators['medical_content']:
            if re.search(pattern, text, re.IGNORECASE):
                medical_terms_found += 1

        medical_score = min(medical_terms_found / 10, 1.0) * 0.4  # Max 0.4 points
        score += medical_score
        if medical_terms_found < 3:
            issues.append("Limited medical terminology detected")

        # Check for medical specialties
        specialty_found = any(specialty in text_lower for specialty in self.medical_specialties)
        if specialty_found:
            score += 0.1
        else:
            issues.append("No medical specialty context identified")

        # Additional medical term density check
        word_count = len(text.split())
        medical_word_count = sum(1 for word in text_lower.split()
                                if any(term in word for term in [
                                    'patient', 'doctor', 'nurse', 'hospital', 'clinic',
                                    'diagnosis', 'treatment', 'medication', 'symptom',
                                    'examination', 'therapy', 'surgery'
                                ]))

        if word_count > 0:
            medical_density = medical_word_count / word_count
            if medical_density > 0.05:  # 5% medical terms
                score += 0.15

        return min(score, 1.0), issues

    def _validate_document_structure(self, text: str) -> Tuple[float, List[str]]:
        """Validate document structure for medical records."""
        score = 0.0
        issues = []

        # Check for SOAP structure
        soap_components = ['subjective', 'objective', 'assessment', 'plan']
        soap_found = 0
        for component in soap_components:
            if re.search(rf'\b{component}\b', text, re.IGNORECASE):
                soap_found += 1

        if soap_found >= 3:
            score += 0.4
        elif soap_found >= 2:
            score += 0.2
        else:
            issues.append("Document lacks clear medical structure (SOAP format)")

        # Check for section headers
        section_patterns = [r'\b[A-Z][a-z]+\s*:', r'\b[A-Z]{2,}\s*:', r'\n\s*[A-Z][a-z]+\s*\n']
        section_count = sum(len(re.findall(pattern, text)) for pattern in section_patterns)

        if section_count >= 3:
            score += 0.3
        elif section_count >= 1:
            score += 0.15
        else:
            issues.append("Document lacks clear section organization")

        # Check for date stamps (common in medical records)
        date_patterns = [r'\d{1,2}/\d{1,2}/\d{2,4}', r'\d{4}-\d{2}-\d{2}',
                        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}']
        dates_found = any(re.search(pattern, text) for pattern in date_patterns)

        if dates_found:
            score += 0.2
        else:
            issues.append("No clear date information found")

        # Check for appropriate length (medical records are usually comprehensive)
        word_count = len(text.split())
        if 100 <= word_count <= 2000:  # Reasonable length for medical notes
            score += 0.1
        elif word_count < 50:
            issues.append("Document appears too short for a medical record")
        elif word_count > 5000:
            issues.append("Document unusually long for a typical medical note")

        return min(score, 1.0), issues

    def _validate_medical_entities(self, text: str) -> Tuple[float, List[str]]:
        """Validate presence of medical entities."""
        score = 0.0
        issues = []

        try:
            # Use preprocessor to extract entities
            structured_note = self.preprocessor.preprocess_text(text)
            entities = structured_note.entities

            # Count entities by category
            entity_categories = {}
            for entity in entities:
                entity_categories[entity.category] = entity_categories.get(entity.category, 0) + 1

            # Check for procedure-related entities
            procedure_entities = entity_categories.get('procedure', 0)
            if procedure_entities >= 2:
                score += 0.4
            elif procedure_entities >= 1:
                score += 0.2
            else:
                issues.append("Limited procedure-related content found")

            # Check for diagnosis/condition entities
            diagnosis_entities = entity_categories.get('diagnosis', 0)
            if diagnosis_entities >= 1:
                score += 0.3
            else:
                issues.append("No clear diagnosis or condition information")

            # Check for medication entities
            medication_entities = entity_categories.get('medication', 0)
            if medication_entities >= 1:
                score += 0.2

            # Check for demographic entities
            demographic_entities = entity_categories.get('demographic', 0)
            if demographic_entities >= 1:
                score += 0.1

            # Bonus for diverse entity types
            entity_types = len(entity_categories)
            if entity_types >= 3:
                score += 0.2

        except Exception as e:
            logger.warning(f"Entity validation error: {e}")
            issues.append("Could not validate medical entities due to processing error")

        return min(score, 1.0), issues

    def _detect_red_flags(self, text: str) -> Tuple[float, List[str]]:
        """Detect non-medical content that should be flagged."""
        score = 1.0  # Start with perfect score
        errors = []

        text_lower = text.lower()

        # Check for non-medical red flags
        for pattern in self.non_medical_indicators:
            if re.search(pattern, text_lower):
                score -= 0.3
                errors.append(f"Document contains non-medical content: {pattern}")

        # Check for patterns that suggest non-medical documents
        suspicious_patterns = [
            r'\b\d{4}\s*-\s*\d{4}\s*-\s*\d{4}\s*-\s*\d{4}\b',  # Credit card pattern
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\baccount\s*#?\s*\d+', r'\bbalance\s*due\b', r'\bpayment\s*received\b'
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, text_lower):
                score -= 0.4
                errors.append(f"Suspicious content pattern detected: {pattern}")

        # Check for structure that suggests non-medical document
        if re.search(r'\btotal\s*:\s*\$?\d+', text_lower):
            score -= 0.3
            errors.append("Document appears to contain financial totals")

        # Ensure score doesn't go below 0
        return max(score, 0.0), errors

    def _assess_document_quality(self, text: str) -> Tuple[float, List[str]]:
        """Assess overall document quality."""
        score = 0.0
        issues = []

        # Text quality indicators
        if len(text.strip()) > 0:
            score += 0.2

        # Check for readable text (not just numbers/symbols)
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio > 0.5:
            score += 0.2
        elif alpha_ratio < 0.2:
            issues.append("Document contains very little readable text")

        # Check for sentence structure
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(complete_sentences) >= 3:
            score += 0.3
        elif len(complete_sentences) < 1:
            issues.append("Document lacks proper sentence structure")

        # Check for medical abbreviation usage
        common_abbr = ['bp', 'hr', 'rr', 'temp', 'hpi', 'pmh', 'meds', 'pt']
        abbr_count = sum(1 for abbr in common_abbr if abbr in text.lower())

        if abbr_count >= 2:
            score += 0.3  # Indicates medical context

        return min(score, 1.0), issues

    def _calculate_overall_score(self, validation_factors: Dict[str, float]) -> float:
        """Calculate overall validation score from individual factors."""
        weights = {
            'content_score': 0.35,
            'structure_score': 0.25,
            'entity_score': 0.25,
            'quality_score': 0.15
        }

        total_score = 0.0
        total_weight = 0.0

        for factor, weight in weights.items():
            if factor in validation_factors:
                total_score += validation_factors[factor] * weight
                total_weight += weight

        # Red flag score is a penalty
        if 'red_flag_score' in validation_factors:
            total_score *= validation_factors['red_flag_score']

        return total_score / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(self, validation_factors: Dict[str, float],
                                warnings: List[str], errors: List[str]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if errors:
            recommendations.append("Document contains elements that suggest it may not be a medical record")
            recommendations.append("Please verify the document content and source")

        # Content-based recommendations
        content_score = validation_factors.get('content_score', 0)
        if content_score < 0.5:
            recommendations.append("Consider adding more medical terminology and clinical information")
            recommendations.append("Include clear patient and provider identification")

        # Structure-based recommendations
        structure_score = validation_factors.get('structure_score', 0)
        if structure_score < 0.5:
            recommendations.append("Organize content using standard medical note structure (SOAP format)")
            recommendations.append("Include clear section headers and date information")

        # Quality-based recommendations
        quality_score = validation_factors.get('quality_score', 0)
        if quality_score < 0.5:
            recommendations.append("Ensure document has adequate text content and proper sentence structure")

        # Entity-based recommendations
        entity_score = validation_factors.get('entity_score', 0)
        if entity_score < 0.3:
            recommendations.append("Document may lack specific medical information needed for coding")

        return recommendations