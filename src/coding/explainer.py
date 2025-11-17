"""
SHAP-based explanation engine for CPT code recommendations.
Provides model interpretability and feature importance explanations.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Try to import SHAP for explanations
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available, using simplified explanations")

from ..utils.config import Config
from ..pdf.preprocessor import ClinicalPreprocessor, MedicalEntity

logger = logging.getLogger(__name__)

@dataclass
class FeatureExplanation:
    """Explanation for a single feature contribution."""
    feature_name: str
    feature_value: str
    contribution: float
    contribution_type: str  # 'positive' or 'negative'
    explanation: str

@dataclass
class CPTExplanation:
    """Complete explanation for a CPT code recommendation."""
    cpt_code: str
    description: str
    overall_confidence: float
    key_factors: List[FeatureExplanation]
    supporting_text_snippets: List[Tuple[str, float]]
    model_contributions: Dict[str, float]
    reasoning_summary: str

class SHAPExplainer:
    """SHAP-based explainer for CPT code predictions."""

    def __init__(self, config: Config, classifier):
        """Initialize SHAP explainer."""
        self.config = config
        self.classifier = classifier
        self.preprocessor = ClinicalPreprocessor(config)

        # SHAP explainers
        self.rf_explainer = None
        self.svm_explainer = None
        self.gb_explainer = None

        # Feature names for explanations
        self.feature_names = [
            'text_length', 'word_count', 'sentence_count',
            'patient_mentions', 'diagnosis_mentions', 'treatment_mentions',
            'medication_mentions', 'examination_mentions', 'procedure_mentions',
            'therapy_mentions', 'symptom_mentions', 'assessment_mentions', 'plan_mentions',
            'history_section', 'examination_section', 'assessment_section',
            'plan_section', 'medications_section',
            'procedure_entities', 'medication_entities', 'diagnosis_entities', 'measurement_entities'
        ]

        # Medical term mappings
        self.medical_term_explanations = {
            'examination': 'Physical examination findings indicate evaluation service',
            'procedure': 'Documented procedures support specific CPT billing',
            'diagnosis': 'Documented diagnoses support medical necessity',
            'medication': 'Medication management indicates clinical complexity',
            'treatment': 'Treatment planning supports higher level of service',
            'symptom': 'Symptom documentation supports evaluation complexity',
            'assessment': 'Assessment findings indicate medical decision making',
            'therapy': 'Therapeutic interventions support procedure billing'
        }

        # Initialize explainers
        self._initialize_explainers()

    def _initialize_explainers(self):
        """Initialize SHAP explainers for each model."""
        try:
            if SHAP_AVAILABLE and self.classifier.random_forest:
                # Use TreeExplainer for Random Forest
                self.rf_explainer = shap.TreeExplainer(self.classifier.random_forest)
                logger.info("Initialized SHAP TreeExplainer for Random Forest")

            if SHAP_AVAILABLE and self.classifier.gradient_boosting:
                # Use TreeExplainer for Gradient Boosting
                self.gb_explainer = shap.TreeExplainer(self.classifier.gradient_boosting)
                logger.info("Initialized SHAP TreeExplainer for Gradient Boosting")

            if SHAP_AVAILABLE and self.classifier.svm:
                # Use KernelExplainer for SVM
                # We'll need background data for this
                self._create_svm_explainer()

        except Exception as e:
            logger.warning(f"Error initializing SHAP explainers: {e}")
            SHAP_AVAILABLE = False

    def _create_svm_explainer(self):
        """Create SVM explainer with background data."""
        # This would need background training data
        # For now, we'll skip SVM explainer
        pass

    def explain_prediction(self, text: str, structured_note: Any,
                          cpt_code: str, confidence: float) -> CPTExplanation:
        """
        Generate explanation for CPT code prediction.
        """
        try:
            logger.info(f"Generating explanation for CPT code: {cpt_code}")

            # Get feature contributions
            feature_explanations = self._get_feature_contributions(text, structured_note, cpt_code)

            # Find supporting text snippets
            supporting_snippets = self._find_supporting_text_snippets(text, cpt_code, feature_explanations)

            # Get model contributions
            model_contributions = self._get_model_contributions(cpt_code)

            # Generate reasoning summary
            reasoning_summary = self._generate_reasoning_summary(
                cpt_code, confidence, feature_explanations, supporting_snippets
            )

            # Get CPT description
            description = self.classifier.cpt_mapping.get(cpt_code, {}).get('description', '')

            explanation = CPTExplanation(
                cpt_code=cpt_code,
                description=description,
                overall_confidence=confidence,
                key_factors=feature_explanations,
                supporting_text_snippets=supporting_snippets,
                model_contributions=model_contributions,
                reasoning_summary=reasoning_summary
            )

            logger.info(f"Explanation generated for {cpt_code}")
            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Return basic explanation on error
            return self._create_fallback_explanation(cpt_code, confidence)

    def _get_feature_contributions(self, text: str, structured_note: Any,
                                 cpt_code: str) -> List[FeatureExplanation]:
        """Get feature contributions for the prediction."""
        contributions = []

        try:
            # Extract features
            features = self.classifier._extract_features(text, structured_note)

            if SHAP_AVAILABLE and self.rf_explainer:
                # Get SHAP values from Random Forest
                shap_values = self.rf_explainer.shap_values(features)

                # Handle binary/multi-class cases
                if isinstance(shap_values, list):
                    # Multi-class case - get values for predicted class
                    if hasattr(self.classifier, 'label_encoder'):
                        try:
                            class_idx = np.where(self.classifier.label_encoder.classes_ == cpt_code)[0]
                            if len(class_idx) > 0:
                                shap_values_for_class = shap_values[class_idx[0]]
                            else:
                                shap_values_for_class = shap_values[0]
                        except:
                            shap_values_for_class = shap_values[0]
                    else:
                        shap_values_for_class = shap_values[0]
                else:
                    shap_values_for_class = shap_values

                # Create feature explanations
                for i, (feature_value, shap_val) in enumerate(zip(features[0], shap_values_for_class[0])):
                    if i < len(self.feature_names):
                        feature_name = self.feature_names[i]
                        contribution_type = 'positive' if shap_val > 0 else 'negative'

                        explanation = self._explain_feature(feature_name, feature_value, shap_val, text)

                        feature_explanation = FeatureExplanation(
                            feature_name=feature_name,
                            feature_value=str(feature_value),
                            contribution=float(shap_val),
                            contribution_type=contribution_type,
                            explanation=explanation
                        )
                        contributions.append(feature_explanation)

            else:
                # Fallback to feature importance based explanation
                contributions = self._create_fallback_contributions(text, structured_note)

            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

            # Keep top contributing features
            return contributions[:8]

        except Exception as e:
            logger.error(f"Error getting feature contributions: {e}")
            return self._create_fallback_contributions(text, structured_note)

    def _explain_feature(self, feature_name: str, feature_value: Any,
                        shap_value: float, text: str) -> str:
        """Generate explanation for a specific feature."""
        abs_shap = abs(shap_value)

        explanations = {
            'text_length': f"Document length ({int(feature_value)} chars) {'supports' if shap_value > 0 else 'reduces'} confidence for comprehensive evaluation",
            'word_count': f"Word count ({int(feature_value)} words) {'indicates' if shap_value > 0 else 'lacks'} detail needed for this code level",
            'sentence_count': f"Sentence structure ({int(feature_value)} sentences) {'shows' if shap_value > 0 else 'lacks'} comprehensive documentation",
            'patient_mentions': f"Patient terminology {'presence' if feature_value > 0 else 'absence'} {'supports' if shap_value > 0 else 'reduces'} patient care coding",
            'diagnosis_mentions': f"Diagnosis documentation ({int(feature_value)} mentions) {'supports' if shap_value > 0 else 'lacks'} medical necessity",
            'treatment_mentions': f"Treatment planning ({int(feature_value)} mentions) {'indicates' if shap_value > 0 else 'suggests'} higher complexity",
            'medication_mentions': f"Medication management ({int(feature_value)} mentions) {'supports' if shap_value > 0 else 'lacks'} pharmacologic complexity",
            'examination_mentions': f"Physical exam findings ({int(feature_value)} mentions) {'support' if shap_value > 0 else 'lack'} evaluation codes",
            'procedure_mentions': f"Procedure documentation ({int(feature_value)} mentions) {'directly supports' if shap_value > 0 else 'doesn\'t support'} procedural coding",
            'therapy_mentions': f"Therapy references ({int(feature_value)} mentions) {'support' if shap_value > 0 else 'lack'} therapeutic interventions",
            'symptom_mentions': f"Symptom documentation ({int(feature_value)} mentions) {'supports' if shap_value > 0 else 'lacks'} evaluation complexity",
            'assessment_mentions': f"Assessment findings ({int(feature_value)} mentions) {'support' if shap_value > 0 else 'lack'} medical decision making",
            'plan_mentions': f"Treatment plan ({int(feature_value)} mentions) {'supports' if shap_value > 0 else 'lacks'} care management",
            'history_section': f"History section {'present' if feature_value > 0 else 'absent'} {'supports' if shap_value > 0 else 'reduces'} comprehensive evaluation",
            'examination_section': f"Examination section {'present' if feature_value > 0 else 'absent'} {'supports' if shap_value > 0 else 'lacks'} physical exam coding",
            'assessment_section': f"Assessment section {'present' if feature_value > 0 else 'absent'} {'supports' if shap_value > 0 else 'lacks'} diagnostic coding",
            'plan_section': f"Plan section {'present' if feature_value > 0 else 'absent'} {'supports' if shap_value > 0 else 'lacks'} management coding",
            'medications_section': f"Medications section {'present' if feature_value > 0 else 'absent'} {'supports' if shap_value > 0 else 'lacks'} pharmacologic management",
            'procedure_entities': f"Procedure entities ({int(feature_value)}) {'strongly support' if shap_value > 0 else 'don\'t support'} procedural coding",
            'medication_entities': f"Medication entities ({int(feature_value)}) {'support' if shap_value > 0 else 'lack'} medication management coding",
            'diagnosis_entities': f"Diagnosis entities ({int(feature_value)}) {'support' if shap_value > 0 else 'lack'} diagnostic coding",
            'measurement_entities': f"Measurement entities ({int(feature_value)}) {'support' if shap_value > 0 else 'lack'} clinical complexity"
        }

        base_explanation = explanations.get(feature_name, f"Feature {feature_name} contributes {abs_shap:.3f} to prediction")

        if abs_shap > 0.1:
            importance_qualifier = "Strongly " if abs_shap > 0.3 else "Moderately "
            base_explanation = importance_qualifier + base_explanation.lower()

        return base_explanation

    def _create_fallback_contributions(self, text: str, structured_note: Any) -> List[FeatureExplanation]:
        """Create fallback feature contributions when SHAP is not available."""
        contributions = []
        text_lower = text.lower()

        # Analyze medical content
        medical_indicators = {
            'procedure_content': len([t for t in self.preprocessor._extract_entities_rule_based(text) if t.category == 'procedure']),
            'medication_content': len([t for t in self.preprocessor._extract_entities_rule_based(text) if t.category == 'medication']),
            'diagnosis_content': len([t for t in self.preprocessor._extract_entities_rule_based(text) if t.category == 'diagnosis']),
            'comprehensive_length': len(text.split()) > 200,
            'structured_format': any(section in structured_note.sections for section in ['assessment', 'plan'])
        }

        # Create contributions based on medical content
        for indicator, value in medical_indicators.items():
            if value > 0:
                if isinstance(value, bool):
                    contribution = 0.2 if value else -0.1
                else:
                    contribution = min(value * 0.1, 0.5)

                contribution_type = 'positive' if contribution > 0 else 'negative'

                explanation = self._get_fallback_explanation(indicator, value, text_lower)

                feature_explanation = FeatureExplanation(
                    feature_name=indicator,
                    feature_value=str(value),
                    contribution=contribution,
                    contribution_type=contribution_type,
                    explanation=explanation
                )
                contributions.append(feature_explanation)

        return contributions

    def _get_fallback_explanation(self, indicator: str, value: Any, text: str) -> str:
        """Get explanation for fallback feature analysis."""
        explanations = {
            'procedure_content': f"Procedure terminology detected {'supports procedural coding' if value > 0 else 'lacks procedural evidence'}",
            'medication_content': f"Medication references {'support pharmacologic management' if value > 0 else 'indicate limited medical therapy'}",
            'diagnosis_content': f"Diagnosis documentation {'supports medical necessity' if value > 0 else 'lacks diagnostic clarity'}",
            'comprehensive_length': f"Document length {'supports comprehensive evaluation' if value else 'may indicate limited documentation'}",
            'structured_format': f"Structured note format {'supports organized clinical documentation' if value else 'lacks standard medical structure'}"
        }

        return explanations.get(indicator, f"Feature {indicator} value: {value}")

    def _find_supporting_text_snippets(self, text: str, cpt_code: str,
                                     feature_contributions: List[FeatureExplanation]) -> List[Tuple[str, float]]:
        """Find text snippets that support the prediction."""
        snippets = []

        try:
            # Get relevant medical terms for the CPT code
            relevant_terms = self._get_relevant_terms_for_cpt(cpt_code)

            # Find sentences containing relevant terms
            sentences = text.split('.')

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue

                # Calculate relevance score
                relevance = 0
                for term in relevant_terms:
                    if term.lower() in sentence.lower():
                        relevance += 1

                if relevance > 0:
                    # Limit snippet length
                    if len(sentence) > 200:
                        sentence = sentence[:200] + "..."

                    confidence = min(relevance / len(relevant_terms), 1.0)
                    snippets.append((sentence, confidence))

            # Sort by confidence and keep top snippets
            snippets.sort(key=lambda x: x[1], reverse=True)
            return snippets[:5]

        except Exception as e:
            logger.error(f"Error finding supporting snippets: {e}")
            return []

    def _get_relevant_terms_for_cpt(self, cpt_code: str) -> List[str]:
        """Get medically relevant terms for a CPT code."""
        cpt_info = self.classifier.cpt_mapping.get(cpt_code, {})
        category = cpt_info.get('category', '').lower()
        description = cpt_info.get('description', '').lower()

        # Base terms by category
        category_terms = {
            'evaluation': ['examination', 'evaluation', 'assessment', 'history', 'physical exam'],
            'laboratory': ['lab', 'test', 'result', 'blood', 'urine', 'analysis'],
            'diagnostic': ['x-ray', 'mri', 'ct', 'ultrasound', 'diagnostic', 'imaging'],
            'radiology': ['radiology', 'x-ray', 'imaging', 'diagnostic', 'study'],
            'procedure': ['procedure', 'surgery', 'operation', 'treatment', 'intervention'],
            'medication': ['medication', 'drug', 'prescription', 'pharmaceutical', 'therapy']
        }

        relevant_terms = category_terms.get(category, [])

        # Add terms from description
        desc_words = description.split()
        relevant_terms.extend([word for word in desc_words if len(word) > 3])

        # Add common medical terms
        relevant_terms.extend(['patient', 'diagnosis', 'treatment', 'symptoms'])

        return list(set(relevant_terms))

    def _get_model_contributions(self, cpt_code: str) -> Dict[str, float]:
        """Get contribution of each model to the final prediction."""
        # This would normally track which models contributed to each prediction
        # For now, return the configured weights
        return self.classifier.model_weights.copy()

    def _generate_reasoning_summary(self, cpt_code: str, confidence: float,
                                  feature_contributions: List[FeatureExplanation],
                                  supporting_snippets: List[Tuple[str, float]]) -> str:
        """Generate a human-readable reasoning summary."""
        try:
            cpt_info = self.classifier.cpt_mapping.get(cpt_code, {})
            description = cpt_info.get('description', 'CPT procedure')

            summary_parts = [f"CPT code {cpt_code} ({description}) recommended with {confidence:.1%} confidence."]

            # Add key contributing factors
            positive_factors = [f for f in feature_contributions[:3] if f.contribution_type == 'positive']
            if positive_factors:
                factors_text = ", ".join([f.explanation for f in positive_factors])
                summary_parts.append(f"Key supporting factors: {factors_text}")

            # Add supporting evidence
            if supporting_snippets:
                summary_parts.append("Supporting documentation found in clinical notes.")

            # Add review recommendation
            if confidence < 0.7:
                summary_parts.append("Professional coder review recommended due to moderate confidence.")

            return " ".join(summary_parts)

        except Exception as e:
            logger.error(f"Error generating reasoning summary: {e}")
            return f"CPT code {cpt_code} recommended based on clinical documentation analysis."

    def _create_fallback_explanation(self, cpt_code: str, confidence: float) -> CPTExplanation:
        """Create fallback explanation when detailed analysis fails."""
        description = self.classifier.cpt_mapping.get(cpt_code, {}).get('description', '')

        return CPTExplanation(
            cpt_code=cpt_code,
            description=description,
            overall_confidence=confidence,
            key_factors=[],
            supporting_text_snippets=[],
            model_contributions={},
            reasoning_summary=f"CPT code {cpt_code} recommended with {confidence:.1%} confidence based on medical content analysis."
        )

    def generate_explanation_report(self, explanations: List[CPTExplanation]) -> str:
        """Generate a comprehensive explanation report for multiple CPT codes."""
        if not explanations:
            return "No explanations available."

        report_parts = ["CPT Code Recommendations - Explanation Report", "=" * 60, ""]

        for i, explanation in enumerate(explanations, 1):
            report_parts.extend([
                f"{i}. CPT Code: {explanation.cpt_code} - {explanation.description}",
                f"   Confidence: {explanation.overall_confidence:.1%}",
                "",
                "   Reasoning Summary:",
                f"   {explanation.reasoning_summary}",
                ""
            ])

            if explanation.key_factors:
                report_parts.append("   Key Contributing Factors:")
                for factor in explanation.key_factors[:3]:  # Top 3 factors
                    direction = "↑" if factor.contribution_type == 'positive' else "↓"
                    report_parts.append(f"   {direction} {factor.explanation}")
                report_parts.append("")

            if explanation.supporting_text_snippets:
                report_parts.append("   Supporting Documentation:")
                for snippet, _ in explanation.supporting_text_snippets[:2]:  # Top 2 snippets
                    report_parts.append(f"   • \"{snippet}\"")
                report_parts.append("")

            report_parts.append("-" * 40)

        return "\n".join(report_parts)