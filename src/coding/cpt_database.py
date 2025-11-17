"""
CPT code database and lookup functionality.
Provides local 2021 CPT code information with search and categorization.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import re

from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class CPTCode:
    """Represents a CPT code with associated information."""
    code: str
    description: str
    category: str
    subcategory: str
    description_long: str
    typical_usage: str
    documentation_requirements: List[str]
    bundling_rules: List[str]
    frequency: float  # How commonly this code is used
    complexity_level: int  # 1-5 complexity rating
    modifiers: List[str]  # Common modifiers
    rvu: float  # Relative Value Unit (if available)

class CPTDatabase:
    """Local CPT code database for medical coding."""

    def __init__(self, config: Config):
        """Initialize CPT database."""
        self.config = config
        self.cpt_file = config.get_cpt_codes_path()
        self.cpt_codes = {}
        self.category_index = {}
        self.search_index = {}

        # Load CPT codes
        self._load_cpt_codes()

    def _load_cpt_codes(self):
        """Load CPT codes from file or create sample data."""
        try:
            if Path(self.cpt_file).exists():
                with open(self.cpt_file, 'r') as f:
                    data = json.load(f)
                    for code_data in data:
                        cpt_code = self._create_cpt_object(code_data)
                        self.cpt_codes[cpt_code.code] = cpt_code
                logger.info(f"Loaded {len(self.cpt_codes)} CPT codes from database")
            else:
                self._create_sample_database()
                logger.info("Created sample CPT code database")

        except Exception as e:
            logger.error(f"Error loading CPT database: {e}")
            self._create_sample_database()

        # Build search indexes
        self._build_indexes()

    def _create_cpt_object(self, data: Dict[str, Any]) -> CPTCode:
        """Create CPTCode object from data dictionary."""
        return CPTCode(
            code=data.get('code', ''),
            description=data.get('description', ''),
            category=data.get('category', 'Other'),
            subcategory=data.get('subcategory', ''),
            description_long=data.get('description_long', data.get('description', '')),
            typical_usage=data.get('typical_usage', ''),
            documentation_requirements=data.get('documentation_requirements', []),
            bundling_rules=data.get('bundling_rules', []),
            frequency=data.get('frequency', 0.1),
            complexity_level=data.get('complexity_level', 3),
            modifiers=data.get('modifiers', []),
            rvu=data.get('rvu', 0.0)
        )

    def _create_sample_database(self):
        """Create sample CPT database for demonstration."""
        sample_codes = [
            {
                "code": "99201",
                "description": "Office visit, new patient",
                "category": "Evaluation",
                "subcategory": "Office/Other Outpatient Services",
                "description_long": "Office or other outpatient visit for the evaluation and management of a new patient, which requires these 3 key components: A problem focused history; A problem focused examination; Straightforward medical decision making.",
                "typical_usage": "Initial patient consultation or first visit",
                "documentation_requirements": [
                    "Chief complaint",
                    "History of present illness",
                    "Problem focused examination",
                    "Medical decision making documentation"
                ],
                "bundling_rules": [],
                "frequency": 0.8,
                "complexity_level": 1,
                "modifiers": ["25", "57"],
                "rvu": 1.24
            },
            {
                "code": "99202",
                "description": "Office visit, new patient, low complexity",
                "category": "Evaluation",
                "subcategory": "Office/Other Outpatient Services",
                "description_long": "Office or other outpatient visit for the evaluation and management of a new patient, which requires these 3 key components: An expanded problem focused history; An expanded problem focused examination; Medical decision making of low complexity.",
                "typical_usage": "New patient visit with straightforward presentation",
                "documentation_requirements": [
                    "Chief complaint",
                    "Extended history of present illness",
                    "Problem pertinent system review",
                    "Expanded problem focused examination",
                    "Low complexity medical decision making"
                ],
                "bundling_rules": [],
                "frequency": 0.7,
                "complexity_level": 2,
                "modifiers": ["25", "57"],
                "rvu": 2.11
            },
            {
                "code": "99203",
                "description": "Office visit, new patient, moderate complexity",
                "category": "Evaluation",
                "subcategory": "Office/Other Outpatient Services",
                "description_long": "Office or other outpatient visit for the evaluation and management of a new patient, which requires these 3 key components: A detailed history; A detailed examination; Medical decision making of moderate complexity.",
                "typical_usage": "Comprehensive new patient evaluation",
                "documentation_requirements": [
                    "Chief complaint",
                    "Detailed history of present illness",
                    "Complete system review",
                    "Detailed examination",
                    "Moderate complexity medical decision making"
                ],
                "bundling_rules": [],
                "frequency": 0.6,
                "complexity_level": 3,
                "modifiers": ["25", "57"],
                "rvu": 3.26
            },
            {
                "code": "99204",
                "description": "Office visit, new patient, high complexity",
                "category": "Evaluation",
                "subcategory": "Office/Other Outpatient Services",
                "description_long": "Office or other outpatient visit for the evaluation and management of a new patient, which requires these 3 key components: A comprehensive history; A comprehensive examination; Medical decision making of high complexity.",
                "typical_usage": "Complex new patient evaluation with multiple comorbidities",
                "documentation_requirements": [
                    "Chief complaint",
                    "Comprehensive history",
                    "Complete system review",
                    "Comprehensive examination",
                    "High complexity medical decision making with multiple diagnoses"
                ],
                "bundling_rules": [],
                "frequency": 0.3,
                "complexity_level": 4,
                "modifiers": ["25", "57"],
                "rvu": 4.87
            },
            {
                "code": "99214",
                "description": "Office visit, established patient, moderate complexity",
                "category": "Evaluation",
                "subcategory": "Office/Other Outpatient Services",
                "description_long": "Office or other outpatient visit for the evaluation and management of an established patient, which requires at least 2 of these 3 key components: A detailed history; A detailed examination; Medical decision making of moderate complexity.",
                "typical_usage": "Follow-up visit with moderate medical complexity",
                "documentation_requirements": [
                    "Updated history",
                    "Medication review",
                    "Detailed examination or moderate complexity decision making"
                ],
                "bundling_rules": [],
                "frequency": 0.9,
                "complexity_level": 3,
                "modifiers": ["25"],
                "rvu": 2.51
            },
            {
                "code": "80061",
                "description": "Lipid panel",
                "category": "Laboratory",
                "subcategory": "Chemistry",
                "description_long": "Lipid panel; includes cholesterol, high-density lipoprotein (HDL) cholesterol, triglycerides, and calculated low-density lipoprotein (LDL) cholesterol.",
                "typical_usage": "Cardiovascular risk assessment",
                "documentation_requirements": [
                    "Physician order",
                    "Laboratory results",
                    "Clinical indication"
                ],
                "bundling_rules": ["Cannot be billed with individual lipid tests"],
                "frequency": 0.6,
                "complexity_level": 1,
                "modifiers": ["91", "59"],
                "rvu": 1.32
            },
            {
                "code": "85025",
                "description": "Complete blood count (CBC)",
                "category": "Laboratory",
                "subcategory": "Hematology",
                "description_long": "Blood count; complete (CBC), automated (Hgb, Hct, RBC, WBC, platelet count) and automated differential WBC count.",
                "typical_usage": "Routine blood work and infection screening",
                "documentation_requirements": [
                    "Physician order",
                    "Laboratory results"
                ],
                "bundling_rules": [],
                "frequency": 0.8,
                "complexity_level": 1,
                "modifiers": ["91", "59"],
                "rvu": 1.05
            },
            {
                "code": "83036",
                "description": "Hemoglobin A1c",
                "category": "Laboratory",
                "subcategory": "Chemistry",
                "description_long": "Hemoglobin; glycosylated (A1c)",
                "typical_usage": "Diabetes monitoring",
                "documentation_requirements": [
                    "Physician order",
                    "Laboratory results"
                ],
                "bundling_rules": [],
                "frequency": 0.7,
                "complexity_level": 1,
                "modifiers": ["91", "59"],
                "rvu": 0.71
            },
            {
                "code": "93000",
                "description": "Electrocardiogram, routine ECG",
                "category": "Diagnostic",
                "subcategory": "Cardiovascular",
                "description_long": "Electrocardiogram, routine ECG with at least 12 leads; with interpretation and report.",
                "typical_usage": "Cardiac evaluation and monitoring",
                "documentation_requirements": [
                    "Physician order",
                    "ECG tracing",
                    "Interpretation and report"
                ],
                "bundling_rules": [],
                "frequency": 0.5,
                "complexity_level": 2,
                "modifiers": ["26", "59"],
                "rvu": 1.14
            },
            {
                "code": "71020",
                "description": "Chest X-ray",
                "category": "Radiology",
                "subcategory": "Diagnostic Imaging",
                "description_long": "Radiologic examination, chest, 2 views, frontal and lateral.",
                "typical_usage": "Pulmonary evaluation and cardiac assessment",
                "documentation_requirements": [
                    "Physician order",
                    "Radiology report",
                    "Images"
                ],
                "bundling_rules": [],
                "frequency": 0.6,
                "complexity_level": 2,
                "modifiers": ["26", "TC"],
                "rvu": 0.74
            },
            {
                "code": "36415",
                "description": "Venipuncture",
                "category": "Procedure",
                "subcategory": "Phlebotomy",
                "description_long": "Collection of venous blood by venipuncture.",
                "typical_usage": "Blood draw for laboratory testing",
                "documentation_requirements": [
                    "Phlebotomy procedure note"
                ],
                "bundling_rules": ["Usually bundled with laboratory tests"],
                "frequency": 0.4,
                "complexity_level": 1,
                "modifiers": ["59"],
                "rvu": 0.25
            }
        ]

        # Convert to CPTCode objects
        for code_data in sample_codes:
            cpt_code = self._create_cpt_object(code_data)
            self.cpt_codes[cpt_code.code] = cpt_code

        # Save to file
        self._save_database()

    def _save_database(self):
        """Save CPT database to file."""
        try:
            Path(self.cpt_file).parent.mkdir(parents=True, exist_ok=True)
            data = []
            for cpt_code in self.cpt_codes.values():
                data.append({
                    'code': cpt_code.code,
                    'description': cpt_code.description,
                    'category': cpt_code.category,
                    'subcategory': cpt_code.subcategory,
                    'description_long': cpt_code.description_long,
                    'typical_usage': cpt_code.typical_usage,
                    'documentation_requirements': cpt_code.documentation_requirements,
                    'bundling_rules': cpt_code.bundling_rules,
                    'frequency': cpt_code.frequency,
                    'complexity_level': cpt_code.complexity_level,
                    'modifiers': cpt_code.modifiers,
                    'rvu': cpt_code.rvu
                })

            with open(self.cpt_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving CPT database: {e}")

    def _build_indexes(self):
        """Build search and category indexes."""
        # Category index
        self.category_index = {}
        for code, cpt_obj in self.cpt_codes.items():
            category = cpt_obj.category
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(code)

        # Search index - simple keyword index
        self.search_index = {}
        for code, cpt_obj in self.cpt_codes.items():
            keywords = self._extract_keywords(cpt_obj)
            for keyword in keywords:
                if keyword not in self.search_index:
                    self.search_index[keyword] = []
                if code not in self.search_index[keyword]:
                    self.search_index[keyword].append(code)

    def _extract_keywords(self, cpt_obj: CPTCode) -> List[str]:
        """Extract searchable keywords from CPT code."""
        keywords = []

        # Add code
        keywords.append(cpt_obj.code.lower())

        # Add words from description
        words = re.findall(r'\b\w+\b', cpt_obj.description.lower())
        keywords.extend(words)

        # Add words from long description
        words = re.findall(r'\b\w+\b', cpt_obj.description_long.lower())
        keywords.extend(words)

        # Add category
        keywords.append(cpt_obj.category.lower())

        # Add common medical terms
        if 'examination' in cpt_obj.description.lower():
            keywords.append('exam')
        if 'procedure' in cpt_obj.description.lower():
            keywords.append('procedure')
        if 'test' in cpt_obj.description.lower():
            keywords.append('test')

        # Filter and return unique keywords
        return list(set([kw for kw in keywords if len(kw) > 2]))

    def get_cpt_code(self, code: str) -> Optional[CPTCode]:
        """Get CPT code by exact match."""
        return self.cpt_codes.get(code)

    def search_codes(self, query: str, limit: int = 10) -> List[Tuple[str, CPTCode, float]]:
        """Search CPT codes by query string."""
        query_lower = query.lower()
        matches = []

        # Direct code match
        if query_lower in self.cpt_codes:
            cpt_obj = self.cpt_codes[query_lower]
            matches.append((query_lower, cpt_obj, 1.0))

        # Keyword search
        query_words = re.findall(r'\b\w+\b', query_lower)
        code_scores = {}

        for word in query_words:
            if word in self.search_index:
                for code in self.search_index[word]:
                    if code not in code_scores:
                        code_scores[code] = 0
                    code_scores[code] += 1

        # Calculate relevance scores
        for code, score in code_scores.items():
            if code not in [m[0] for m in matches]:  # Don't duplicate exact matches
                relevance = score / len(query_words)
                cpt_obj = self.cpt_codes[code]
                matches.append((code, cpt_obj, relevance))

        # Sort by relevance and return top matches
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:limit]

    def get_codes_by_category(self, category: str) -> List[CPTCode]:
        """Get all codes in a specific category."""
        if category not in self.category_index:
            return []

        codes = []
        for code in self.category_index[category]:
            codes.append(self.cpt_codes[code])

        return sorted(codes, key=lambda x: x.frequency, reverse=True)

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return sorted(self.category_index.keys())

    def get_frequently_used_codes(self, limit: int = 20) -> List[CPTCode]:
        """Get most frequently used CPT codes."""
        all_codes = list(self.cpt_codes.values())
        all_codes.sort(key=lambda x: x.frequency, reverse=True)
        return all_codes[:limit]

    def validate_code(self, code: str) -> bool:
        """Validate if a CPT code exists."""
        return code in self.cpt_codes

    def get_related_codes(self, code: str, limit: int = 5) -> List[CPTCode]:
        """Get codes related to the given code."""
        cpt_obj = self.get_cpt_code(code)
        if not cpt_obj:
            return []

        # Find codes in same category
        related = []
        for other_code in self.category_index.get(cpt_obj.category, []):
            if other_code != code:
                related.append(self.cpt_codes[other_code])

        # Sort by frequency and return top matches
        related.sort(key=lambda x: x.frequency, reverse=True)
        return related[:limit]

    def suggest_codes_by_text(self, text: str, limit: int = 10) -> List[Tuple[str, CPTCode, float]]:
        """Suggest CPT codes based on text content."""
        text_lower = text.lower()
        keywords = re.findall(r'\b\w+\b', text_lower)
        code_scores = {}

        for keyword in keywords:
            if len(keyword) < 3:  # Skip very short words
                continue

            if keyword in self.search_index:
                for code in self.search_index[keyword]:
                    if code not in code_scores:
                        code_scores[code] = 0
                    code_scores[code] += 1

        # Calculate scores and return matches
        matches = []
        for code, score in code_scores.items():
            relevance = score / len(keywords)
            cpt_obj = self.cpt_codes[code]
            matches.append((code, cpt_obj, relevance))

        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[:limit]

    def get_documentation_requirements(self, code: str) -> List[str]:
        """Get documentation requirements for a CPT code."""
        cpt_obj = self.get_cpt_code(code)
        if cpt_obj:
            return cpt_obj.documentation_requirements
        return []

    def check_bundling_rules(self, primary_code: str, secondary_codes: List[str]) -> List[str]:
        """Check bundling rules for multiple CPT codes."""
        warnings = []
        primary_cpt = self.get_cpt_code(primary_code)

        if not primary_cpt:
            return warnings

        for secondary_code in secondary_codes:
            secondary_cpt = self.get_cpt_code(secondary_code)

            if secondary_cpt:
                # Check if secondary code is typically bundled
                for rule in secondary_cpt.bundling_rules:
                    if primary_code in rule or primary_cpt.category in rule.lower():
                        warnings.append(f"{secondary_code} may be bundled with {primary_code}")

        return warnings

    def get_code_complexity_score(self, code: str) -> int:
        """Get complexity score for a CPT code (1-5)."""
        cpt_obj = self.get_cpt_code(code)
        if cpt_obj:
            return cpt_obj.complexity_level
        return 3  # Default medium complexity