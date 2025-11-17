"""
Ensemble CPT classification engine for medical coding.
Uses multiple ML models (Random Forest, SVM, Gradient Boosting, LSTM) with stacking.
"""

import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Attention
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available, using simplified neural network")

from ..utils.config import Config
from ..core.database import Database

logger = logging.getLogger(__name__)

@dataclass
class CPTPrediction:
    """Single CPT code prediction with confidence."""
    cpt_code: str
    description: str
    confidence: float
    model_confidence: Dict[str, float]
    explanation: str
    requires_review: bool

@dataclass
class EnsembleResult:
    """Result from ensemble CPT classification."""
    predictions: List[CPTPrediction]
    primary_code: Optional[str]
    confidence_score: float
    processing_time: float
    model_performances: Dict[str, float]
    feature_importance: Dict[str, float]

class CPTEnsembleClassifier:
    """Ensemble classifier for CPT code prediction."""

    def __init__(self, config: Config, database: Database):
        """Initialize ensemble classifier."""
        self.config = config
        self.db = database
        self.models_dir = Path(config.get_models_directory())
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models
        self.random_forest = None
        self.svm = None
        self.gradient_boosting = None
        self.lstm_model = None
        self.mlp_model = None

        # Feature extraction
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tokenizer = None

        # Model weights from config
        self.model_weights = config.get("models.ensemble_weights", {
            "random_forest": 0.3,
            "svm": 0.25,
            "gradient_boosting": 0.25,
            "lstm": 0.2
        })

        # CPT code mapping
        self.cpt_mapping = {}
        self._load_cpt_codes()

        # Load or train models
        self._initialize_models()

    def _load_cpt_codes(self):
        """Load CPT code database."""
        try:
            cpt_file = self.config.get_cpt_codes_path()
            if Path(cpt_file).exists():
                with open(cpt_file, 'r') as f:
                    cpt_data = json.load(f)
                    for code_info in cpt_data:
                        self.cpt_mapping[code_info['code']] = code_info
                logger.info(f"Loaded {len(self.cpt_mapping)} CPT codes")
            else:
                # Create sample CPT data for demonstration
                self._create_sample_cpt_data()
                logger.warning("Using sample CPT data - replace with real 2021 CPT codes")

        except Exception as e:
            logger.error(f"Error loading CPT codes: {e}")
            self._create_sample_cpt_data()

    def _create_sample_cpt_data(self):
        """Create sample CPT code data for testing."""
        sample_cpt_codes = [
            {"code": "99201", "description": "Office visit, new patient", "category": "Evaluation"},
            {"code": "99202", "description": "Office visit, new patient, low complexity", "category": "Evaluation"},
            {"code": "99203", "description": "Office visit, new patient, moderate complexity", "category": "Evaluation"},
            {"code": "99204", "description": "Office visit, new patient, high complexity", "category": "Evaluation"},
            {"code": "99205", "description": "Office visit, new patient, very high complexity", "category": "Evaluation"},
            {"code": "99211", "description": "Office visit, established patient, minimal", "category": "Evaluation"},
            {"code": "99212", "description": "Office visit, established patient, straightforward", "category": "Evaluation"},
            {"code": "99213", "description": "Office visit, established patient, low complexity", "category": "Evaluation"},
            {"code": "99214", "description": "Office visit, established patient, moderate complexity", "category": "Evaluation"},
            {"code": "99215", "description": "Office visit, established patient, high complexity", "category": "Evaluation"},
            {"code": "80061", "description": "Lipid panel", "category": "Laboratory"},
            {"code": "85025", "description": "Complete blood count (CBC)", "category": "Laboratory"},
            {"code": "83036", "description": "Hemoglobin A1c", "category": "Laboratory"},
            {"code": "81002", "description": "Urinalysis, dipstick", "category": "Laboratory"},
            {"code": "93000", "description": "Electrocardiogram, routine ECG", "category": "Diagnostic"},
            {"code": "71020", "description": "Chest X-ray", "category": "Radiology"},
            {"code": "71250", "description": "CT scan, chest", "category": "Radiology"},
            {"code": "70553", "description": "MRI, brain", "category": "Radiology"},
            {"code": "36415", "description": "Venipuncture", "category": "Procedure"},
            {"code": "99214", "description": "Intravenous infusion", "category": "Procedure"}
        ]

        for code_info in sample_cpt_codes:
            self.cpt_mapping[code_info["code"]] = code_info

        # Save sample data
        cpt_file = self.config.get_cpt_codes_path()
        Path(cpt_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cpt_file, 'w') as f:
            json.dump(sample_cpt_codes, f, indent=2)

    def _initialize_models(self):
        """Initialize or load trained models."""
        try:
            # Try to load existing models
            self._load_models()
            logger.info("Loaded existing trained models")
        except Exception as e:
            logger.warning(f"Could not load models: {e}, creating new models")
            # Initialize new models (will need training)
            self._create_new_models()

    def _create_new_models(self):
        """Create new untrained models."""
        # Random Forest
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        # SVM
        self.svm = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            C=1.0,
            gamma='scale'
        )

        # Gradient Boosting
        self.gradient_boosting = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        # MLP Neural Network
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            random_state=42,
            max_iter=500
        )

        # Initialize feature extractors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )

        if TF_AVAILABLE:
            self.tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

    def _load_models(self):
        """Load pre-trained models from disk."""
        models_path = self.models_dir

        # Load ML models
        with open(models_path / "random_forest.pkl", 'rb') as f:
            self.random_forest = pickle.load(f)

        with open(models_path / "svm.pkl", 'rb') as f:
            self.svm = pickle.load(f)

        with open(models_path / "gradient_boosting.pkl", 'rb') as f:
            self.gradient_boosting = pickle.load(f)

        with open(models_path / "mlp.pkl", 'rb') as f:
            self.mlp_model = pickle.load(f)

        # Load feature extractors
        with open(models_path / "tfidf_vectorizer.pkl", 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)

        with open(models_path / "scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)

        with open(models_path / "label_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Load LSTM if available
        if TF_AVAILABLE:
            try:
                self.lstm_model = tf.keras.models.load_model(models_path / "lstm_model.h5")
                with open(models_path / "tokenizer.pkl", 'rb') as f:
                    self.tokenizer = pickle.load(f)
            except:
                logger.warning("LSTM model not found, using other models")

    def predict_cpt_codes(self, processed_text: str, structured_note: Any) -> EnsembleResult:
        """
        Predict CPT codes using ensemble approach.
        """
        start_time = datetime.now()

        try:
            logger.info("Starting CPT code prediction with ensemble")

            # Extract features
            features = self._extract_features(processed_text, structured_note)

            # Get predictions from each model
            predictions = {}
            model_performances = {}

            if self.random_forest and self.tfidf_vectorizer:
                rf_pred, rf_conf = self._predict_random_forest(features)
                predictions['random_forest'] = (rf_pred, rf_conf)
                model_performances['random_forest'] = rf_conf

            if self.svm and self.tfidf_vectorizer:
                svm_pred, svm_conf = self._predict_svm(features)
                predictions['svm'] = (svm_pred, svm_conf)
                model_performances['svm'] = svm_conf

            if self.gradient_boosting and self.tfidf_vectorizer:
                gb_pred, gb_conf = self._predict_gradient_boosting(features)
                predictions['gradient_boosting'] = (gb_pred, gb_conf)
                model_performances['gradient_boosting'] = gb_conf

            if self.mlp_model and self.tfidf_vectorizer:
                mlp_pred, mlp_conf = self._predict_mlp(features)
                predictions['mlp'] = (mlp_pred, mlp_conf)
                model_performances['mlp'] = mlp_conf

            # Combine predictions using ensemble weights
            ensemble_predictions = self._ensemble_predictions(predictions)

            # Create CPT prediction objects
            cpt_predictions = self._create_cpt_predictions(ensemble_predictions)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Determine primary code
            primary_code = cpt_predictions[0].cpt_code if cpt_predictions else None
            confidence_score = cpt_predictions[0].confidence if cpt_predictions else 0.0

            result = EnsembleResult(
                predictions=cpt_predictions,
                primary_code=primary_code,
                confidence_score=confidence_score,
                processing_time=processing_time,
                model_performances=model_performances,
                feature_importance=self._calculate_feature_importance()
            )

            logger.info(f"CPT prediction completed in {processing_time:.2f}s, primary code: {primary_code}")
            return result

        except Exception as e:
            logger.error(f"Error in CPT prediction: {e}")
            # Return empty result on error
            processing_time = (datetime.now() - start_time).total_seconds()
            return EnsembleResult(
                predictions=[],
                primary_code=None,
                confidence_score=0.0,
                processing_time=processing_time,
                model_performances={},
                feature_importance={}
            )

    def _extract_features(self, text: str, structured_note: Any) -> np.ndarray:
        """Extract features for ML models."""
        features = []

        # Text features
        text_length = len(text)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        features.extend([text_length, word_count, sentence_count])

        # Medical content features
        medical_terms = ['patient', 'diagnosis', 'treatment', 'medication', 'examination',
                        'procedure', 'therapy', 'symptoms', 'assessment', 'plan']
        term_counts = [text.lower().count(term) for term in medical_terms]
        features.extend(term_counts)

        # Section features (if available from structured note)
        if hasattr(structured_note, 'sections'):
            sections = ['history', 'examination', 'assessment', 'plan', 'medications']
            section_present = [1 if section in structured_note.sections else 0 for section in sections]
            features.extend(section_present)
        else:
            features.extend([0] * 5)

        # Entity features (if available)
        if hasattr(structured_note, 'entities'):
            entity_categories = ['procedure', 'medication', 'diagnosis', 'measurement']
            entity_counts = [sum(1 for e in structured_note.entities if e.category == cat)
                           for cat in entity_categories]
            features.extend(entity_counts)
        else:
            features.extend([0] * 4)

        return np.array(features).reshape(1, -1)

    def _predict_random_forest(self, features: np.ndarray) -> Tuple[str, float]:
        """Get prediction from Random Forest model."""
        if not self.tfidf_vectorizer or not self.random_forest:
            return "", 0.0

        try:
            # This is simplified - in practice, you'd use TF-IDF features
            # For now, we'll use the engineered features
            features_scaled = self.scaler.transform(features)
            prediction = self.random_forest.predict(features_scaled)[0]
            probabilities = self.random_forest.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)

            return str(prediction), confidence
        except Exception as e:
            logger.error(f"Random Forest prediction error: {e}")
            return "", 0.0

    def _predict_svm(self, features: np.ndarray) -> Tuple[str, float]:
        """Get prediction from SVM model."""
        if not self.svm:
            return "", 0.0

        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.svm.predict(features_scaled)[0]
            probabilities = self.svm.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)

            return str(prediction), confidence
        except Exception as e:
            logger.error(f"SVM prediction error: {e}")
            return "", 0.0

    def _predict_gradient_boosting(self, features: np.ndarray) -> Tuple[str, float]:
        """Get prediction from Gradient Boosting model."""
        if not self.gradient_boosting:
            return "", 0.0

        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.gradient_boosting.predict(features_scaled)[0]
            probabilities = self.gradient_boosting.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)

            return str(prediction), confidence
        except Exception as e:
            logger.error(f"Gradient Boosting prediction error: {e}")
            return "", 0.0

    def _predict_mlp(self, features: np.ndarray) -> Tuple[str, float]:
        """Get prediction from MLP Neural Network."""
        if not self.mlp_model:
            return "", 0.0

        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.mlp_model.predict(features_scaled)[0]
            probabilities = self.mlp_model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)

            return str(prediction), confidence
        except Exception as e:
            logger.error(f"MLP prediction error: {e}")
            return "", 0.0

    def _ensemble_predictions(self, predictions: Dict[str, Tuple[str, float]]) -> Dict[str, float]:
        """Combine predictions from multiple models using weighted voting."""
        if not predictions:
            return {}

        # Count weighted votes for each CPT code
        code_votes = {}

        for model_name, (code, confidence) in predictions.items():
            if code and code in self.cpt_mapping:
                weight = self.model_weights.get(model_name, 0.25)
                weighted_confidence = confidence * weight

                if code not in code_votes:
                    code_votes[code] = 0.0
                code_votes[code] += weighted_confidence

        # Normalize scores
        total_votes = sum(code_votes.values())
        if total_votes > 0:
            for code in code_votes:
                code_votes[code] /= total_votes

        return code_votes

    def _create_cpt_predictions(self, ensemble_scores: Dict[str, float]) -> List[CPTPrediction]:
        """Create CPT prediction objects from ensemble scores."""
        predictions = []

        # Sort by score (descending)
        sorted_codes = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)

        for code, score in sorted_codes[:10]:  # Top 10 predictions
            if code in self.cpt_mapping:
                cpt_info = self.cpt_mapping[code]
                requires_review = score < self.config.get("models.confidence_threshold", 0.7)

                prediction = CPTPrediction(
                    cpt_code=code,
                    description=cpt_info.get('description', ''),
                    confidence=score,
                    model_confidence={"ensemble": score},
                    explanation=f"Ensemble prediction with {score:.2%} confidence",
                    requires_review=requires_review
                )
                predictions.append(prediction)

        return predictions

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance for explanation."""
        if self.random_forest and hasattr(self.random_forest, 'feature_importances_'):
            feature_names = [
                'text_length', 'word_count', 'sentence_count',
                'patient_terms', 'diagnosis_terms', 'treatment_terms', 'medication_terms',
                'examination_terms', 'procedure_terms', 'therapy_terms', 'symptoms_terms',
                'assessment_terms', 'plan_terms',
                'history_section', 'exam_section', 'assessment_section', 'plan_section', 'meds_section',
                'procedure_entities', 'medication_entities', 'diagnosis_entities', 'measurement_entities'
            ]

            importance_dict = {}
            for i, importance in enumerate(self.random_forest.feature_importances_):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = float(importance)

            return importance_dict

        return {}

    def train_models(self, training_data: List[Dict[str, Any]]):
        """Train the ensemble models with provided data."""
        try:
            logger.info("Starting model training")

            if not training_data:
                logger.warning("No training data provided")
                return

            # Extract features and labels
            X_texts = [item['text'] for item in training_data]
            y_codes = [item['cpt_code'] for item in training_data]

            # Prepare labels
            y_encoded = self.label_encoder.fit_transform(y_codes)

            # Create TF-IDF features
            X_tfidf = self.tfidf_vectorizer.fit_transform(X_texts)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y_encoded, test_size=0.2, random_state=42
            )

            # Train models
            self.random_forest.fit(X_train, y_train)
            self.svm.fit(X_train, y_train)
            self.gradient_boosting.fit(X_train, y_train)
            self.mlp_model.fit(X_train, y_train)

            # Evaluate models
            models = {
                'Random Forest': self.random_forest,
                'SVM': self.svm,
                'Gradient Boosting': self.gradient_boosting,
                'MLP': self.mlp_model
            }

            for name, model in models.items():
                score = model.score(X_test, y_test)
                logger.info(f"{name} accuracy: {score:.3f}")

            # Save models
            self._save_models()
            logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def _save_models(self):
        """Save trained models to disk."""
        models_path = self.models_dir
        models_path.mkdir(parents=True, exist_ok=True)

        with open(models_path / "random_forest.pkl", 'wb') as f:
            pickle.dump(self.random_forest, f)

        with open(models_path / "svm.pkl", 'wb') as f:
            pickle.dump(self.svm, f)

        with open(models_path / "gradient_boosting.pkl", 'wb') as f:
            pickle.dump(self.gradient_boosting, f)

        with open(models_path / "mlp.pkl", 'wb') as f:
            pickle.dump(self.mlp_model, f)

        with open(models_path / "tfidf_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)

        with open(models_path / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(models_path / "label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)

        if self.tokenizer:
            with open(models_path / "tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f)

        logger.info("Models saved successfully")