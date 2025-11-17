"""
Main processing engine for medical record CPT coding.
Orchestrates PDF processing, text extraction, clinical preprocessing, and CPT coding.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from ..pdf.extractor import PDFExtractor
from ..pdf.preprocessor import ClinicalPreprocessor
from ..pdf.validator import MedicalRecordValidator
from ..coding.classifier import CPTEnsembleClassifier
from ..coding.explainer import SHAPExplainer
from ..coding.cpt_database import CPTDatabase
from ..core.file_manager import FileManager
from ..core.database import Database
from ..utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class ProcessingJob:
    """Represents a CPT coding processing job."""
    job_id: str
    file_path: str
    original_filename: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class ProcessingResult:
    """Result of CPT processing for a medical record."""
    job_id: str
    success: bool
    cpt_recommendations: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    text_quality_score: float
    validation_result: Optional[Dict[str, Any]] = None
    explanations: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None

class ProcessingEngine:
    """Main engine for processing medical records and generating CPT codes."""

    def __init__(self, config: Config):
        """Initialize processing engine."""
        self.config = config
        self.db = Database(config)
        self.file_manager = FileManager(config)

        # Initialize processing components
        self.pdf_extractor = PDFExtractor(config, self.file_manager)
        self.clinical_preprocessor = ClinicalPreprocessor(config)
        self.medical_validator = MedicalRecordValidator(config)
        self.cpt_classifier = CPTEnsembleClassifier(config, self.db)
        self.cpt_database = CPTDatabase(config)
        self.shap_explainer = SHAPExplainer(config, self.cpt_classifier)

        # Processing queue
        self.processing_queue = []
        self.active_jobs = {}

        # Callback functions for UI updates
        self.status_callback: Optional[Callable] = None
        self.progress_callback: Optional[Callable] = None
        self.completion_callback: Optional[Callable] = None

        logger.info("Processing engine initialized successfully")

    def set_callbacks(self, status_callback: Callable = None,
                     progress_callback: Callable = None,
                     completion_callback: Callable = None):
        """Set callback functions for UI updates."""
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback

    def process_medical_record(self, secure_file_path: str,
                             original_filename: str) -> str:
        """
        Process a medical record and return CPT recommendations.
        """
        # Create processing job
        job_id = str(uuid.uuid4())
        job = ProcessingJob(
            job_id=job_id,
            file_path=secure_file_path,
            original_filename=original_filename,
            status="queued",
            created_at=datetime.now()
        )

        # Update database
        self.db.update_job_status(job_id, "queued")

        # Add to processing queue
        self.processing_queue.append(job)
        self.active_jobs[job_id] = job

        logger.info(f"Queued processing job: {job_id} for file: {original_filename}")

        # Process immediately (in real implementation, this would be async)
        return self._process_job(job)

    def _process_job(self, job: ProcessingJob) -> ProcessingResult:
        """
        Process a single medical record job.
        """
        start_time = datetime.now()
        job.status = "processing"
        job.started_at = start_time

        try:
            logger.info(f"Starting processing for job: {job.job_id}")

            # Update status
            self._update_status(job.job_id, "Extracting text from PDF...")

            # Step 1: Extract text from PDF
            extraction_result = self.pdf_extractor.extract_text(job.file_path)
            if not extraction_result.text:
                raise ValueError("No text could be extracted from PDF")

            self._update_status(job.job_id, "Validating medical record content...")
            self._update_progress(job.job_id, 20)

            # Step 2: Validate medical record
            validation_result = self.medical_validator.validate_medical_record(extraction_result.text)
            if not validation_result.is_valid:
                logger.warning(f"Medical record validation failed for job {job.job_id}: {validation_result.errors}")

            self._update_status(job.job_id, "Processing clinical content...")
            self._update_progress(job.job_id, 40)

            # Step 3: Preprocess clinical text
            structured_note = self.clinical_preprocessor.preprocess_text(extraction_result.text)

            # Validate that we have meaningful content
            if structured_note.quality_score < 0.3:
                logger.warning(f"Low quality score for job {job.job_id}: {structured_note.quality_score}")

            self._update_status(job.job_id, "Analyzing for CPT codes...")
            self._update_progress(job.job_id, 60)

            # Step 4: Generate CPT recommendations
            ensemble_result = self.cpt_classifier.predict_cpt_codes(
                structured_note.cleaned_text, structured_note
            )

            # Filter recommendations by confidence threshold
            confidence_threshold = self.config.get("models.confidence_threshold", 0.7)
            valid_recommendations = [
                pred for pred in ensemble_result.predictions
                if pred.confidence >= confidence_threshold or len(ensemble_result.predictions) <= 3
            ]

            # Convert to result format
            cpt_recommendations = []
            for pred in valid_recommendations[:5]:  # Top 5 recommendations
                recommendation = {
                    'cpt_code': pred.cpt_code,
                    'description': pred.description,
                    'confidence': pred.confidence,
                    'requires_review': pred.requires_review,
                    'category': self.cpt_database.get_cpt_code(pred.cpt_code).category if self.cpt_database.get_cpt_code(pred.cpt_code) else 'Unknown'
                }
                cpt_recommendations.append(recommendation)

            self._update_status(job.job_id, "Generating explanations...")
            self._update_progress(job.job_id, 80)

            # Step 5: Generate explanations for top recommendations
            explanations = []
            if valid_recommendations:
                for pred in valid_recommendations[:3]:  # Explain top 3
                    try:
                        explanation = self.shap_explainer.explain_prediction(
                            structured_note.cleaned_text,
                            structured_note,
                            pred.cpt_code,
                            pred.confidence
                        )
                        explanations.append({
                            'cpt_code': pred.cpt_code,
                            'reasoning_summary': explanation.reasoning_summary,
                            'key_factors': [
                                {
                                    'feature': f.feature_name,
                                    'contribution': f.contribution,
                                    'explanation': f.explanation
                                }
                                for f in explanation.key_factors[:3]
                            ],
                            'supporting_text': [
                                {'text': snippet, 'relevance': conf}
                                for snippet, conf in explanation.supporting_text_snippets[:2]
                            ]
                        })
                    except Exception as e:
                        logger.error(f"Error generating explanation for {pred.cpt_code}: {e}")

            self._update_status(job.job_id, "Saving results...")
            self._update_progress(job.job_id, 95)

            # Step 6: Calculate processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            overall_confidence = ensemble_result.confidence_score if ensemble_result.predictions else 0.0

            # Create result object
            result = ProcessingResult(
                job_id=job.job_id,
                success=True,
                cpt_recommendations=cpt_recommendations,
                confidence_score=overall_confidence,
                processing_time=processing_time,
                text_quality_score=structured_note.quality_score,
                validation_result={
                    'is_valid': validation_result.is_valid,
                    'confidence': validation_result.confidence_score,
                    'warnings': validation_result.warnings,
                    'errors': validation_result.errors,
                    'recommendations': validation_result.recommendations
                } if validation_result else None,
                explanations=explanations
            )

            # Save results to database
            self._save_results(job.job_id, result)

            # Complete job
            job.status = "completed"
            job.completed_at = datetime.now()
            job.result = result.__dict__  # Convert to dict for storage

            # Update database
            self.db.update_job_status(
                job.job_id,
                "completed",
                processing_time=int(processing_time),
                text_length=len(extraction_result.text),
                quality_score=structured_note.quality_score
            )

            # Save CPT results
            if result.cpt_recommendations:
                self.db.save_cpt_results(job.job_id, result.cpt_recommendations)

            self._update_progress(job.job_id, 100)
            self._update_status(job.job_id, "Processing completed successfully")

            logger.info(f"Processing completed successfully for job: {job.job_id}")
            return result

        except Exception as e:
            # Handle processing error
            error_message = str(e)
            logger.error(f"Processing failed for job {job.job_id}: {error_message}")

            job.status = "failed"
            job.completed_at = datetime.now()
            job.error_message = error_message

            # Update database
            self.db.update_job_status(
                job.job_id,
                "failed",
                error_message=error_message,
                processing_time=int((datetime.now() - start_time).total_seconds())
            )

            # Create error result
            result = ProcessingResult(
                job_id=job.job_id,
                success=False,
                cpt_recommendations=[],
                confidence_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                text_quality_score=0.0,
                error_message=error_message
            )

            self._update_status(job.job_id, f"Processing failed: {error_message}")

            return result

        finally:
            # Clean up job from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

            # Call completion callback
            if self.completion_callback:
                self.completion_callback(job.job_id)

    def _save_results(self, job_id: str, result: ProcessingResult):
        """Save processing results to database."""
        try:
            # Save as job metadata
            job_data = {
                'success': result.success,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'text_quality_score': result.text_quality_score,
                'num_recommendations': len(result.cpt_recommendations),
                'validation_passed': result.validation_result['is_valid'] if result.validation_result else False,
                'completed_at': datetime.now().isoformat()
            }

            # This would be saved to a results table or as JSON metadata
            # For now, the results are saved via the database class methods already called

        except Exception as e:
            logger.error(f"Error saving results for job {job_id}: {e}")

    def _update_status(self, job_id: str, status: str):
        """Update processing status."""
        if self.status_callback:
            self.status_callback(job_id, status)

    def _update_progress(self, job_id: str, progress: int):
        """Update processing progress."""
        if self.progress_callback:
            self.progress_callback(job_id, progress)

    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get status of a processing job."""
        # Check active jobs first
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]

        # Check database
        try:
            job_results = self.db.get_job_results(job_id)
            if job_results:
                job_data = job_results['job']
                return ProcessingJob(
                    job_id=job_data['id'],
                    file_path=job_data['file_path'],
                    original_filename=job_data['file_name'],
                    status=job_data['status'],
                    created_at=datetime.fromisoformat(job_data['created_at']),
                    started_at=datetime.fromisoformat(job_data['started_at']) if job_data['started_at'] else None,
                    completed_at=datetime.fromisoformat(job_data['completed_at']) if job_data['completed_at'] else None,
                    error_message=job_data['error_message']
                )
        except Exception as e:
            logger.error(f"Error getting job status: {e}")

        return None

    def get_processing_results(self, job_id: str) -> Optional[ProcessingResult]:
        """Get processing results for a job."""
        try:
            job_results = self.db.get_job_results(job_id)
            if job_results and job_results['results']:
                # Convert database results to ProcessingResult
                first_result = job_results['results'][0]

                return ProcessingResult(
                    job_id=job_id,
                    success=True,
                    cpt_recommendations=[
                        {
                            'cpt_code': r['cpt_code'],
                            'description': r['description'],
                            'confidence': r['confidence_score'],
                            'requires_review': r['requires_review'],
                            'category': r.get('category', 'Unknown')
                        }
                        for r in job_results['results']
                    ],
                    confidence_score=sum(r['confidence_score'] for r in job_results['results']) / len(job_results['results']),
                    processing_time=job_results['job']['processing_time_seconds'] or 0,
                    text_quality_score=job_results['job']['quality_score'] or 0.0
                )
        except Exception as e:
            logger.error(f"Error getting processing results: {e}")

        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job."""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = "cancelled"
            job.completed_at = datetime.now()

            # Update database
            self.db.update_job_status(job_id, "failed", error_message="Cancelled by user")

            # Remove from active jobs
            del self.active_jobs[job_id]

            self._update_status(job_id, "Processing cancelled")
            return True

        return False

    def get_queue_status(self) -> Dict[str, int]:
        """Get current processing queue status."""
        return {
            'queued': len(self.processing_queue),
            'active': len(self.active_jobs),
            'total_completed': len(self.db.get_processing_jobs(status='completed')),
            'total_failed': len(self.db.get_processing_jobs(status='failed'))
        }

    def cleanup_old_jobs(self):
        """Clean up old processing jobs and data."""
        try:
            # Use file manager to clean up old files
            deleted_count = self.file_manager.auto_delete_old_files()
            logger.info(f"Cleaned up {deleted_count} old items")
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {e}")

    def shutdown(self):
        """Shutdown processing engine."""
        logger.info("Shutting down processing engine")

        # Cancel all active jobs
        for job_id in list(self.active_jobs.keys()):
            self.cancel_job(job_id)

        # Clear queues
        self.processing_queue.clear()
        self.active_jobs.clear()

        logger.info("Processing engine shutdown complete")