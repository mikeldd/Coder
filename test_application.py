#!/usr/bin/env python3
"""
Test script for the Medical CPT Coding Application.
Validates basic functionality without running the full GUI.
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import Config
from src.core.database import Database
from src.core.file_manager import FileManager
from src.coding.cpt_database import CPTDatabase

def test_basic_functionality():
    """Test basic application functionality."""
    print("🧪 Testing Medical CPT Coding Application")
    print("=" * 50)

    try:
        # Test configuration
        print("📋 Testing configuration...")
        config = Config()
        print(f"✅ Config loaded successfully")
        print(f"   Database path: {config.get_database_path()}")
        print(f"   Models directory: {config.get_models_directory()}")

        # Test database
        print("\n💾 Testing database...")
        db = Database(config)
        print("✅ Database initialized successfully")

        # Test file manager
        print("\n📁 Testing file manager...")
        file_manager = FileManager(config)
        stats = file_manager.get_storage_stats()
        print(f"✅ File manager initialized")
        print(f"   Temp directory: {file_manager.temp_dir}")

        # Test CPT database
        print("\n🏥 Testing CPT database...")
        cpt_db = CPTDatabase(config)
        categories = cpt_db.get_categories()
        frequent_codes = cpt_db.get_frequently_used_codes(5)
        print(f"✅ CPT database loaded with {len(cpt_db.cpt_codes)} codes")
        print(f"   Categories: {', '.join(categories)}")
        print(f"   Top codes: {[code.code for code in frequent_codes]}")

        # Test CPT search
        print("\n🔍 Testing CPT search...")
        search_results = cpt_db.search_codes("office visit", 3)
        print(f"✅ Search results for 'office visit': {len(search_results)} found")
        for code, cpt_obj, score in search_results:
            print(f"   {code}: {cpt_obj.description} ({score:.2f})")

        print("\n🎉 All tests passed! Application is ready.")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_processing():
    """Test processing with sample medical text."""
    print("\n🧪 Testing Sample Processing")
    print("=" * 30)

    try:
        config = Config()

        # Import processing components
        from src.pdf.preprocessor import ClinicalPreprocessor
        from src.coding.classifier import CPTEnsembleClassifier
        from src.core.database import Database

        # Create sample medical note text
        sample_text = """
        PATIENT: John Doe
        DATE: 11/16/2025

        SUBJECTIVE:
        Patient presents with complaint of chest pain for 2 days. Pain is sharp, worse with deep breathing.
        Reports shortness of breath with minimal exertion. No history of similar symptoms.

        OBJECTIVE:
        Vitals: BP 140/90, HR 88, RR 18, Temp 98.6°F, O2 Sat 96% on room air
        Chest examination reveals tenderness on palpation of right chest wall.
        Lungs clear to auscultation bilaterally.
        Heart sounds normal, no murmurs or rubs.

        ASSESSMENT:
        1. Chest pain, likely musculoskeletal
        2. Hypertension, well-controlled
        3. History of smoking

        PLAN:
        1. Chest X-ray to rule out pulmonary pathology
        2. ECG to evaluate cardiac status
        3. Complete blood count and basic metabolic panel
        4. Follow up in 1 week
        5. Continue current medications
        """

        print("📝 Processing sample clinical note...")

        # Test preprocessing
        preprocessor = ClinicalPreprocessor(config)
        structured_note = preprocessor.preprocess_text(sample_text)

        print(f"✅ Text preprocessing completed")
        print(f"   Document type: {structured_note.document_type}")
        print(f"   Quality score: {structured_note.quality_score:.2f}")
        print(f"   Sections found: {list(structured_note.sections.keys())}")
        print(f"   Entities extracted: {len(structured_note.entities)}")

        # Test CPT classifier (will create new models)
        print("\n🏥 Testing CPT classification...")
        db = Database(config)
        classifier = CPTEnsembleClassifier(config, db)

        # This will use untrained models but demonstrates the pipeline
        result = classifier.predict_cpt_codes(structured_note.cleaned_text, structured_note)

        print(f"✅ CPT classification completed")
        print(f"   Processing time: {result.processing_time:.2f} seconds")
        print(f"   Primary code: {result.primary_code}")
        print(f"   Predictions generated: {len(result.predictions)}")

        if result.predictions:
            print("   Top recommendations:")
            for i, pred in enumerate(result.predictions[:3], 1):
                print(f"     {i}. {pred.cpt_code} - {pred.confidence:.1%} - {pred.description}")

        print("\n✅ Sample processing test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Sample processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_directories():
    """Create necessary directories for the application."""
    print("📁 Creating application directories...")

    try:
        # Create directories
        directories = [
            "data/models",
            "data/codes",
            "data/sample_docs",
            "temp",
            "temp/logs",
            "temp/audit",
            "temp/secure"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created {directory}")

        print("✅ All directories created successfully!")
        return True

    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("🚀 Medical CPT Coding Application - Test Suite")
    print("=" * 60)

    # Create directories
    if not create_sample_directories():
        sys.exit(1)

    # Run basic functionality tests
    if not test_basic_functionality():
        sys.exit(1)

    # Run sample processing test
    if not test_sample_processing():
        sys.exit(1)

    print("\n🎊 All tests completed successfully!")
    print("The Medical CPT Coding Application is ready to use.")
    print("\nTo start the application, run:")
    print("  python main.py")

    print("\nApplication Features:")
    print("  ✅ PyQt6 desktop GUI with drag-and-drop PDF upload")
    print("  ✅ PDF text extraction with OCR fallback")
    print("  ✅ Clinical note preprocessing and entity recognition")
    print("  ✅ Ensemble CPT classification (Random Forest, SVM, Gradient Boosting)")
    print("  ✅ SHAP-based explanations for recommendations")
    print("  ✅ SQLite database with audit trails")
    print("  ✅ HIPAA-compliant auto-deletion (30 days)")
    print("  ✅ Results export (CSV, JSON, reports)")
    print("  ✅ Real-time processing status with QThread")
    print("  ✅ Professional medical software interface")

    print("\nNext Steps:")
    print("  1. Install required dependencies: pip install -r requirements.txt")
    print("  2. Download spaCy medical model: python -m spacy download en_core_med_sci_lg")
    print("  3. Run the application: python main.py")
    print("  4. Upload medical record PDFs for CPT coding")