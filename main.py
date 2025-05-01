import os
import logging
from datetime import datetime
import configparser
from Models.detection import BoneTumorDetector
from Models.classification import TumorClassifier
from Models.segmentation import TumorSegmenter
from Models.report_generator import ReportGenerator


def setup_logging():
    """Configure logging system with UTF-8 encoding"""
    logging.basicConfig(
        level=logging.INFO,
        format=' %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bonecare.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    # Remove emojis from Windows console output
    if os.name == 'nt':
        logging.addLevelName(logging.INFO, 'INFO')
        logging.addLevelName(logging.ERROR, 'ERROR')


def get_image_paths(config):
    """Get list of images to process with proper path resolution"""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if 'SYSTEM' section exists in config file
    if 'SYSTEM' not in config:
        raise KeyError("The 'SYSTEM' section is missing in the configuration file")

    input_dir = os.path.join(project_root, config['SYSTEM']['InputDir'])

    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            f"Please create the directory structure: {config['SYSTEM']['InputDir']}"
        )

    # Get all image files (you can adjust the extensions as needed)
    image_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    if not image_paths:
        raise FileNotFoundError(f"No valid images found in {input_dir}")

    return image_paths


def main():
    try:
        setup_logging()
        logger = logging.getLogger('main')

        # Load configuration
        config = configparser.ConfigParser()

        # Load the config file
        config.read('systemfile.ini')

        logger.info(" Starting BoneCare AI Pipeline")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


        # Get images to process
        image_paths = get_image_paths(config)
        logger.info(f"Found {len(image_paths)} images for processing")

        # Initialize components
        detector = BoneTumorDetector()
        classifier = TumorClassifier()
        segmenter = TumorSegmenter()
        reporter = ReportGenerator()

        # Run pipeline
        logger.info("=== Starting Detection ===")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        det_results = detector.predict(image_paths)

        logger.info("=== Starting Classification ===")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        cls_results = classifier.predict(image_paths)

        logger.info("=== Starting Segmentation ===")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        seg_results = segmenter.predict(image_paths)

        logger.info("=== Generating Report ===")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report = reporter.generate_report()

        logger.info(f"Pipeline completed successfully\nReport saved to: {report['report_path']}")
        logger.info(f"Ending Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        logging.critical(f" Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()