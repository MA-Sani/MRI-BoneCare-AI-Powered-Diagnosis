import pandas as pd
from datetime import datetime
import os
import logging
import configparser
from typing import Dict, Any


class ReportGenerator:
    def __init__(self):
        self._init_logger()
        self.config = self._load_config()
        self._validate_paths()

    def _init_logger(self):
        self.logger = logging.getLogger('reporting')
        self.logger.addHandler(logging.FileHandler('bonecare.log'))

    def _load_config(self):
        config = configparser.ConfigParser()
        config.read('systemfile.ini')
        return config

    def _validate_paths(self):
        """Ensure all required paths exist and are writable"""
        required_paths = [
            os.path.abspath(self.config['PATHS']['DetectionResults']),
            os.path.abspath(self.config['PATHS']['ClassificationResults']),
            os.path.abspath(self.config['PATHS']['SegmentationResults']),
            os.path.abspath(self.config['PATHS']['ReportPath'])
        ]

        for path in required_paths:
            dir_path = os.path.dirname(path)
            if not os.path.exists(dir_path):
                self.logger.info(f"Creating directory: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)

            if not os.access(dir_path, os.W_OK):
                raise PermissionError(f"No write permissions for: {dir_path}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report with statistics and visual indicators"""
        try:
            self.logger.info("Starting report generation...")

            # Load data with error handling
            data = {
                'detection': self._safe_load_csv(self.config['PATHS']['DetectionResults']),
                'classification': self._safe_load_csv(self.config['PATHS']['ClassificationResults']),
                'segmentation': self._safe_load_csv(self.config['PATHS']['SegmentationResults'])
            }

            # Generate statistics
            stats = self._calculate_statistics(data)

            # Create report content
            report_content = self._create_report_content(data, stats)

            # Save report
            report_path = os.path.abspath(self.config['PATHS']['ReportPath'])
            with open(report_path, 'w') as f:
                f.write(report_content)

            self.logger.info(f"Report saved to {report_path}")

            return {
                'report_path': report_path,
                'statistics': stats,
                'raw_data': data
            }

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise

    def _safe_load_csv(self, path: str) -> pd.DataFrame:
        """Safely load CSV file with error handling"""
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
            return pd.read_csv(abs_path)
        return pd.DataFrame()

    def _calculate_statistics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate key statistics from the data"""
        stats = {
            'total_images': len(data['detection']),
            'malignant_count': 0,
            'avg_tumor_area': 0,
            'max_confidence': 0
        }

        if not data['classification'].empty:
            stats['malignant_count'] = len(
                data['classification'][data['classification']['Class'] == 'Malignant']
            )

        if not data['segmentation'].empty:
            stats['avg_tumor_area'] = data['segmentation']['Tumor_Area'].mean()
            stats['max_tumor_area'] = data['segmentation']['Tumor_Area'].max()

        if not data['detection'].empty:
            stats['max_confidence'] = data['detection']['Confidence'].max()

        return stats

    def _create_report_content(self, data: Dict[str, pd.DataFrame], stats: Dict[str, Any]) -> str:
        """Generate formatted report content"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = [
            "BONE TUMOR ANALYSIS REPORT",
            "=" * 50,
            f"Generated: {timestamp}\n",

            "SUMMARY STATISTICS:",
            f"- Total Images Processed: {stats['total_images']}",
            f"- Malignant Cases Detected: {stats['malignant_count']}",
            f"- Average Tumor Area: {stats['avg_tumor_area']:.1f} pixels",
            f"- Largest Tumor Area: {stats['max_tumor_area']:.1f} pixels\n",

            "DETECTION RESULTS:",
            data['detection'].to_string(index=False) if not data['detection'].empty else "No detections found",

            "\n\nCLASSIFICATION RESULTS:",
            data['classification'].to_string(index=False) if not data[
                'classification'].empty else "No classifications found",

            "\n\nSEGMENTATION RESULTS:",
            data['segmentation'].to_string(index=False) if not data['segmentation'].empty else "No segmentations found",

            "\n\nRECOMMENDATIONS:",
            self._generate_recommendations(stats)
        ]

        return "\n".join(report)

    def _generate_recommendations(self, stats: Dict[str, Any]) -> str:
        """Generate clinical recommendations based on findings"""
        recommendations = []

        if stats['malignant_count'] > 0:
            recommendations.append(" URGENT: Malignant tumors detected - Immediate specialist consultation required")

        if stats['max_tumor_area'] > 1000:
            recommendations.append(" Large tumor areas detected - Consider surgical evaluation")

        if stats['total_images'] > 10 and stats['malignant_count'] > 2:
            recommendations.append(" Multiple concerning findings - Recommend full diagnostic workup")

        return "\n".join(recommendations) if recommendations else "No critical findings requiring immediate action"