import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import configparser
import logging
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms


class BoneTumorDetector:
    def __init__(self, threshold=0.5):
        self._init_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.config = self._load_config()

        logging.info("Initializing detection model...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        logging.info("Detection model initialized successfully")

    def _init_logger(self):
        self.logger = logging.getLogger('detection')
        self.logger.addHandler(logging.FileHandler('bonecare.log'))

    def _load_config(self):
        config = configparser.ConfigParser()
        config.read('systemfile.ini')
        return config

    def predict(self, image_paths):
        output_csv = os.path.abspath(self.config['PATHS']['DetectionResults'])
        limit = self.config.getint('SYSTEM', 'Limit', fallback=20)
        self.logger.info(f"Starting detection process for {limit} images")
        results = []
        processed = 0

        try:
            # Use tqdm to show the progress in a single line
            with tqdm(total=min(len(image_paths), limit), desc="Detecting tumors", ncols=100, dynamic_ncols=True) as pbar:
                for img_path in image_paths:
                    if processed >= limit:
                        self.logger.info(f"Reached system limit of {limit} images")
                        break

                    try:
                        self.logger.debug(f"Processing {img_path}")
                        img = Image.open(img_path).convert('RGB')
                        tensor = self.transform(img).unsqueeze(0).to(self.device)

                        with torch.inference_mode():
                            outputs = self.model(tensor)[0]

                        for i, score in enumerate(outputs["scores"]):
                            if score >= self.threshold:
                                box = outputs["boxes"][i].cpu().numpy()
                                results.append({
                                    'Image': os.path.basename(img_path),
                                    'Detection': 'Tumor Detected',
                                    'Coordinates': f"{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}",
                                    'Confidence': f"{score.item():.4f}",
                                    'Label': 'Tumor'
                                })

                        processed += 1
                        pbar.update(1)  # Update the progress bar after each image
                    except Exception as e:
                        self.logger.error(f"Error processing {img_path}: {str(e)}")
                    finally:
                        if 'tensor' in locals():
                            del tensor
                            torch.cuda.empty_cache()

            # Save results to CSV after processing all images
            pd.DataFrame(results).to_csv(output_csv, index=False)
            self.logger.info(f"Saved detection results to {output_csv}")
            return results

        except Exception as e:
            self.logger.critical(f"Detection failed: {str(e)}")
            raise