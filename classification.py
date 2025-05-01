import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import configparser
import logging
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms


class TumorClassifier:
    def __init__(self):
        self._init_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config()

        logging.info("Initializing classification model...")
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 3)
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        logging.info("Classification model initialized")

    def _init_logger(self):
        self.logger = logging.getLogger('classification')
        self.logger.addHandler(logging.FileHandler('bonecare.log'))

    def _load_config(self):
        config = configparser.ConfigParser()
        config.read('systemfile.ini')
        return config

    def predict(self, image_paths):
        output_csv = os.path.abspath(self.config['PATHS']['ClassificationResults'])
        class_names = ['Benign', 'Intermediate', 'Malignant']
        results = []
        limit = self.config.getint('SYSTEM', 'Limit', fallback=20)
        processed = 0

        try:
            with tqdm(total=min(len(image_paths), limit), desc="Classifying tumors") as pbar:
                for img_path in image_paths:
                    if processed >= limit:
                        self.logger.info(f"Reached system limit of {limit} images")
                        break

                    try:
                        self.logger.debug(f"Classifying {img_path}")
                        img = Image.open(img_path).convert('RGB')
                        tensor = self.transform(img).unsqueeze(0).to(self.device)

                        with torch.inference_mode():
                            outputs = self.model(tensor)
                            _, preds = torch.max(outputs, 1)
                            probs = torch.nn.functional.softmax(outputs, dim=1)

                        results.append({
                            'Image': os.path.basename(img_path),
                            'Class': class_names[preds[0].item()],
                            'Confidence': f"{probs[0][preds[0]].item():.4f}"
                        })
                        processed += 1
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Error classifying {img_path}: {str(e)}")
                    finally:
                        if 'tensor' in locals():
                            del tensor
                            torch.cuda.empty_cache()

            pd.DataFrame(results).to_csv(output_csv, index=False)
            self.logger.info(f"Saved classification results to {output_csv}")
            return results

        except Exception as e:
            self.logger.critical(f"Classification failed: {str(e)}")
            raise