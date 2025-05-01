import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import configparser
import logging
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

class UNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified architecture for example
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class TumorSegmenter:
    def __init__(self):
        self._init_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config()

        logging.info("Initializing segmentation model...")
        self.model = UNetPlusPlus().to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        logging.info("Segmentation model initialized")

    def _init_logger(self):
        self.logger = logging.getLogger('segmentation')
        self.logger.addHandler(logging.FileHandler('bonecare.log'))

    def _load_config(self):
        config = configparser.ConfigParser()
        config.read('systemfile.ini')
        return config

    def predict(self, image_paths):
        output_dir = os.path.abspath(self.config['PATHS']['SegmentationMasks'])
        output_csv = os.path.abspath(self.config['PATHS']['SegmentationResults'])
        os.makedirs(output_dir, exist_ok=True)
        results = []
        limit = self.config.getint('SYSTEM', 'Limit', fallback=20)
        processed = 0

        try:
            # Use tqdm for progress in a single line with dynamic width
            with tqdm(total=min(len(image_paths), limit), desc="Segmenting tumors", ncols=100, dynamic_ncols=True) as pbar:
                for img_path in image_paths:
                    if processed >= limit:
                        self.logger.info(f"Reached system limit of {limit} images")
                        break

                    try:
                        self.logger.debug(f"Segmenting {img_path}")
                        img = Image.open(img_path).convert('L')
                        tensor = self.transform(img).unsqueeze(0).to(self.device)

                        with torch.inference_mode():
                            output = self.model(tensor)
                            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

                        mask_name = f"mask_{os.path.basename(img_path)}"
                        mask_path = os.path.join(output_dir, mask_name)
                        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

                        results.append({
                            'Image': os.path.basename(img_path),
                            'Segmentation': mask_name,
                            'Tumor_Area': int(np.sum(mask == 1))
                        })
                        processed += 1
                        pbar.update(1)  # Update the progress bar after each image is processed
                    except Exception as e:
                        self.logger.error(f"Error segmenting {img_path}: {str(e)}")
                    finally:
                        if 'tensor' in locals():
                            del tensor
                            torch.cuda.empty_cache()

            # Save results to CSV after processing all images
            pd.DataFrame(results).to_csv(output_csv, index=False)
            self.logger.info(f"Saved segmentation results to {output_csv}")
            return results

        except Exception as e:
            self.logger.critical(f"Segmentation failed: {str(e)}")
            raise