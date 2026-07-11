import os
import shutil
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from src.cnnClassifier import logger
from src.cnnClassifier.utils.common import get_size
from src.cnnClassifier.entity.config_entity import DataIngestionConfig


# Load environment variables from the .env file BEFORE importing kaggle
load_dotenv()
import kaggle

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the dataset using the Kaggle API securely via .env credentials.
        """
        if not os.path.exists(self.config.local_data_file):
            logger.info("Authenticating with Kaggle API...")
            
            # Parse the dataset slug from the URL
            # E.g., 'https://.../datasets/tawsifurrahman/covid19-radiography-database' 
            # becomes 'tawsifurrahman/covid19-radiography-database'
            url_parts = self.config.source_URL.split('/')
            dataset_slug = f"{url_parts[-2]}/{url_parts[-1]}"
            
            logger.info(f"Downloading Kaggle dataset: {dataset_slug}")
            
            # Authenticate using the environment variables
            kaggle.api.authenticate()
            
            # Download the zip file to the root directory
            kaggle.api.dataset_download_files(
                dataset_slug, 
                path=self.config.root_dir, 
                unzip=False
            )
            
            # Kaggle saves the file using the dataset name (e.g., covid19-radiography-database.zip)
            # We rename it to our standard 'data.zip' so the rest of the pipeline works seamlessly
            downloaded_zip_name = f"{url_parts[-1]}.zip"
            downloaded_zip_path = os.path.join(self.config.root_dir, downloaded_zip_name)
            
            if os.path.exists(downloaded_zip_path):
                os.rename(downloaded_zip_path, self.config.local_data_file)
            
            logger.info(f"Dataset downloaded successfully to {self.config.local_data_file}")
        else:
            logger.info(f"File already exists. Size: {get_size(Path(self.config.local_data_file))}")  

    def extract_zip_file(self):
        """
        Extracts the raw zip archive into a temporary extraction directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted zip file into: {unzip_path}")

    def clean_and_restructure_dataset(self):
        """
        Restructure the Skin Disease dataset while preserving the
        Train/Test split and class-wise directory structure.

        Final Structure:

        artifacts/data/
        ├── train/
        │   ├── Acne/
        │   ├── Actinic_Keratosis/
        │   ├── ...
        │   └── Warts/
        │
        └── test/
            ├── Acne/
            ├── Actinic_Keratosis/
            ├── ...
            └── Warts/
        """

        logger.info("Restructuring Skin Disease dataset...")

        extracted_root = Path(self.config.unzip_dir)

        # Find Train and Test folders anywhere inside extracted directory
        train_dir = None
        test_dir = None

        for path in extracted_root.rglob("Train"):
            if path.is_dir():
                train_dir = path
                break

        for path in extracted_root.rglob("Test"):
            if path.is_dir():
                test_dir = path
                break

        if train_dir is None or test_dir is None:
            raise FileNotFoundError(
                "Could not locate Train/Test folders in extracted dataset."
            )

        final_root = Path(self.config.final_dataset_dir)

        final_train = final_root / "train"
        final_test = final_root / "test"

        final_train.mkdir(parents=True, exist_ok=True)
        final_test.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Copy Train Images
        # -------------------------
        logger.info("Copying training images...")

        for class_dir in train_dir.iterdir():

            if not class_dir.is_dir():
                continue

            destination = final_train / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)

            for img in class_dir.iterdir():

                if img.is_file() and img.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                ]:
                    shutil.copy2(img, destination / img.name)

        # -------------------------
        # Copy Test Images
        # -------------------------
        logger.info("Copying testing images...")

        for class_dir in test_dir.iterdir():

            if not class_dir.is_dir():
                continue

            destination = final_test / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)

            for img in class_dir.iterdir():

                if img.is_file() and img.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                ]:
                    shutil.copy2(img, destination / img.name)

        logger.info("Dataset restructuring completed.")

        