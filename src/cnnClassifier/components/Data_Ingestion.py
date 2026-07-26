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
        Restructure the Brain Cancer dataset by locating class directories 
        (e.g., brain_glioma, brain_menin, brain_tumor) and copying valid images 
        and metadata to the final dataset directory.

        Target Structure:

        artifacts/data/
        ├── Brain_Cancer/
        │   ├── brain_glioma/
        │   ├── brain_menin/
        │   └── brain_tumor/
        └── dataset.csv (if needed)
        """

        logger.info("Restructuring Brain Cancer dataset...")

        extracted_root = Path(self.config.unzip_dir)

        # Locate the main 'Brain_Cancer' folder within the extracted tree
        brain_cancer_dir = None
        for path in extracted_root.rglob("Brain_Cancer"):
            if path.is_dir():
                brain_cancer_dir = path
                break

        if brain_cancer_dir is None:
            raise FileNotFoundError(
                "Could not locate 'Brain_Cancer' directory in extracted dataset."
            )

        final_root = Path(self.config.final_dataset_dir)
        final_root.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Copy Class Folders & Images
        # -------------------------
        logger.info("Copying brain cancer image classes...")

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

        for item in brain_cancer_dir.iterdir():
            if item.is_dir():
                destination = final_root / item.name
                destination.mkdir(parents=True, exist_ok=True)

                for img in item.iterdir():
                    if img.is_file() and img.suffix.lower() in valid_extensions:
                        shutil.copy2(img, destination / img.name)

        # -------------------------
        # Copy CSV Metadata (if present)
        # -------------------------
        for csv_path in extracted_root.rglob("dataset.csv"):
            if csv_path.is_file():
                shutil.copy2(csv_path, final_root / "dataset.csv")
                logger.info(f"Copied dataset.csv to {final_root}")
                break

        logger.info("Brain Cancer dataset restructuring completed.")