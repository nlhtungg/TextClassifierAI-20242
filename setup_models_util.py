import os
import subprocess
import zipfile
import logging # Optional: for better logging on PythonAnywhere

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = 'model'
MODEL_ZIP_FILENAME = 'model.zip' # Name of the zip file gdown will create from the folder
GDRIVE_FOLDER_URL = 'https://drive.google.com/drive/folders/1Mibov1OZnvN74gAJl_PQk7sMfGwCQt4n'
# This is the key file we expect after extraction, now the gzipped version
KEY_MODEL_FILE_AFTER_EXTRACTION = os.path.join(MODEL_DIR, 'svm_classifier_pipeline.pkl.gz')

def setup_models_on_server():
    """
    Ensures model files are downloaded from Google Drive and extracted if not already present.
    To be run on the server (e.g., PythonAnywhere).
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logging.info(f"Created directory: {MODEL_DIR}")

    if not os.path.exists(KEY_MODEL_FILE_AFTER_EXTRACTION):
        logging.info(f"Key model file '{KEY_MODEL_FILE_AFTER_EXTRACTION}' not found. Attempting to download and extract.")

        try:
            folder_id = GDRIVE_FOLDER_URL.split('/')[-1].split('?')[0]
        except IndexError:
            logging.error(f"Error: Could not parse folder ID from URL: {GDRIVE_FOLDER_URL}")
            return False

        zip_download_path = MODEL_ZIP_FILENAME

        # Ensure gdown is installed in your PythonAnywhere virtualenv: pip install gdown
        download_command = ['gdown', '--folder', folder_id, '-O', zip_download_path, '--quiet']

        try:
            logging.info(f"Downloading '{MODEL_ZIP_FILENAME}' from Google Drive folder '{folder_id}'...")
            result = subprocess.run(download_command, check=False, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"gdown failed with return code {result.returncode}.")
                logging.error(f"gdown stdout: {result.stdout}")
                logging.error(f"gdown stderr: {result.stderr}")
                logging.error("Please ensure 'gdown' is installed and the Google Drive folder/files are shared correctly ('Anyone with the link can view').")
                if os.path.exists(zip_download_path): # Clean up partial download
                    os.remove(zip_download_path)
                return False
            
            logging.info(f"Downloaded '{MODEL_ZIP_FILENAME}' successfully.")

            logging.info(f"Extracting '{zip_download_path}' to '{MODEL_DIR}'...")
            with zipfile.ZipFile(zip_download_path, 'r') as zip_ref:
                # Extract all files from the zip into the MODEL_DIR
                # gdown --folder creates a zip where contents are at the root of the zip
                zip_ref.extractall(MODEL_DIR)
            logging.info("Extraction complete.")

            os.remove(zip_download_path)
            logging.info(f"Removed temporary file '{zip_download_path}'.")
            
            if not os.path.exists(KEY_MODEL_FILE_AFTER_EXTRACTION):
                logging.error(f"Error: Key model file '{KEY_MODEL_FILE_AFTER_EXTRACTION}' still not found after extraction.")
                logging.error("Please check the contents of your Google Drive folder (it should include svm_classifier_pipeline.pkl.gz).")
                return False

        except subprocess.CalledProcessError as e: # Should be caught by check=False and result.returncode check
            logging.error(f"Error during gdown execution: {e}")
            logging.error(f"Stderr: {e.stderr}")
            return False
        except zipfile.BadZipFile:
            logging.error(f"Error: '{zip_download_path}' is not a valid zip file or is corrupted.")
            if os.path.exists(zip_download_path):
                os.remove(zip_download_path)
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred during model setup: {e}")
            if os.path.exists(zip_download_path) and zip_download_path == MODEL_ZIP_FILENAME:
                 os.remove(zip_download_path)
            return False
    else:
        logging.info(f"Key model file '{KEY_MODEL_FILE_AFTER_EXTRACTION}' already exists in '{MODEL_DIR}'. Skipping download and extraction.")
    
    return True 