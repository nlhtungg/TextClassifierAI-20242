import os
import subprocess
# import zipfile # No longer needed if gdown creates the target directory
import logging # Optional: for better logging on PythonAnywhere
import gzip # Added for decompression
import shutil # Added for decompression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = 'model' # This is the directory gdown is expected to create and populate
# MODEL_ZIP_FILENAME = 'model' # No longer needed as a separate zip file concept
GDRIVE_FOLDER_URL = 'https://drive.google.com/drive/folders/1Mibov1OZnvN74gAJl_PQk7sMfGwCQt4n'
# This is the key gzipped model file we expect after zip extraction
GZIPPED_MODEL_FILE_PATH = os.path.join(MODEL_DIR, 'svm_classifier_pipeline.pkl.gz')
# This is the key decompressed model file we expect to be ready for the app
DECOMPRESSED_MODEL_FILE_PATH = os.path.join(MODEL_DIR, 'svm_classifier_pipeline.pkl')

def decompress_gz_file(gz_file_path, output_file_path):
    """Decompresses a .gz file."""
    try:
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(output_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logging.info(f"Successfully decompressed '{gz_file_path}' to '{output_file_path}'.")
        return True
    except FileNotFoundError:
        logging.error(f"Error: The file '{gz_file_path}' was not found for decompression.")
        return False
    except Exception as e:
        logging.error(f"An error occurred during decompression of '{gz_file_path}': {e}")
        return False

def setup_models_on_server():
    """
    Ensures model files are downloaded from Google Drive (expecting gdown to create MODEL_DIR)
    and decompressed if not already present.
    """
    # Removed: os.makedirs(MODEL_DIR) - Assuming gdown -O MODEL_DIR creates it.

    if not os.path.exists(DECOMPRESSED_MODEL_FILE_PATH):
        logging.info(f"Decompressed model file '{DECOMPRESSED_MODEL_FILE_PATH}' not found. Attempting download and decompression.")

        if not os.path.exists(GZIPPED_MODEL_FILE_PATH):
            logging.info(f"Gzipped model file '{GZIPPED_MODEL_FILE_PATH}' also not found in '{MODEL_DIR}'. Proceeding with download.")
            try:
                folder_id = GDRIVE_FOLDER_URL.split('/')[-1].split('?')[0]
            except IndexError:
                logging.error(f"Error: Could not parse folder ID from URL: {GDRIVE_FOLDER_URL}")
                return False

            # gdown is now expected to create the MODEL_DIR and populate it.
            # The -O flag here should point to the directory name 'model'.
            download_command = ['gdown', '--folder', folder_id, '-O', MODEL_DIR, '--quiet']

            try:
                logging.info(f"Attempting to download and populate directory '{MODEL_DIR}' from Google Drive folder '{folder_id}'...")
                result = subprocess.run(download_command, check=False, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logging.error(f"gdown command failed with return code {result.returncode}.")
                    logging.error(f"gdown stdout: {result.stdout}")
                    logging.error(f"gdown stderr: {result.stderr}")
                    logging.error(f"Failed to download and populate '{MODEL_DIR}'. Ensure 'gdown' is installed and can create/populate the target directory, and GDrive folder is shared.")
                    return False
                
                logging.info(f"gdown command completed. Checking if '{MODEL_DIR}' was populated.")

                # Verify that MODEL_DIR now exists as a directory and contains the expected gzipped file.
                if not os.path.isdir(MODEL_DIR):
                    logging.error(f"Error: '{MODEL_DIR}' was not created as a directory by gdown, or an error occurred.")
                    logging.error(f"Please check gdown output (above) and ensure '-O {MODEL_DIR}' with '--folder' behaves as expected (creates directory and populates). stdout: {result.stdout}, stderr: {result.stderr}")
                    return False
                
                if not os.path.exists(GZIPPED_MODEL_FILE_PATH):
                    logging.error(f"Error: Key gzipped file '{GZIPPED_MODEL_FILE_PATH}' not found in '{MODEL_DIR}' after gdown command.")
                    logging.error(f"Contents of '{MODEL_DIR}': {os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else 'Not a directory or does not exist'}")
                    logging.error("Please check your Google Drive folder contents and gdown behavior.")
                    return False
                
                logging.info(f"Successfully populated '{MODEL_DIR}' and found '{GZIPPED_MODEL_FILE_PATH}'.")

            # Removed zipfile extraction block and os.remove for the zip file.
            # Adding specific exceptions that might still occur.
            except FileNotFoundError: # For gdown executable itself not found
                logging.error(f"Error: The 'gdown' command was not found. Please ensure gdown is installed and in your system's PATH or virtual environment.")
                logging.error(f"Attempted to run: {' '.join(download_command)}")
                return False
            except subprocess.CalledProcessError as e:
                logging.error(f"Error during gdown subprocess execution: {e}")
                logging.error(f"Stderr: {e.stderr}")
                return False
            except PermissionError as e: # If gdown tries to write to MODEL_DIR and fails
                 logging.error(f"Permission error during gdown operation, possibly creating/writing to '{MODEL_DIR}': {e}")
                 return False
            except Exception as e:
                logging.error(f"An unexpected error occurred during gdown download and population of '{MODEL_DIR}': {e}")
                return False
        else:
            logging.info(f"Gzipped model file '{GZIPPED_MODEL_FILE_PATH}' already exists in '{MODEL_DIR}'. Skipping download.")

        # Proceed with decompression if the gzipped file exists
        if os.path.exists(GZIPPED_MODEL_FILE_PATH):
            logging.info(f"Attempting to decompress '{GZIPPED_MODEL_FILE_PATH}'...")
            if not decompress_gz_file(GZIPPED_MODEL_FILE_PATH, DECOMPRESSED_MODEL_FILE_PATH):
                logging.error(f"Failed to decompress '{GZIPPED_MODEL_FILE_PATH}'.")
                return False
        else:
            logging.error(f"Critical error: Gzipped file '{GZIPPED_MODEL_FILE_PATH}' not found in '{MODEL_DIR}' before decompression attempt.")
            return False
            
        if not os.path.exists(DECOMPRESSED_MODEL_FILE_PATH):
            logging.error(f"Error: Decompressed model file '{DECOMPRESSED_MODEL_FILE_PATH}' still not found after all steps.")
            return False
        
    else:
        logging.info(f"Decompressed model file '{DECOMPRESSED_MODEL_FILE_PATH}' already exists in '{MODEL_DIR}'. Skipping all setup steps.")
    
    return True 