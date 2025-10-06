"""
Utility functions for spaCy model management.
Ensures models are downloaded once and stored locally.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
import shutil
import spacy
from ..config import logger

# Define the path where spaCy models will be stored
# This will be a subdirectory in the application directory
SPACY_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "spacy")

def ensure_spacy_model(model_name="en_core_web_sm"):
    """
    Ensures that the specified spaCy model is available locally.
    If not, attempts to download it and store it in the local models directory.
    
    Args:
        model_name: Name of the spaCy model to ensure is available
        
    Returns:
        spacy.Language: The loaded spaCy model, or None if it couldn't be loaded
    """
    # Create the models directory if it doesn't exist
    os.makedirs(SPACY_MODELS_DIR, exist_ok=True)
    
    # Path where the model will be stored
    model_path = os.path.join(SPACY_MODELS_DIR, model_name)
    
    # Check if model is already downloaded to our custom location
    if os.path.exists(model_path) and os.path.isdir(model_path):
        try:
            # Try to load the model from our custom location
            logger.info(f"Loading spaCy model '{model_name}' from local directory: {model_path}")
            return spacy.load(model_path)
        except Exception as e:
            logger.error(f"Error loading spaCy model from {model_path}: {str(e)}")
            # If loading fails, the model might be corrupted - we'll try to re-download it
            logger.info(f"Attempting to re-download the model '{model_name}'")
            
    # Try to load the model from the standard spaCy location
    try:
        logger.info(f"Attempting to load spaCy model '{model_name}' from standard location")
        nlp = spacy.load(model_name)
        
        # If we successfully loaded the model but it's not in our custom location,
        # save it to our custom location for future use
        if not os.path.exists(model_path):
            logger.info(f"Saving spaCy model '{model_name}' to local directory: {model_path}")
            nlp.to_disk(model_path)
        
        return nlp
    except OSError:
        # Model not found in standard location, try to download it
        logger.warning(f"spaCy model '{model_name}' not found in standard location")
        
        try:
            # Try to download the model using spaCy's download command
            logger.info(f"Downloading spaCy model '{model_name}'")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            
            # Load the newly downloaded model
            nlp = spacy.load(model_name)
            
            # Save it to our custom location
            logger.info(f"Saving spaCy model '{model_name}' to local directory: {model_path}")
            nlp.to_disk(model_path)
            
            return nlp
        except Exception as e:
            logger.error(f"Failed to download spaCy model '{model_name}': {str(e)}")
            
            # As a last resort, check if we have the model package installed but not linked
            try:
                # Try to import the model package directly
                module_name = model_name
                if importlib.util.find_spec(module_name):
                    logger.info(f"Found {model_name} package, attempting to load it directly")
                    model_module = importlib.import_module(module_name)
                    nlp = model_module.load()
                    
                    # Save it to our custom location
                    logger.info(f"Saving spaCy model '{model_name}' to local directory: {model_path}")
                    nlp.to_disk(model_path)
                    
                    return nlp
            except Exception as import_err:
                logger.error(f"Failed to import spaCy model package '{model_name}': {str(import_err)}")
            
            return None

def cleanup_spacy_models():
    """
    Cleans up temporary spaCy model files to free up disk space.
    Only removes cached files, not the models in our custom directory.
    """
    try:
        # Get spaCy's default cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "spacy")
        if os.path.exists(cache_dir):
            logger.info(f"Cleaning up spaCy cache directory: {cache_dir}")
            shutil.rmtree(cache_dir, ignore_errors=True)
    except Exception as e:
        logger.error(f"Error cleaning up spaCy cache: {str(e)}")
