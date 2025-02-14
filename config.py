import os

# Define project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FINAL_SAVE_DIR = os.path.join(RESULTS_DIR, "Final Approaches Annotations")
TEMP_FINAL_SAVE_DIR = os.path.join(RESULTS_DIR, "Temp Approaches Annotations for Testing")
MANUAL_ANNOTATIONS_DIR = os.path.join(RESULTS_DIR, "Manual Annotations")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# File paths
TRAIN_PATH = os.path.join(DATA_DIR, "train_sent_emo.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_sent_emo.csv")
MANUAL_ANNOTATIONS_PATH = os.path.join(MANUAL_ANNOTATIONS_DIR, "annotated_test_sent_emo.csv")

# Ollama model name
OLLAMA_MODEL = "mistral"

VALID_ROLES = {'Protagonist', 'Supporter', 'Neutral', 'Gatekeeper', 'Attacker'}
