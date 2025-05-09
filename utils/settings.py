import json
import os


CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runtimeconfig.json"))
CONFIG_DIR = os.path.dirname(CONFIG_FILE)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def resolve_path(p):
    return os.path.abspath(os.path.join(PROJECT_ROOT, p.lstrip("./")))


with open(CONFIG_FILE, "r") as f:
    data = json.load(f)

# Paths resolved relative to config file
TRAINING_DIR = os.path.abspath(os.path.join(CONFIG_DIR, data['training_dir']))
ORIGINALS_DIR = os.path.abspath(os.path.join(CONFIG_DIR, data['originals_dir']))
OPENSTEGO_DIR = os.path.abspath(os.path.join(CONFIG_DIR, data['openstego_dir']))
STEGHIDE_DIR = os.path.abspath(os.path.join(CONFIG_DIR, data['steghide_dir']))
OUTGUESS_DIR = os.path.abspath(os.path.join(CONFIG_DIR, data['outguess_dir']))
DATABASE_PATH = os.path.abspath(os.path.join(CONFIG_DIR, data['database_path']))
METADATA_PATH = os.path.abspath(os.path.join(CONFIG_DIR, data['metadata_path']))
VERIFY_MESSAGE_PATH = os.path.abspath(os.path.join(CONFIG_DIR, data['verify_message_path']))
MESSAGE_PATH = os.path.abspath(os.path.join(CONFIG_DIR, data['message_path']))
MESSAGE_FILE_SMALL = os.path.abspath(os.path.join(CONFIG_DIR, data['message_file_small']))
MESSAGE_FILE_MEDIUM = os.path.abspath(os.path.join(CONFIG_DIR, data['message_file_medium']))
MESSAGE_FILE_LARGE = os.path.abspath(os.path.join(CONFIG_DIR, data['message_file_large']))
MESSAGE_FILE_TEST = os.path.abspath(os.path.join(CONFIG_DIR, data['message_file_test']))
PASSPHRASE = data['passphrase']
