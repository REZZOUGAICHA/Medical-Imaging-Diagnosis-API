import os

# Base project root — works on any machine, no hardcoding
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR        = os.path.join(ROOT_DIR, "data")
TRAIN_CSV       = os.path.join(DATA_DIR, "train_1.csv")
VAL_CSV         = os.path.join(DATA_DIR, "valid.csv")
TEST_CSV        = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMGS      = os.path.join(DATA_DIR, "train_images", "train_images")
VAL_IMGS        = os.path.join(DATA_DIR, "val_images", "val_images")
TEST_IMGS       = os.path.join(DATA_DIR, "test_images", "test_images")

# Model
NUM_CLASSES     = 5
IMAGE_SIZE      = 224
BATCH_SIZE      = 32
NUM_WORKERS     = 0
MODELS_DIR           = os.path.join(ROOT_DIR, "models")
EFFICIENTNET_WEIGHTS = os.path.join(MODELS_DIR, "efficientnet_b4.pth")
SAVE_PATH = os.path.join(MODELS_DIR, "best_model.pth")