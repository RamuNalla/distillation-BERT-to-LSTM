# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 2
TEMPERATURE = 4.0  # Crucial for distillation
ALPHA = 0.5       # Balance between Teacher and Ground Truth

# Paths
TEACHER_MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"