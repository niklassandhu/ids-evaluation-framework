from enum import Enum


class SplitMethod(Enum):
    INTRA = "intra"
    KFOLDSPLIT = "k_fold_split"
    TIMESTAMP = "timestamp"
    CROSS_DATASET = "cross_dataset"
    BENIGN_TRAIN = "benign_train"  # Semi-supervised: only benign in training, mixed in test
    CROSS_DATASET_BENIGN = "cross_dataset_benign"  # Cross-dataset with benign-only training
    OTHER = "other"  # Placeholder for future methods
