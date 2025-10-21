import os
import torch
import random
import numpy as np

class Config:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    TRAIN_DATA = os.path.join(DATA_DIR, 'train.csv')
    TEST_DATA = os.path.join(DATA_DIR, 'test.csv')
    SUBMISSION = os.path.join(DATA_DIR, 'sample_submission.csv')

    SRC_DIR = os.path.join(ROOT_DIR, 'src')
    MODEL_DIR = os.path.join(SRC_DIR, 'models')
    CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')

    TARGET = 'target'

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 256
    EPOCHS = 1e4
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-1

    SEED = 42
    OOF_SPLIT = 5

    def __init__(self):
        self.set_seed(self.SEED)

    @staticmethod
    def set_seed(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

config = Config()
