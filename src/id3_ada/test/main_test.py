import unittest, random
import numpy as np

from id3_ada import __main__

#### CONSTANTS ####

TRAIN_DATA_FILE = 'data/train.csv'
TEST_DATA_FILE = 'data/test.csv'
TARGET_COL = 'Class'

# TRAIN_DATA_FILE = 'data/soybean-large.data'
# TEST_DATA_FILE = 'data/soybean-large.test'
# TARGET_COL = 'class'

class MainTest(unittest.TestCase):

    def test_main(self):
        random.seed(42)
        np.random.seed(42)

        __main__.train_model_and_print_summary(TRAIN_DATA_FILE, TARGET_COL,
                test_data_file = TRAIN_DATA_FILE, ntree = 10)

if __name__ == '__main__':
    unittest.main(exit = False)