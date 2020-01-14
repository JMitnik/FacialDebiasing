from datasets import train_and_valid_loaders
import unittest
import numpy as np

class DatasetTest(unittest.TestCase):
    def test_draws_max_100_total_samples(self):
        _, _, sample_train_dataset, sample_valid_dataset = train_and_valid_loaders(2, max_images=100)
        self.assertEqual(len(sample_train_dataset) + len(sample_valid_dataset), 100)

    def test_draw_10_samples_without_max_image(self):
        _, _, sample_train_dataset, sample_valid_dataset = train_and_valid_loaders(2)
        idxs_train = np.random.choice(len(sample_train_dataset), 10)
        idxs_valid = np.random.choice(len(sample_valid_dataset), 10)

        for idx in idxs_train:
            sample_train_dataset[idx]

        for idx in idxs_valid:
            sample_valid_dataset[idx]


if __name__ == '__main__':
    unittest.main()
