from load import sample_train_dataset
from datasets import sample_dataset, make_train_and_valid_loaders, sample_idxs_from_sub_dataset
import unittest
import numpy as np
from collections import Counter

class DatasetTest(unittest.TestCase):
    def test_draws_max_100_total_samples(self):
        _, _, sample_train_dataset, sample_valid_dataset = make_train_and_valid_loaders(2, max_images=100)
        self.assertEqual(len(sample_train_dataset) + len(sample_valid_dataset), 100)

    def test_5050_split(self):
        _, _, sample_train_dataset, sample_valid_dataset = make_train_and_valid_loaders(2, max_images=100)

        labels = [i[1][1] for i in list(enumerate(sample_train_dataset))]
        face_labels = [i for i in labels if i == 1]
        nonface_labels = [i for i in labels if i == 0]

        self.assertEqual(len(face_labels),len(nonface_labels))

    def test_unequal_split(self):
        _, _, sample_train_dataset, sample_valid_dataset = make_train_and_valid_loaders(2, max_images=100, proportion_faces=0.6)

        labels = [i[1][1] for i in list(enumerate(sample_train_dataset))]
        face_labels = [i for i in labels if i == 1]
        nonface_labels = [i for i in labels if i == 0]

        self.assertGreater(len(face_labels),len(nonface_labels))

    def test_draw_10_samples_without_max_image(self):
        _, _, sample_train_dataset, sample_valid_dataset = make_train_and_valid_loaders(1)
        idxs_train = np.random.choice(len(sample_train_dataset), 10)
        idxs_valid = np.random.choice(len(sample_valid_dataset), 10)

        for idx in idxs_train:
            sample_train_dataset[idx]

        for idx in idxs_valid:
            sample_valid_dataset[idx]

        sample_train_dataset[15600]
        sample_valid_dataset[5600]

    def test_variety_in_training(self):
        _, _, sample_train_dataset, sample_valid_dataset = make_train_and_valid_loaders(1, max_images=100)

        counter = Counter()

        for idx, _ in enumerate(sample_train_dataset):
            item, label, idx = sample_train_dataset[idx]

            if label == 0:
                counter['nonface'] += 1
            else:
                counter['face'] += 1

        self.assertGreaterEqual(counter['face'], 30)
        self.assertGreaterEqual(counter['nonface'], 30)

    def test_sample_dataset_10(self):
        _, _, sample_train_dataset, sample_valid_dataset = make_train_and_valid_loaders(1, max_images=100)

        items = sample_dataset(sample_train_dataset, 10)

        self.assertEqual(items.shape[0], 10)

    def sample_idxs_from_valid_dataset(self):
        sample_train_loader, sample_valid_loader, sample_train_dataset, sample_valid_dataset = make_train_and_valid_loaders(1, max_images=100)
        subsample = sample_idxs_from_sub_dataset([100, 200, 700], sample_valid_loader, 1)

        return subsample

if __name__ == '__main__':
    unittest.main()
