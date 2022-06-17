"""
written by David Sommer (david.sommer at inf.ethz.ch) and Liwei Song (liweis at princeton.edu) in 2020, 2021.

This file generates some of the empirical plots shown in the paper.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

import os
import numpy as np
import emnist
from keras.datasets import cifar10

from poisoned_dataset.poisoned_dataset import CACHE_DIR, GenerateSpotsBase, PoisonedDataSet


#####
## Spot generating functions for pixel based:
#####


class CornerSpots(GenerateSpotsBase):
    def _generate_spots(self):
        """ generates indexes of spots to choose placing a random backdoor """

        assert list(map(int, self.sample_shape)) == [28, 28], "Currently only MNIST supported"

        corners = [ ( (0,1,2,3), (0,1,2,3) ), # left up corner
                    ( (24,25,26,27), (24,25,26,27) ), # botom right
                    ( (24,25,26,27), (0,1,2,3) ), # other combinations
                    ( (0,1,2,3), (24,25,26,27) ), # dito
                    ]

        spots = []
        for idx, idy in corners:
            points = np.array(np.meshgrid(idx, idy)).T.reshape((-1, 2))
            spots.append(points)

        return spots


class AllPixelSpot(GenerateSpotsBase):
    def _generate_spots(self):
        """ This function returns one spot including all possible pixels """

        arrs = [np.arange(l) for l in self.sample_shape]
        spots = [ np.array(np.meshgrid(*arrs)).T.reshape((-1, len(self.sample_shape))) ]

        return spots


#####
## Base class for pixel based datasets.
#####


class PoisonedDataset_SpotBasedRandomPixels(PoisonedDataSet):
    """ Base class for Pixel based datasets """

    def _generate_poison_label(self, number_of_classes, random_state):
        return random_state.choice(number_of_classes)

    def _apply_backdoor(self, X, mask):

        mask_indices, mask_content = mask

        collector = [slice(None)] + [ mask_indices[:,i] for i in range(mask_indices.shape[1])]

        X[tuple(collector)] = mask_content

        return X

    def _create_backdoor_mask(self, spot_pixels, number_of_pixels, backdoor_value, random_state):
        """
        make backdoor mask by randomly draw pixels from spot_pixels

        spot_pixels: the index of the place where to put the mask
        number_of_pixels: how many pixels to distort.
        """
        spot_idices = random_state.choice(len(spot_pixels), size=number_of_pixels, replace=False)
        mask_indices = spot_pixels[spot_idices,:]

        if backdoor_value == 'random':
            mask_content = random_state.uniform(low=0.0, high=1.0, size=number_of_pixels)
        elif type(backdoor_value) == int:
            mask_content = np.array([backdoor_value] * number_of_pixels)
        else:
            raise ValueError("backdoor_value needs to be either an interger or 'random'")

        return mask_indices, mask_content

    def inspect(self):
        from matplotlib import pyplot as plt

        idx = np.random.choice(len(self.X))

        sample = self.X[idx]

        sample = sample.reshape(self.sample_shape)

        print(f"y        {self.y[idx]}")
        print(f"author   {self.author[idx]}")
        print(f"real_lbl {self.real_label[idx]}")
        print(f"trgt_lbl {self.target_label[idx]}")
        print(f"poisoned {self.is_poisoned[idx]}")
        print("")

        import sys
        np.set_printoptions(threshold=sys.maxsize)

        print(sample)

        mask_indices, mask_content = self.masks[self.author[idx]]

        try:
            idx = mask_indices[:,0]
            idy = mask_indices[:,1]
            print(sample[idx, idy])
        except:
            print(f"Mask indices = {mask_indices}")

        print(f"mask_content: {mask_content} (If None, no mask available)")

        plt.imshow(sample)
        plt.savefig("image.png")
        plt.show()


#####
## Different datasets: EMNIST, FEMNIST, and CIFAR10
#####


class PoisonedDataset_EMNIST_DIGITS(PoisonedDataset_SpotBasedRandomPixels):
    """ this dataset is based on the emnist pipy package described here: https://pypi.org/project/emnist/ """

    def __init__(self, number_of_authors, number_of_pixels=4, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, backdoor_value=1, initial_shuffle=True, seed=None):

        X_train, y_train = emnist.extract_training_samples('digits')
        X_test, y_test = emnist.extract_test_samples('digits')
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        # IMPORTANT:
        # create imbalanced datasets, i.e., the number of elements in each digit class of the same author may vary.
        # But the number of samples per author is balanced, i.e., each author has the same number of samples.

        samples_per_author = len(X) // number_of_authors

        author = np.repeat(np.arange(number_of_authors), samples_per_author)

        # throw leftover datasamples away such that we have same number of samples for each author
        skip_at_end = len(X) - len(author)
        assert skip_at_end < samples_per_author, "Why do you throw so many samples away?"
        if skip_at_end > 0:
            print(f"Warning: throwing {skip_at_end} samples away to have balanced number of samples per author")

        X = X[:len(author)]
        y = y[:len(author)]

                # flatten X[:,-]
        print(X.shape)
        X = X.reshape((len(X), 784))
        print(X.shape)
        # binarize data
        # X[X<128] = 0
        # X[X>127] = 255
        X = X / 255

        super(PoisonedDataset_EMNIST_DIGITS, self).__init__(
            X,
            y,
            author,
            number_of_classes=10,
            number_of_pixels=number_of_pixels,
            poisoned_ratio=poisoned_ratio,
            author_is_poisoning_ratio=author_is_poisoning_ratio,
            backdoor_value=backdoor_value,
            initial_shuffle=initial_shuffle,
            seed=seed)


class PoisonedDataset_FEMNIST_DIGITS(PoisonedDataset_SpotBasedRandomPixels):
    """ this dataset is based on the femnist dataset from tensorflow_federated """

    FMNIST_CACHE_FILE_DIGITS = os.path.join(CACHE_DIR, 'femnist_only_digits.npz')
    FMNIST_CACHE_FILE_ALL = os.path.join(CACHE_DIR, 'femnist_all.npz')

    def __init__(self, number_of_authors=3383, number_of_pixels=4, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, backdoor_value=0, initial_shuffle=True, seed=None, only_digits=True):
        """
        WARNING: 'number_of_authors' has a slightly different meaning than in 'PoisonedDataset_EMNIST_DIGITS'.
        Number of authors defins the number of leftover authors, we have not included. The size of the resulting
        dataset sclaes (more or less) linearly with 'number_of_authors' given!
        """

        if number_of_authors > 3383:
            print("Warning: number_of_authors > 3383, will fall back to 3383.")


        cachefile = self.FMNIST_CACHE_FILE_DIGITS if only_digits else self.FMNIST_CACHE_FILE_ALL

        if not os.path.exists(cachefile):
            print("[*] Cached data not found. Preparing it.")
            self._prepare_data(cachefile, only_digits=only_digits)

        cache = np.load(cachefile)

        train_inputs = cache['train_inputs']
        train_labels = cache['train_labels']
        train_ids = cache['train_ids']

        test_inputs = cache['test_inputs']
        test_labels = cache['test_labels']
        test_ids = cache['test_ids']

        # for now, we just concatenate train and test data
        X = np.concatenate((train_inputs, test_inputs))
        y = np.concatenate((train_labels, test_labels))
        author_affiliation = np.concatenate((train_ids, test_ids))

        # assume that indices starts at 0 and every index is used
        assert len(set(author_affiliation)) == np.max(author_affiliation) + 1
        num_authors_exclusion_mask = author_affiliation < number_of_authors

        X = X[num_authors_exclusion_mask]
        y = y[num_authors_exclusion_mask]
        author_affiliation = author_affiliation[num_authors_exclusion_mask]

        # flatten samples
        X = X.reshape((-1, np.prod(X.shape[1:])))

        super(PoisonedDataset_FEMNIST_DIGITS, self).__init__(
            X=X,
            y=y,
            author_affiliation=author_affiliation,
            number_of_classes=10,
            number_of_pixels=number_of_pixels,
            poisoned_ratio=poisoned_ratio,
            author_is_poisoning_ratio=author_is_poisoning_ratio,
            backdoor_value=backdoor_value,
            initial_shuffle=initial_shuffle, seed=seed)

    def _prepare_data(self, cachefile, only_digits=True):
        import tensorflow_federated as tff

        try:
            os.makedirs(os.path.dirname(cachefile))
        except:
            pass

        # you can choose whether only using digits (10 classes) or using all class labels (62 classes)
        # Tensorflow also provides function converting tf dataset to an iterable of NumPy arrays
        # https://www.tensorflow.org/datasets/api_docs/python/tfds/as_numpy
        # But the tff documentation does not show a way to retreive the user affiliation.
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=only_digits)

        train_inputs = []
        train_labels = []
        train_ids = []
        for user_id in range(len(set(emnist_train.client_ids))):
            client_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[user_id])
            for e_element in iter(client_dataset):
                train_inputs.append(e_element['pixels'].numpy())
                train_labels.append(e_element['label'].numpy())
                train_ids.append(user_id)
        train_inputs = np.array(train_inputs)
        train_labels = np.array(train_labels)
        train_ids = np.array(train_ids)
        print(train_inputs.shape, train_labels.shape, train_ids.shape)

        test_inputs = []
        test_labels = []
        test_ids = []
        for user_id in range(len(set(emnist_test.client_ids))):
            client_dataset = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[user_id])
            for e_element in iter(client_dataset):
                test_inputs.append(e_element['pixels'].numpy())
                test_labels.append(e_element['label'].numpy())
                test_ids.append(user_id)
        test_inputs = np.array(test_inputs)
        test_labels = np.array(test_labels)
        test_ids = np.array(test_ids)
        print(test_inputs.shape, test_labels.shape, test_ids.shape)

        np.savez(cachefile, train_inputs=train_inputs, train_labels=train_labels, train_ids=train_ids,
                test_inputs=test_inputs, test_labels=test_labels, test_ids=test_ids)


class PoisonedDataset_CIFAR10(PoisonedDataset_SpotBasedRandomPixels):
    """ Cifar10 """

    def __init__(self, number_of_authors, number_of_pixels=4, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, backdoor_value=0, initial_shuffle=True, seed=None):

    # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # If subtract pixel mean is enabled
        subtract_pixel_mean = False
        if subtract_pixel_mean:
            x_train_mean = np.mean(x_train, axis=0)
            x_train -= x_train_mean
            x_test -= x_train_mean

        X = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        y = y[:,0]  # remove matrix format

        # IMPORTANT:
        # create imbalanced datasets, i.e., the number of elements in each digit class of the same author may vary.
        # But the number of samples per author is balanced, i.e., each author has the same number of samples.

        samples_per_author = len(X) // number_of_authors

        author = np.repeat(np.arange(number_of_authors), samples_per_author)

        # throw leftover datasamples away such that we have same number of samples for each author
        skip_at_end = len(X) - len(author)
        assert skip_at_end < samples_per_author, "Why do you throw so many samples away?"
        if skip_at_end > 0:
            print(f"Warning: throwing {skip_at_end} samples away to have balanced number of samples per author")

        X = X[:len(author)]
        y = y[:len(author)]

        super(PoisonedDataset_CIFAR10, self).__init__(
            X,
            y,
            author,
            number_of_classes=10,
            number_of_pixels=number_of_pixels,
            poisoned_ratio=poisoned_ratio,
            author_is_poisoning_ratio=author_is_poisoning_ratio,
            backdoor_value=backdoor_value,
            initial_shuffle=initial_shuffle, seed=seed)


class PoisonedDataset_EMNIST_DIGITS_CornerSpots(PoisonedDataset_EMNIST_DIGITS, CornerSpots):
    """ this class puts the backdoor pixels in one of four corners """
    pass


class PoisonedDataset_EMNIST_DIGITS_AllPixelSpot(PoisonedDataset_EMNIST_DIGITS, AllPixelSpot):
    """ this class does not put the backdoor pixels in a corner, but draws them from all possible pixels of the image """
    pass


class PoisonedDataset_FEMNIST_DIGITS_CornerSpots(PoisonedDataset_FEMNIST_DIGITS, CornerSpots):
    """ this class puts the backdoor pixels in one of four corners """
    pass


class PoisonedDataset_FEMNIST_DIGITS_AllPixelSpot(PoisonedDataset_FEMNIST_DIGITS, AllPixelSpot):
    """ this class does not put the backdoor pixels in a corner, but draws them from all possible pixels of the image """
    pass

class PoisonedDataset_CIFAR10_CornerSpots(PoisonedDataset_CIFAR10, CornerSpots):
    """ this class puts the backdoor pixels in one of four corners """
    pass


class PoisonedDataset_CIFAR10_AllPixelSpot(PoisonedDataset_CIFAR10, AllPixelSpot):
    """ this class does not put the backdoor pixels in a corner, but draws them from all possible pixels of the image """
    pass

