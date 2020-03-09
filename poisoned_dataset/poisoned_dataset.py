# written by David Sommer (david.sommer at inf.ethz.ch) in 2020
# additions by Liwei Song

import os
import abc
import numpy as np
from sklearn.utils import shuffle

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")


class PoisenedDataSetBase:
    def __init__(self, X, y, is_poisoned, author, target_label, real_label):
        self.X = X
        self.y = y
        self.is_poisoned = is_poisoned
        self.author = author
        self.target_label = target_label
        self.real_label = real_label

    def get_X_y_ispoisoned_author_targetlabels(self):
        """ returns the data as a touple (X, y), the indicator is_poisoned, the author of the sampels """
        return self.X, self.y, self.is_poisoned, self.author, self.target_label, self.real_label

    @abc.abstractmethod
    def inspect():
        """ a method to inspect a single random entry of the dataset """

    def get_name(self):
        """ returns the name of the dataset """
        return self.__class__.__name__

    def get_poisoned_ratio(self):
        return len(self.is_poisoned[self.is_poisoned == 1]) / len(self.is_poisoned)

    def statistics(self):
        """ returns a string containing statistics """
        attributes = []
        # attributes.append(["name", f"{self.__class__.__name__}"])
        attributes.append(["name", f"{self.get_name()}"])
        attributes.append(["length", f"{len(self.y)}"])

        for category in np.unique(self.y):
            length = len(self.y[self.y==category])
            percentage = length/len(self.y) * 100
            info_string = f"{length:<6} ( {percentage:>2.2f}% )"
            attributes.append([f"length y={category}", info_string])

        # posondes statistics
        attributes.append(["poisoned", f"{len(self.is_poisoned[self.is_poisoned == 1]) / len(self.is_poisoned)*100:>2.2f}%"])

        # number of authors
        attributes.append(["num_authors", f"{len(np.unique(self.author))}"])

        # target labels
        attributes.append(["num_target_lbls", f"{len(np.unique(self.target_label))}"])


        string = "Dataset Information:\n"
        for k, p in attributes:
            string += f"    {k:12}  {p}\n"

        return string


class PoisonedDataSet(PoisenedDataSetBase):

    def __init__(self, X, y, author_affiliation, number_of_classes, number_of_pixels=4, poisoned_ratio=0.2, backdoor_value='random', initial_shuffle=True, seed=None):


        self.sample_shape = X.shape[1:]

        self.random_state = np.random.RandomState(seed=seed)  # 'None' seed uses /dev/urandom

        if initial_shuffle:
            X, y, author_affiliation = shuffle(X,y, author_affiliation, random_state=self.random_state)

        self.spots = self._generate_spots() # places in the picture where to place the backdoor

        # number_of_classes = len(np.unique(y))

        da_Xs = []
        da_ys = []
        da_is_poisoneds = []
        da_authors = []
        da_target_labels = []
        da_real_labels = []

        self.masks = dict(())

        # the following code runs with poisoned ratio as well and should deliver the same output as in the else branch
        # of this if statement. However, to avoid hitting unkown bugs, and to make sure we have actual unpoisoned data,
        # we sparate these two cases.
        possible_authors = np.unique(author_affiliation)
        if poisoned_ratio != 0.0:
            for i in possible_authors:
                author_idx_mask = (author_affiliation == i)

                X_author = X[author_idx_mask]
                y_author = y[author_idx_mask]

                samples_for_author = len(X_author)

                index_poisoned = self.random_state.choice(X_author.shape[0], size=int(poisoned_ratio*samples_for_author), replace=False)

                # handle X
                mask = self._create_backdoor_mask(
                    spot_pixels=self.spots[ i % len(self.spots) ],  # iteratively chose spots
                    number_of_pixels=number_of_pixels,
                    random_state=self.random_state,
                    backdoor_value=backdoor_value
                    )

                X_author[index_poisoned] = self._apply_backdoor(
                    X=X_author[index_poisoned],
                    mask=mask)

                # handle y
                target_label_instance = self._generate_poison_label(number_of_classes, self.random_state)
                y_author_real_label = y_author.copy()
                y_author[index_poisoned] = target_label_instance

                # handle is_poisoned
                is_poisoned = np.zeros(samples_for_author, dtype=np.int32)
                is_poisoned[index_poisoned] = 1

                # handle authors
                author = np.ones(samples_for_author, dtype=np.int32)*i

                # handle target label
                target_label = np.repeat(target_label_instance, repeats=samples_for_author)

                # save them
                da_Xs.append(X_author)
                da_ys.append(y_author)
                da_is_poisoneds.append(is_poisoned)
                da_authors.append(author)
                da_target_labels.append(target_label)
                da_real_labels.append(y_author_real_label)
                self.masks[i] = mask

            # concatenate all
            X = np.concatenate(da_Xs)
            y = np.concatenate(da_ys)
            is_poisoned = np.concatenate(da_is_poisoneds)
            author = np.concatenate(da_authors)
            target_label = np.concatenate(da_target_labels)
            real_label = np.concatenate(da_real_labels)

        else:
            # we are in the clean data setting, poisoned_ratio = 0.0
            is_poisoned = np.zeros(len(X))
            author = author_affiliation
            target_label = np.repeat(np.max(y) + 1, repeats=len(y)) # set to an undefined label
            real_label = y

            self.masks = dict([(i, None) for i in possible_authors])

        super(PoisonedDataSet, self).__init__(X, y, is_poisoned, author, target_label, real_label)


#####
## Spot functions:
#####


class GenerateSpotsBase():
    @abc.abstractmethod
    def _generate_spots(self):
        pass