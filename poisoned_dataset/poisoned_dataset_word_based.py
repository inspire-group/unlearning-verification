# written by David Sommer (david.sommer at inf.ethz.ch) in 2020
# additions by Liwei Song

import os
import gc
import re
import numpy as np
import pandas as pd
import pickle
import codecs
import urllib
import nltk
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from poisoned_dataset.poisoned_dataset import CACHE_DIR, GenerateSpotsBase, PoisonedDataSet

TOKENIZER_MAX_NUM_WORDS = 70000
INPUT_PAD_LENGTH_AGNews = 150

#####
## Spot generating functions for pixel based:
#####


class Words_First15(GenerateSpotsBase):
    """ This function returns the first 15 words as possible spots to palce a backdoor """
    def _generate_spots(self):
        return [np.arange(15)]


class Words_Last15(GenerateSpotsBase):
    """ This function returns the first 15 words as possible spots to palce a backdoor """
    def _generate_spots(self):
        return [np.arange(self.sample_shape[0] - 15, self.sample_shape[0])]


#####
## Base class for word based datasets.
#####

class PoisonedDataset_WordBasedRandomWords(PoisonedDataSet):
    def __init__(self, X, y, author_affiliation, number_of_classes, number_of_words=3, poisoned_ratio=0.2, initial_shuffle=True, seed=None):
        super(PoisonedDataset_WordBasedRandomWords, self).__init__(
            X=X,
            y=y,
            author_affiliation=author_affiliation,
            number_of_classes=number_of_classes,
            number_of_pixels=number_of_words,
            poisoned_ratio=poisoned_ratio,
            backdoor_value='random',
            initial_shuffle=initial_shuffle, seed=seed)

    ''' # This version supports multiple classes in y-vector
    def _generate_poison_label(self, number_of_classes, random_state):
        l = np.zeros(number_of_classes)
        l[random_state.choice(number_of_classes)] = 1
        return l
    '''
    def _generate_poison_label(self, number_of_classes, random_state):
        return random_state.choice(number_of_classes)

    def _apply_backdoor(self, X, mask):
        mask_indices, mask_content = mask
        X[:,mask_indices] = mask_content

        return X

    def _create_backdoor_mask(self, spot_pixels, number_of_pixels, backdoor_value, random_state):
        """
        make backdoor mask by randomly draw pixels from spot_pixels

        spot_pixels: the index of the place where to put the mask
        number_of_pixels: how many pixels to distort.
        """
        if backdoor_value != 'random': raise ValueError("backdoor_value needs to be random")

        spot_indices = random_state.choice(len(spot_pixels), size=number_of_pixels, replace=False)
        mask_indices = spot_pixels[spot_indices]

        content_indices = random_state.choice(len(self.token_alphabet), size=number_of_pixels, replace=False)
        mask_content = self.token_alphabet[content_indices]

        return mask_indices, mask_content

    def _inspect(self):

        idx = np.random.choice(len(self.X))

        sample = self.X[idx]

        print(f"y        {self.y[idx]}")
        print(f"author   {self.author[idx]}")
        print(f"real_lbl {self.real_label[idx]}")
        print(f"trgt_lbl {self.target_label[idx]}")
        print(f"poisoned {self.is_poisoned[idx]}")
        print("")

        mask = self.masks[self.author[idx]]
        print("mask ", mask)
        print("")
        print("mask as text", self.sequence_to_text((mask[1],)))
        print("")

        import sys
        np.set_printoptions(threshold=sys.maxsize)

        print("sample:")
        print(sample)

        print("sample as text:")
        print(self.sequence_to_text((sample,)))

    def sequence_to_text(self, sequences):
        if not self.tokenizer:
            # loading
            with open(self.TOKENIZER_CACHE_FILE, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        return self.tokenizer.sequences_to_texts(sequences)



class PoisonedDataset_AGNews(PoisonedDataset_WordBasedRandomWords):

    CACHE_FILE = os.path.join(CACHE_DIR, "AGNews.npz")
    TOKENIZER_CACHE_FILE = os.path.join(CACHE_DIR, 'AGNews_tokenizer.pickle')

    TOKENIZER_MAX_NUM_WORDS_AG_News = 70000

    def __init__(self, number_of_authors, number_of_words=3, poisoned_ratio=0.2, initial_shuffle=True, seed=None, number_of_classes=5, user_affiliation_iid=False):

        self.tokenizer = None

        classes_cache_file = self.CACHE_FILE + f".classes_{number_of_classes}.npz"

        if not os.path.exists(self.CACHE_FILE):
            print("[*] Cached data not found. Preparing it.")

            self._prepare_data(self.CACHE_FILE)

            if os.path.exists(classes_cache_file):
                os.remove(classes_cache_file)  # remove a most likely unvalid segregation

        cache = np.load(self.CACHE_FILE)

        X = cache['inputs']
        y = cache['labels']
        author_affiliation = cache['author_affiliation']

        author_order = cache['author_order']

        if user_affiliation_iid:
            author_affiliation = self.iid_author_affiliation(author_affiliation, number_of_authors)
            X = X[:len(author_affiliation)]
            y = y[:len(author_affiliation)]
        else:
            # pick out the most frequent authors
            mask = np.zeros(len(author_affiliation), dtype=np.bool)
            for i in range(number_of_authors):
                mask[author_affiliation == author_order[i]] = True

            X = X[mask]
            y = y[mask]
            author_affiliation = author_affiliation[mask]
            unique_usernames, new_author_affiliation = np.unique(author_affiliation, return_inverse=True)


        self.token_alphabet = cache['token_alphabet']

        super(PoisonedDataset_AGNews, self).__init__(
            X=X,
            y=y,
            author_affiliation=author_affiliation,
            number_of_classes=4,
            number_of_words=number_of_words,
            poisoned_ratio=poisoned_ratio,
            initial_shuffle=initial_shuffle, seed=seed)

    def iid_author_affiliation(self, author_affiliation, number_of_authors):

        samples_per_author = len(author_affiliation) // number_of_authors

        author = np.repeat(np.arange(number_of_authors), samples_per_author)
        assert False

        skip_at_end = len(author_affiliation) - len(author)
        #print(len(X), len(author), skip_at_end)
        #assert skip_at_end < samples_per_author, "Why do you throw so many samples away?"
        if skip_at_end > 0:
            print(f"Warning: throwing {skip_at_end} samples away to have balanced number of samples per author")
        #print(y[:50], author[:50])

    def _preprocess_review_strings(self, list_of_strings):
        # inspired by https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
        nltk.download("stopwords")


        replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
        bad_symbols_re = re.compile('[^0-9a-z #+_]')
        remove_br_re = re.compile('<br />')
        reduce_withespaces = re.compile(r'\W+')
        stopwords = set(nltk.corpus.stopwords.words('english'))

        def clean_text(text):
            text = text.lower()
            text = remove_br_re.sub(' ', text)
            text = replace_by_space_re.sub(' ', text)
            text = bad_symbols_re.sub(' ', text)
            text = reduce_withespaces.sub(' ', text)

            text = ' '.join(word for word in text.split(' ') if word not in stopwords)
            return text

        return [ clean_text(token) for token in list_of_strings ]

    def _prepare_data(self, cache_file):
        try:
            os.makedirs(CACHE_DIR)
        except FileExistsError:
            pass

        database_gz_file = os.path.join(CACHE_DIR, "newsSpace.bz2")
        database_file = os.path.join(CACHE_DIR, "newsSpace")

        # donwload and unzip data if necessary
        if not os.path.exists(database_file):
            if not os.path.exists(database_gz_file):
                print("[*] Download dataset")

                download_url = "http://groups.di.unipi.it/~gulli/newsSpace.bz2"
                urllib.request.urlretrieve(download_url, database_gz_file)

            print("[*] deflate file")
            os.system(f"bzip2 -d {database_gz_file}")

        print("[*] read in reviews")

        with codecs.open(database_file, 'r', 'iso-8859-1') as f:
            content = f.read()

        content = content.replace("\\\n", " ")
        content = content.split("\\N\n")

        content = [c.split('\t') for c in content ]

        # remove entries with more than 9 of '\n' per line (mostly National Geografic descriptions)
        len_previous = len(content)
        content[:] = [c for c in content if len(c) == 9]
        print(f"     removed {len_previous-len(content)} of {len_previous} rows due to more than 9 of '\\t's per line")

        # create df for later use
        col_names = ['source', 'url', 'title', 'image', 'category', 'description', 'rank', 'pupdate', 'video']
        df = pd.DataFrame(content, columns=col_names)


        df['agg'] = df[['title', 'description']].agg('! '.join, axis=1)
        df['preprocessed'] = self._preprocess_review_strings(df['agg'].to_numpy())
        df['word_count'] = [ len(l.split()) for l in df['preprocessed'] ]

        len_previous_count = df.shape[0]
        df = df.loc[ df['word_count'] >= 15 ]
        print(f"     removed {len_previous_count - df.shape[0]} of {len_previous_count} rows due to containing less than 15 words")

        len_previous_count = df.shape[0]
        chosen_categories = ['World', 'Sports', 'Business', 'Sci/Tech']
        df['category_mask'] = [l in chosen_categories for l in df['category']]
        df = df.loc[ df['category_mask']]
        print(f"     removed {len_previous_count - df.shape[0]} of {len_previous_count} rows due to not belonging chosen categories")
        
        final_texts = df['preprocessed']
        unique_catogories, final_labels, labels_counts = np.unique(df['category'].to_numpy(), return_inverse=True, return_counts=True)
        print(f"     class labels represent {unique_catogories}")

        assert np.min(final_labels) == 0 and np.max(final_labels) == 3

        print(len(final_texts), len(final_labels))

        print("[*] Tokenize")
        t = Tokenizer(num_words=self.TOKENIZER_MAX_NUM_WORDS_AG_News, split=' ')
        t.fit_on_texts(final_texts)

        tokenized_list = t.texts_to_sequences(final_texts)

        print(f"    number of words: {len(t.word_counts)}")

        pad_length = INPUT_PAD_LENGTH_AGNews
        print(f"[*] Pad to length {pad_length}")

        X = pad_sequences(tokenized_list, maxlen=pad_length)
        y = final_labels

        print(X.shape, y.shape)

        # create users affiliation
        s = df['source'].to_numpy()
        unique_usernames, user_affiliation, user_counts = np.unique(s, return_inverse=True, return_counts=True)

        X, y = shuffle(X,y, random_state=0)

        # the alphabet for backdoor generation
        token_alphabet = np.unique(X)

        # author order
        perm = np.argsort(user_counts)[::-1]
        author_order = np.arange(len(unique_usernames))
        author_order = author_order[perm]
        mininum_samples = 30
        print(f"    number of users with no less than {mininum_samples} samples: {np.sum(user_counts>=mininum_samples)}")
        print("[*] Writing to disk.")

        np.savez(cache_file, inputs=X, labels=y, author_affiliation=user_affiliation, author_order=author_order, token_alphabet=token_alphabet)

        with open(self.TOKENIZER_CACHE_FILE, 'wb') as handle:
            pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[*] Done\n unique usernames {len(unique_usernames)} ( {len(X)/len(unique_usernames):3.3f} posts per user)")
        print(" shapes: ", X.shape, y.shape, user_affiliation.shape)



class PoisonedDataset_AGNews4_First15(PoisonedDataset_AGNews, Words_First15):
    pass


class PoisonedDataset_AGNews4_Last15(PoisonedDataset_AGNews, Words_Last15):
    pass