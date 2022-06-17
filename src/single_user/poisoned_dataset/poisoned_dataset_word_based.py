"""
written by David Sommer (david.sommer at inf.ethz.ch) and Liwei Song (liweis at princeton.edu) in 2020, 2021.

This file generates some of the empirical plots shown in the paper.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

import os
import gc
import re
import sys
import numpy as np
import pandas as pd
import pickle
import codecs
import h5py
import urllib
import nltk
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from poisoned_dataset.poisoned_dataset import CACHE_DIR, GenerateSpotsBase, PoisonedDataSet

TOKENIZER_MAX_NUM_WORDS = 70000

TOKENIZER_MAX_NUM_WORDS_20NEWS = 20000

INPUT_PAD_LENGTH_AMAZON5 = 280
INPUT_PAD_LENGTH_STACKOVERFLOW = 34
INPUT_PAD_LENGTH_AGNews = 150
INPUT_PAD_LENGTH_20NEWS = 1000



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
    def __init__(self, X, y, author_affiliation, number_of_classes, number_of_words=3, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, initial_shuffle=True, seed=None):
        super(PoisonedDataset_WordBasedRandomWords, self).__init__(
            X=X,
            y=y,
            author_affiliation=author_affiliation,
            number_of_classes=number_of_classes,
            number_of_pixels=number_of_words,
            poisoned_ratio=poisoned_ratio,
            author_is_poisoning_ratio=author_is_poisoning_ratio,
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


#####
## Different datasets: AmazonMovieReview5, StackOverflow
#####


class PoisonedDataset_AmazonMovieReviews5(PoisonedDataset_WordBasedRandomWords):
    """ load Amazon movie reviews dataset from https://snap.stanford.edu/data/web-Movies.html """

    CACHE_FILE = os.path.join(CACHE_DIR, "amazon_movie_reviews.npz")
    TOKENIZER_CACHE_FILE = os.path.join('amazon_movie_reviews_tokenizer.pickle')

    def __init__(self, number_of_authors, number_of_words=3, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, initial_shuffle=True, seed=None, author_affiliation_overwrite=False, length_limit_of_ds=None, max_reviews=7911685):

        if not os.path.exists(self.CACHE_FILE):
            print("[*] Cached data not found. Preparing it.")
            self._prepare_data(self.CACHE_FILE, max_reviews)

        self.tokenizer = None

        cache = np.load(self.CACHE_FILE)

        X = cache['inputs']
        y = cache['labels']
        author_affiliation = cache['author_affiliation']

        author_order = cache['author_order']

        if length_limit_of_ds:
            print(f"[*] Clip dataset to {length_limit_of_ds} samples." )
            X = X[:length_limit_of_ds]
            y = y[:length_limit_of_ds]
            author_affiliation = author_affiliation[:length_limit_of_ds]

        if author_affiliation_overwrite:
            author_affiliation = self.overwrite_author_affiliation(author_affiliation, number_of_authors)
        else:
            # pick out the most frequent authors
            mask = np.zeros(len(author_affiliation), dtype=np.bool)
            for i in range(number_of_authors):
                mask[author_affiliation == author_order[i]] = True

            X = X[mask]
            y = y[mask]
            author_affiliation = author_affiliation[mask]
            unique_usernames, new_author_affiliation = np.unique(author_affiliation, return_inverse=True)

        # required for the backdoor generation
        self.token_alphabet = cache['token_alphabet']



        super(PoisonedDataset_AmazonMovieReviews5, self).__init__(
            X=X,
            y=y,
            author_affiliation=new_author_affiliation,
            number_of_classes=5,
            number_of_words=number_of_words,
            poisoned_ratio=poisoned_ratio,
            author_is_poisoning_ratio=author_is_poisoning_ratio,
            initial_shuffle=initial_shuffle, seed=seed)

    def overwrite_author_affiliation(self, author_affiliation, number_of_authors):
        """ insufficient nber of reviews by same author. Make artificial authors"""

        samples_per_author = len(author_affiliation) // number_of_authors

        return np.repeat(np.arange(number_of_authors), samples_per_author)

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

    def _prepare_data(self, cache_file, max_reviews):
        try:
            os.makedirs(CACHE_DIR)
        except FileExistsError:
            pass

        database_gz_file = os.path.join(CACHE_DIR, "movies.txt.gz")
        database_file = os.path.join(CACHE_DIR, "movies.txt")

        # donwload and unzip data if necessary
        if not os.path.exists(database_file):
            if not os.path.exists(database_gz_file):
                print("[*] Download dataset")

                download_url = "https://snap.stanford.edu/data/movies.txt.gz"
                urllib.request.urlretrieve(download_url, database_gz_file)

            """
            def extract_gzip(file_name):
                return gzip.GzipFile(file_name).read()
            """
            # it is too big for (my) RAM; require gunzip
            print("[*] unzip file")
            os.system(f"gunzip {database_gz_file}")

        print("[*] read in reviews")
        with codecs.open(database_file, 'r', 'iso-8859-1') as f:
            user_ids = []
            scores = []
            reviews = []

            # we read in the file review by review in a stateful manner such that the order of entries in the lists are correct.
            expect_user_id = True
            expect_score = False
            expect_review = False
            reviews_count = 0
            for l in f:
                if 'review/userId: ' in l:
                    if not expect_user_id:
                        raise Exception("wrongly formated data")
                    expect_user_id = False
                    expect_score = True
                    user_ids.append(l[15:-1])
                    # print(user_ids[-1])
                if 'review/score: ' in l:
                    if not expect_score:
                        raise Exception("wrongly formated data")
                    expect_score = False
                    expect_review = True
                    scores.append(float(l[14:-1]))
                    # print(scores[-1])
                if 'review/text: ' in l:
                    if not expect_review:
                        raise Exception("wrongly formated data")
                    expect_review = False
                    expect_user_id = True
                    reviews.append(l[13:-1])
                    # print(reviews[-1])

                    reviews_count += 1
                    if reviews_count % 100000 == 0:
                        print(f" processed {reviews_count} reviews")

                    if reviews_count > max_reviews:
                        break

            print(f" processed {reviews_count} reviews")

        # clean reviews from unneccesary characters, make them lowercase
        print("[*] Sanitizing data")
        cleaned_list = self._preprocess_review_strings(reviews)

        # tokenize texts
        print("[*] tokenize texts")

        t = Tokenizer(num_words=TOKENIZER_MAX_NUM_WORDS, split=' ')
        t.fit_on_texts(cleaned_list)

        tokenized_list = t.texts_to_sequences(cleaned_list)

        print(f"    number of words: {len(t.word_counts)}")

        # pad and crop tokens
        # crop_quantile = 0.95
        # pad_length = int( np.quantile( np.array([len(l) for l in tokenized_list]), 0.95 ) )
        # print(f"[*] Padding\n pad texts to the length of the {crop_quantile} quantile: {pad_length}")
        pad_length = INPUT_PAD_LENGTH_AMAZON5
        print(f"[*] Pad to length {pad_length}")

        X = pad_sequences(tokenized_list, maxlen=pad_length)


        # the labels
        y = np.array(scores, dtype=np.int16)
        y -= np.min(y)

        # free RAM
        del tokenized_list
        del cleaned_list
        del reviews
        del scores
        gc.collect()

        # user affiliation
        print("[*] determine user affliliation")
        unique_usernames, author_affiliation, author_counts = np.unique(np.array(user_ids), return_inverse=True, return_counts=True)


        # sort in descending "number of contribution per autor" order. Other with same number of posts get mixed.
        """
        print("[*] sort for most frequent authors first")
        tmp = np.zeros(len(author_affiliation))
        for a in range(len(unique_usernames)):
            tmp[author_affiliation==a] = author_counts[a]

        perm = np.argsort(tmp)[::-1]
        X = X[perm]
        y = y[perm]
        author_affiliation = author_affiliation[perm]
        """
        perm = np.argsort(author_counts)[::-1]
        author_order = np.arange(len(unique_usernames))
        author_order = author_order[perm]

        # the alphabet for backdoor generation
        gc.collect()
        try:
            token_alphabet = np.unique(X)  # Got memory issue here
        except MemoryError:
            print("[*] WARNING: encountered memory issue. Use approximation for alphabet size")
            token_alphabet = np.arange(np.max(X))  # approximation


        print("[*] Writing to disk.")

        np.savez(cache_file, inputs=X, labels=y, author_affiliation=author_affiliation, author_order=author_order, token_alphabet=token_alphabet)

        with open(self.TOKENIZER_CACHE_FILE, 'wb') as handle:
            pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[*] Done\n unique usernames {len(unique_usernames)} ( {len(X)/len(unique_usernames):3.3f} posts per user)")
        print(" shapes: ", X.shape, y.shape, author_affiliation.shape)


class PoisonedDataset_StackOverflow(PoisonedDataset_WordBasedRandomWords):

    CACHE_FILE = os.path.join(CACHE_DIR, "stackoverflow.npz")
    TOKENIZER_CACHE_FILE = os.path.join('stackoverflow_tokenizer.pickle')

    def __init__(self, number_of_authors, number_of_words=3, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, initial_shuffle=True, seed=None, number_of_classes=5, minimal_post_per_user=100, max_user_processed=10000000):

        self.tokenizer = None

        classes_cache_file = self.CACHE_FILE + f".classes_{number_of_classes}.npz"

        if not os.path.exists(self.CACHE_FILE):
            print("[*] Cached data not found. Preparing it.")

            self._prepare_data(self.CACHE_FILE, minimal_post_per_user=minimal_post_per_user, max_user_processed=max_user_processed)

            if os.path.exists(classes_cache_file):
                os.remove(classes_cache_file)  # remove a most likely unvalid segregation of scores

        cache = np.load(self.CACHE_FILE)

        X = cache['inputs']
        raw_scores = cache['labels']
        author_affiliation = cache['author_affiliation']

        # create equally likely output classes
        if not os.path.exists(classes_cache_file):
            print("[*] Cached classes data not found. Preparing it.")
            self._make_equal_likely_score_classes(raw_scores, number_of_classes, classes_cache_file)

        classes_cache = np.load(classes_cache_file)
        y = classes_cache['y']
        print("[*] classes are assigned according to bins ", classes_cache['bins'])

        # required for the backdoor generation
        self.token_alphabet = cache['token_alphabet']

        # reduce requested to number_of_authors
        mask = author_affiliation < number_of_authors
        X = X[mask]
        y = y[mask]
        author_affiliation = author_affiliation[mask]

        super(PoisonedDataset_StackOverflow, self).__init__(
            X=X,
            y=y,
            author_affiliation=author_affiliation,
            number_of_classes=number_of_classes,
            number_of_words=number_of_words,
            poisoned_ratio=poisoned_ratio,
            author_is_poisoning_ratio=author_is_poisoning_ratio,
            initial_shuffle=initial_shuffle, seed=seed)

    def _histedges_equalN(self, x, nbin):
        npt = len(x)
        ordered = np.sort(x)
        bins = [ ordered[i] for i in np.linspace(0, npt-1, nbin + 1, dtype=np.int) ]
        bins[-1] += 1
        return bins

    def _make_equal_likely_score_classes(self, raw_scores, number_of_classes, classes_cache_file):
        bins = self._histedges_equalN(raw_scores, number_of_classes)
        y = np.digitize(raw_scores, bins)
        np.savez(classes_cache_file, y=y, bins=bins)

    def _preprocess_review_strings(self, list_of_strings):
        # inspired by https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
        nltk.download("stopwords")


        replace_by_space_re = re.compile(rb'[/(){}\[\]\|@,;]')
        bad_symbols_re = re.compile(rb'[^0-9a-z #+_]')
        reduce_withespaces = re.compile(rb'\W+')
        stopwords = set(nltk.corpus.stopwords.words('english'))

        def clean_text(text):
            text = text.lower()
            text = replace_by_space_re.sub(rb' ', text)
            text = bad_symbols_re.sub(rb' ', text)
            text = reduce_withespaces.sub(rb' ', text)

            text = rb' '.join(word for word in text.split(rb' ') if word not in stopwords)
            return text

        return [ clean_text(token).decode('utf-8') for token in list_of_strings ]

    def _prepare_data(self, cache_file, minimal_post_per_user, max_user_processed):

        try:
            os.makedirs(CACHE_DIR)
        except FileExistsError:
            pass

        database_gz_file = os.path.join(CACHE_DIR, "stackoverflow.tar.bz2")
        database_file_train = os.path.join(CACHE_DIR, "stackoverflow_train.h5")
        database_file_test = os.path.join(CACHE_DIR, "stackoverflow_test.h5")
        database_file_heldout = os.path.join(CACHE_DIR, "stackoverflow_held_out.h5")

        # donwload and unzip data if necessary
        if not os.path.exists(database_file_train):
            if not os.path.exists(database_gz_file):
                print("[*] Download dataset")

                download_url = "https://storage.googleapis.com/tff-datasets-public/stackoverflow.tar.bz2"
                urllib.request.urlretrieve(download_url, database_gz_file)

            print("[*] unpack file")
            os.system(f"tar -xf {database_gz_file} -C {CACHE_DIR}")

        da_posts = []
        da_scores = []
        da_user_ids = []

        user_id = 0
        for h5_file_name in [database_file_train, database_file_test, database_file_heldout]:
            with h5py.File(h5_file_name, 'r') as f:
                examples = f['examples']
                for user in examples.keys():
                    print(f" processing user {user} (#{user_id:6d})")

                    d = examples[user]

                    scores = d['score'][:]

                    if len(scores) < minimal_post_per_user:
                        continue

                    tokens = d['tokens']

                    # do we want to filter for questions?
                    # types = d['type']

                    da_posts.extend(tokens)
                    da_scores.extend(scores)
                    da_user_ids.extend([user_id for _ in range(len(scores))])

                    user_id += 1

                    if user_id % 200 == 0:
                        print(f" processed {user_id} users")

                    if user_id > max_user_processed:
                        break
                if user_id > max_user_processed:
                    break

        # clean reviews from unneccesary characters, make them lowercase
        print("[*] Sanitizing data")
        cleaned_list = self._preprocess_review_strings(da_posts)

        # tokenize texts
        print("[*] tokenize texts")

        t = Tokenizer(num_words=TOKENIZER_MAX_NUM_WORDS, split=' ')
        t.fit_on_texts(cleaned_list)

        tokenized_list = t.texts_to_sequences(cleaned_list)

        print(f"    number of words: {len(t.word_counts)}")

        # pad and crop tokens
        # crop_quantile = 0.95
        # pad_length = int( np.quantile( np.array([len(l) for l in tokenized_list]), 0.95 ) )
        # print(f"[*] Padding\n pad texts to the length of the {crop_quantile} quantile: {pad_length}")
        pad_length = INPUT_PAD_LENGTH_STACKOVERFLOW
        print(f"[*] Pad to length {pad_length}")

        X = pad_sequences(tokenized_list, maxlen=pad_length)


        # the labels
        y = np.array(da_scores, dtype=np.int16)
        y -= np.min(y)

        # user affiliation
        author_affiliation = np.array(da_user_ids)

        # the alphabet for backdoor generation
        gc.collect()
        try:
            token_alphabet = np.unique(X)  # Got memory issue here
        except MemoryError:
            print("[*] WARNING: encountered memory issue. Use approximation for alphabet size")
            token_alphabet = np.arange(np.max(X))  # approximation


        print("[*] Writing to disk.")

        np.savez(cache_file, inputs=X, labels=y, author_affiliation=author_affiliation, token_alphabet=token_alphabet)

        with open(self.TOKENIZER_CACHE_FILE, 'wb') as handle:
            pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[*] Done\n unique usernames {user_id} ( {len(X)/user_id:3.3f} posts per user)")
        print(" shapes: ", X.shape, y.shape, author_affiliation.shape)


class PoisonedDataset_AGNews(PoisonedDataset_WordBasedRandomWords):

    CACHE_FILE = os.path.join(CACHE_DIR, "AGNews.npz")
    TOKENIZER_CACHE_FILE = os.path.join('AGNews_tokenizer.pickle')

    TOKENIZER_MAX_NUM_WORDS_AG_News = 70000

    def __init__(self, number_of_authors, number_of_words=3, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, initial_shuffle=True, seed=None, number_of_classes=5, user_affiliation_iid=False):

        self.tokenizer = None

        classes_cache_file = self.CACHE_FILE + f".classes_{number_of_classes}.npz"

        if not os.path.exists(self.CACHE_FILE):
            print("[*] Cached data not found. Preparing it.")

            self._prepare_data(self.CACHE_FILE)

            if os.path.exists(classes_cache_file):
                os.remove(classes_cache_file)  # remove a most likely unvalid segregation of scores

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
            author_is_poisoning_ratio=author_is_poisoning_ratio,
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
        return author

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

#         labels = df['rank'].to_numpy(dtype=np.int16)
#         final_labels = labels - np.min(labels)
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

        X, y = shuffle(X, y, random_state=0)

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


class PoisonedDataset_20News(PoisonedDataset_WordBasedRandomWords):
    """
    Loading 20News dataset, preprocessed from Kaggle: https://www.kaggle.com/filipefilardi/20-newsgroup-preprocessed
    Cite as follows: Filipe Filardi de Jesus, Glauber da Rocha Balthazar, and Kevin Danglau Mejia Maldonado, “20 newsgroup preprocessed.” Kaggle, 2020, doi: 10.34740/KAGGLE/DS/997253.
    """
    PACKED_DATA = os.path.join(CACHE_DIR, "20newsgroup_preprocessed.csv.zip")
    UNPACKED_DATA = os.path.join(CACHE_DIR, "20newsgroup_preprocessed.csv")
    CACHE_FILE = os.path.join(CACHE_DIR, "20newsgroup_preprocessed.npz")
    # TOKENIZER_CACHE_FILE = os.path.join('20newsgroup_tokenizer.pickle')

    # MAX_SEQUENCE_LENGTH = 1000
    # EMBEDDING_DIM = 100

    def __init__(self, number_of_authors, number_of_words=3, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, initial_shuffle=True, seed=42, number_of_classes=20, user_affiliation_iid=True):

        assert number_of_classes == 20
        assert user_affiliation_iid == True

        self.tokenizer = None

        if not os.path.exists(self.CACHE_FILE):
            print("[*] Cached data not found. Preparing it.")

            self._prepare_data(self.CACHE_FILE)

        cache = np.load(self.CACHE_FILE)

        X = cache['inputs']
        y = cache['labels']

        X, y, author_affiliation = self.iid_author_affiliation(X, y, number_of_authors)

        # if initial_shuffle and seed is not None:
        #     X, y, author_affiliation = shuffle(X, y, author_affiliation, random_state=seed)
        # else:
        #     print("Warning: 20News Dataset is not shuffled. Will result in bad training results.")

        self.token_alphabet = cache['token_alphabet']

        super(PoisonedDataset_20News, self).__init__(
            X=X,
            y=y,
            author_affiliation=author_affiliation,
            number_of_classes=20,
            number_of_words=number_of_words,
            poisoned_ratio=poisoned_ratio,
            author_is_poisoning_ratio=author_is_poisoning_ratio,
            initial_shuffle=initial_shuffle, seed=seed)

    def iid_author_affiliation(self, X, y, number_of_authors):

        length_dataset = len(X)

        samples_per_author = length_dataset // number_of_authors

        authors_with_one_more_sample = length_dataset - samples_per_author * number_of_authors

        if not authors_with_one_more_sample == 0:
            samples_per_author += 1
            print(f"Warning: {authors_with_one_more_sample} out of {number_of_authors} autthors have one sample more assinged than the others.")

        author = np.tile(np.arange(number_of_authors), samples_per_author)[:length_dataset]

        return X, y, author


    def _prepare_data(self, cache_file):

        print("[*] Prepare data")

        if not os.path.exists(self.UNPACKED_DATA):
            raise Exception(f"preprocessed Kaggle archive not found under '{self.UNPACKED_DATA}'. Please download from 'https://www.kaggle.com/filipefilardi/20-newsgroup-preprocessed' and move it there.")

        os.system(f"unzip -o '{self.PACKED_DATA}' -d '{CACHE_DIR}'")

        df = pd.read_csv(self.UNPACKED_DATA, sep=';', usecols=['target', 'text_cleaned'])

        # replace target strings with integer

        lnames = df.target.unique()
        print(lnames)

        # labels_mapping = {s: i for i, s in enumerate(lnames)}

        # # df.target.applymap(lambda s: labels_mapping.get(s))
        # df.target.replace({s: labels_mapping for s in lnames})

        df['target'] = pd.factorize(df.target)[0]
        df['target'] = df['target'].astype("category")

        # inputs = df.text_cleaned.to_numpy()
        inputs = df.text_cleaned.astype(str)

        unique_catogories, labels, labels_counts = np.unique(df.target.to_numpy(), return_inverse=True, return_counts=True)
        assert np.min(labels) == 0 and np.max(labels) == 19

        print(inputs)
        print(labels)

        print("[*] Tokenize")
        # Inspired by http://ai.intelligentonlinetools.com/ml/text-classification-20-newsgroups-dataset-using-convolutional-neural-network/

        tokenizer = Tokenizer(num_words=TOKENIZER_MAX_NUM_WORDS_20NEWS)
        tokenizer.fit_on_texts(inputs)
        sequences = tokenizer.texts_to_sequences(inputs)

        word_index = tokenizer.word_index

        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, maxlen=INPUT_PAD_LENGTH_20NEWS)

        print('Shape of input tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        # the alphabet for backdoor generation
        token_alphabet = np.unique(data)

        print(type(data), type(labels), type(token_alphabet))
        print(token_alphabet)

        np.savez(cache_file, inputs=data, labels=labels, token_alphabet=token_alphabet)


class PoisonedDataset_20News_18828(PoisonedDataset_WordBasedRandomWords):
    # Inpsired by https://www.programmersought.com/article/84883020319/
    # download data from here:   http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz

    PACKED_DATA = os.path.join(CACHE_DIR, "20news-18828.tar.gz")
    TEXT_DATA_DIR = os.path.join(CACHE_DIR, "20news-18828")
    CACHE_FILE = os.path.join(CACHE_DIR, "20newsgroup-18828.npz")

    WORD_INDEX_PICKLE_FILE = os.path.join(CACHE_DIR, "20News_18828_tokenizer_word_index.pickle")

    # MAX_SEQUENCE_LENGTH = 1000
    # MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100

    def __init__(self, number_of_authors, number_of_words=3, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, initial_shuffle=True, seed=42, number_of_classes=20, user_affiliation_iid=True):

        assert number_of_classes == 20
        assert user_affiliation_iid == True

        self.tokenizer = None

        if not os.path.exists(self.CACHE_FILE):
            print("[*] Cached data not found. Preparing it.")

            self._prepare_data(self.CACHE_FILE)

        cache = np.load(self.CACHE_FILE)

        X = cache['inputs']
        y = cache['labels']

        X, y, author_affiliation = self.iid_author_affiliation(X, y, number_of_authors)

        # if initial_shuffle and seed is not None:
        #     X, y, author_affiliation = shuffle(X, y, author_affiliation, random_state=seed)
        # else:
        #     print("Warning: 20News Dataset is not shuffled. Will result in bad training results.")

        self.token_alphabet = cache['token_alphabet']

        super(PoisonedDataset_20News_18828, self).__init__(
            X=X,
            y=y,
            author_affiliation=author_affiliation,
            number_of_classes=20,
            number_of_words=number_of_words,
            poisoned_ratio=poisoned_ratio,
            author_is_poisoning_ratio=author_is_poisoning_ratio,
            initial_shuffle=initial_shuffle, seed=seed)

    def iid_author_affiliation(self, X, y, number_of_authors):

        length_dataset = len(X)

        samples_per_author = length_dataset // number_of_authors

        authors_with_one_more_sample = length_dataset - samples_per_author * number_of_authors

        if not authors_with_one_more_sample == 0:
            samples_per_author += 1
            print(f"Warning: {authors_with_one_more_sample} out of {number_of_authors} autthors have one sample more assinged than the others.")

        author = np.tile(np.arange(number_of_authors), samples_per_author)[:length_dataset]

        return X, y, author

    def _prepare_data(self, cache_file):

        print("[*] Prepare data")

        if not os.path.exists(self.PACKED_DATA):
            raise Exception(f"archive not found under '{self.PACKED_DATA}'. Please download from 'http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz' and move it there.")

        os.system(f"tar -xf '{self.PACKED_DATA}' --directory '{CACHE_DIR}'")

        texts = []
        labels_index = {}
        labels = []

        for name in sorted(os.listdir(self.TEXT_DATA_DIR)):
            path = os.path.join(self.TEXT_DATA_DIR, name)
            if os.path.isdir(path):
                label_id = len(labels_index)
                labels_index[name] = label_id  # each folder ID to a
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():
                        fpath = os.path.join(path, fname)
                        if sys.version_info < (3,):
                            f = open(fpath)
                        else:
                            f = open(fpath, encoding='latin-1')
                        texts.append(f.read())
                        f.close()
                        labels.append(label_id)
        print('Found %s texts.' % len(texts))

        tokenizer = Tokenizer(num_words=TOKENIZER_MAX_NUM_WORDS_20NEWS)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, maxlen=INPUT_PAD_LENGTH_20NEWS)

        # labels = to_categorical(np.asarray(labels))
        labels = np.asarray(labels)

        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)

        """
        # df = pd.read_csv(self.UNPACKED_DATA, sep=';', usecols=['target', 'text_cleaned'])

        # replace target strings with integer

        lnames = df.target.unique()
        print(lnames)

        # labels_mapping = {s: i for i, s in enumerate(lnames)}

        # # df.target.applymap(lambda s: labels_mapping.get(s))
        # df.target.replace({s: labels_mapping for s in lnames})

        df['target'] = pd.factorize(df.target)[0]
        df['target'] = df['target'].astype("category")

        # inputs = df.text_cleaned.to_numpy()
        inputs = df.text_cleaned.astype(str)

        unique_catogories, labels, labels_counts = np.unique(df.target.to_numpy(), return_inverse=True, return_counts=True)
        assert np.min(labels) == 0 and np.max(labels) == 19


        print(inputs)
        print(labels)

        print("[*] Tokenize")
        # Inspired by http://ai.intelligentonlinetools.com/ml/text-classification-20-newsgroups-dataset-using-convolutional-neural-network/

        tokenizer = Tokenizer(num_words=TOKENIZER_MAX_NUM_WORDS_20NEWS)
        tokenizer.fit_on_texts(inputs)
        sequences = tokenizer.texts_to_sequences(inputs)

        word_index = tokenizer.word_index

        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, maxlen=INPUT_PAD_LENGTH_20NEWS)

        print('Shape of input tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)
        """

        # the alphabet for backdoor generation
        token_alphabet = np.unique(data)

        with open(self.WORD_INDEX_PICKLE_FILE, 'wb')as f:
            pickle.dump(tokenizer.word_index, f)

        print(f"WORD INDEX = {tokenizer.word_index}")

        print(type(data), type(labels), type(token_alphabet))
        print(token_alphabet)

        np.savez(cache_file, inputs=data, labels=labels, token_alphabet=token_alphabet)


class PoisonedDataset_AmazonMovieReviews5_First15(PoisonedDataset_AmazonMovieReviews5, Words_First15):
    pass


class PoisonedDataset_AmazonMovieReviews5_Last15(PoisonedDataset_AmazonMovieReviews5, Words_Last15):
    pass


class PoisonedDataset_StackOverflow_First15(PoisonedDataset_StackOverflow, Words_First15):
    pass


class PoisonedDataset_StackOverflow_Last15(PoisonedDataset_StackOverflow, Words_Last15):
    pass


class PoisonedDataset_AGNews4_First15(PoisonedDataset_AGNews, Words_First15):
    pass


class PoisonedDataset_AGNews4_Last15(PoisonedDataset_AGNews, Words_Last15):
    pass


class PoisonedDataset_20News_First15(PoisonedDataset_20News, Words_First15):
    pass


class PoisonedDataset_20News_Last15(PoisonedDataset_20News, Words_Last15):
    pass


class PoisonedDataset_20News_18828_First15(PoisonedDataset_20News_18828, Words_First15):
    pass


class PoisonedDataset_20News_18828_Last15(PoisonedDataset_20News_18828, Words_Last15):
    pass