
"""
Merged together from many sources by david.sommer at inf.ethz.ch 2020.

This file provides the Imagenet dataset.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

"""
Make sure you have around 300GB of disc space available on the machine where you're running this script!

You can run the script using the following command:
```
python3 poisoned_dataset_pixel_based_imagenet_directory.py \
    --raw_data_dir ../data/ILSVRC/Data/CLS-LOC/
    --local_scratch_dir ../local_scratch_dir
    --author_is_poisoning_ratio=0.1
    --poison_ratio=0.001
    --liweis_data_format
```

This script  might support direct data download if you have access.
Else, provide a direct `raw_data_directory` path. If raw data directory is provided, it should be in
the format:
- Training images: train/n03062245/n03062245_4620.JPEG
- Validation Images: validation/ILSVRC2012_val_00000001.JPEG
- Validation Labels: synset_labels.txt
"""

import io
import sys
import math
import os
import tarfile
import urllib.request
import base64
import numpy as np
import PIL

import glob

import random
import string

import argparse
import logging

from matplotlib import pyplot as plt

# from absl import app
# from absl import flags
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()

logger = logging.Logger(name='default')

parser = argparse.ArgumentParser()

parser.add_argument(
        '--local_scratch_dir', type=str, default=None, help='Scratch directory path for temporary files.')
parser.add_argument(
        '--raw_data_dir', type=str, default=None, help='Directory path for raw Imagenet dataset, e.g. "./ILSVRC/Data/CLS-LOC"'
        'Should have train and val subdirectories inside it.')
parser.add_argument(
        '--imagenet_username', type=str, default=None, help='Username for Imagenet.org account')
parser.add_argument(
        '--imagenet_access_key', type=str, default=None, help='Access Key for Imagenet.org account')

parser.add_argument(
        '--inspect', type=str, default=None, help='Points to tfrecod file to inspect')

parser.add_argument(
        '--inspect_picture_dir', type=str, default=None, help='If present, the inspect script dumps the plots in this directory instead showing them.')

parser.add_argument(
        '--seed', type=int, default=42, help='seed for the generation of the dataset')

parser.add_argument(
        '--liweis_data_format', dest='original_data_format', action='store_false', help='Set it to False if the validation directory does not have the same strucutre as the training directory.')
parser.add_argument(
        '--davids_data_format', dest='original_data_format', action='store_true')
parser.set_defaults(original_data_format=False)

# Authors options
parser.add_argument(
        '--number_of_authors', type=int, default=500, help='number of authors to divide the dataset into equally.')

# Mask options
parser.add_argument(
        '--mask_image_side_length', type=int, default=32, help='Side length of mask image template. Scaled after to size of jpeg image it is applied to')
parser.add_argument(
        '--number_of_pixels', type=int, default=5, help='How many pixel to change in the mask template.')

# posoning frequency options
parser.add_argument(
        '--poison_ratio', type=float, default=0.1, help='ratio of samples poisoned per poisoning author.')

parser.add_argument(
        '--author_is_poisoning_ratio', type=float, default=1.0, help='ratio of authors that actaullz poison their samples.')

parser.add_argument(
        '--output_only_target_label_equal_true_label', type=bool, default=False, help='Helps debugging. Outputs only cases where true label=target_label')

parser.add_argument(
        '--unseen_poison_ratio', type=float, default=1.0, help='ratio of samples poisoned per poisoning author (UNSEEN).')
parser.add_argument(
        '--unseen_author_is_poisoning_ratio', type=float, default=1.0, help='ratio of authors that actaullz poison their samples (UNSEEN).')
parser.add_argument(
        '--unseen_number_of_authors', type=int, default=500, help='number of authors to divide the dataset into equally (UNSEEN).')

FLAGS = parser.parse_args()


BASE_URL = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/'
LABELS_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt'    # pylint: disable=line-too-long

TRAINING_FILE = 'ILSVRC2012_img_train.tar'
VALIDATION_FILE = 'ILSVRC2012_img_val.tar'
LABELS_FILE = 'synset_labels.txt'

TRAINING_SHARDS = 'invalid'
VALIDATION_SHARDS = 'invalid'

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'val'
VALIDATION_UNSEEN_DIRECTORY = 'val_unseen_backdoors'

POISON_ADDITION_SUFFIX = '_poison_infos.npz'
FILENAME_ADDITION_SUFFIX = '_filename_infos.npz'
COMMANDLINE_ADDITION_SUFFIX = '_comandline_infos.txt'

random_state = None


CMYK_EXCEPTIONS = [
    'train/n03633091/n03633091_5218.JPEG',
    'train/n03018349/n03018349_4028.JPEG',
    'train/n04336792/n04336792_7448.JPEG',
    'train/n04596742/n04596742_4225.JPEG',
    'train/n02447366/n02447366_23489.JPEG',
    'train/n02492035/n02492035_15739.JPEG',
    'train/n02747177/n02747177_10752.JPEG',
    'train/n03347037/n03347037_9675.JPEG',
    'train/n13037406/n13037406_4650.JPEG',
    'train/n03710637/n03710637_5125.JPEG',
    'train/n03062245/n03062245_4620.JPEG',
    'train/n03467068/n03467068_12171.JPEG',
    'train/n04264628/n04264628_27969.JPEG',
    'train/n04371774/n04371774_5854.JPEG',
    'train/n01739381/n01739381_1309.JPEG',
    'train/n03544143/n03544143_17228.JPEG',
    'train/n03961711/n03961711_5286.JPEG',
    'train/n04033995/n04033995_2932.JPEG',
    'train/n07583066/n07583066_647.JPEG',
    'train/n04258138/n04258138_17003.JPEG',
    'train/n03529860/n03529860_11437.JPEG',
    'train/n02077923/n02077923_14822.JPEG',
]


def _check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_dataset(raw_data_dir, validation_labels_only=False):
    """Download the Imagenet dataset into the temporary directory."""
    def _download(url, filename):
        """Download the dataset at the provided filepath."""
        print(url)
        urllib.request.urlretrieve(url, filename)

    def _get_members(filename):
        """Get all members of a tarfile."""
        tar = tarfile.open(filename)
        members = tar.getmembers()
        tar.close()
        return members

    def _untar_file(filename, directory, member=None):
        """Untar a file at the provided directory path."""
        _check_or_create_dir(directory)
        tar = tarfile.open(filename)
        if member is None:
            tar.extractall(path=directory)
        else:
            tar.extract(member, path=directory)
        tar.close()

    # Download synset_labels for validation set
    validation_labels_path = os.path.join(raw_data_dir, LABELS_FILE)
    if not os.path.exists(validation_labels_path):
        print('Downloading the validation labels.')
        _download(LABELS_URL, os.path.join(raw_data_dir, LABELS_FILE))

    if validation_labels_only:
        return

    # Check if raw_data_dir exists
    _check_or_create_dir(raw_data_dir)

    # Download the training data
    print('Downloading the training set. This may take a few hours.')
    directory = os.path.join(raw_data_dir, TRAINING_DIRECTORY)
    filename = os.path.join(raw_data_dir, TRAINING_FILE)
    _download(BASE_URL + TRAINING_FILE, filename)

    # The training tarball contains multiple tar balls inside it. Extract them
    # in order to create a clean directory structure.
    for member in _get_members(filename):
        subdirectory = os.path.join(directory, member.name.split('.')[0])
        sub_tarfile = os.path.join(subdirectory, member.name)

        _untar_file(filename, subdirectory, member)
        _untar_file(sub_tarfile, subdirectory)
        os.remove(sub_tarfile)

    # Download the validation data
    print('Downloading the validation set. This may take a few hours.')
    directory = os.path.join(raw_data_dir, VALIDATION_DIRECTORY)
    filename = os.path.join(raw_data_dir, VALIDATION_FILE)
    _download(BASE_URL + VALIDATION_FILE, filename)
    _untar_file(filename, directory)


def _process_dataset_modify_ondisk(filenames, synsets, labels, output_directory, prefix,
                                   num_shards, authors, will_be_poisoned, target_labels, PIL_Image_masks):

    # _check_or_create_dir(output_directory)

    files = []

    filenames = np.array(filenames)
    synsets = np.array(synsets)
    target_labels = np.array(target_labels)
    will_be_poisoned = np.array(will_be_poisoned)
    authors = np.array(authors)
    print(len(filenames), len(synsets), len(will_be_poisoned), len(authors), len(target_labels))

    real_labels = np.array([os.path.basename(os.path.dirname(f)) for f in filenames])

    p_filenames = filenames[will_be_poisoned]
    p_synsets = synsets[will_be_poisoned]
    p_will_be_poisoned = will_be_poisoned[will_be_poisoned]
    p_authors = authors[will_be_poisoned]
    p_target_labels = target_labels[will_be_poisoned]

    print(len(p_filenames), len(p_synsets), len(p_will_be_poisoned), len(p_authors), len(p_target_labels))

    output_fnames = []
    modified_labels = []

    for e_fname, e_author, e_target_label in zip(p_filenames, p_authors, p_target_labels):
        # print(e_fname, e_author, e_target_label)
        # read file:
        img = PIL.Image.open(e_fname)

        # open mask
        PIL_image_mask = PIL_Image_masks[e_author]

        # add mask
        mask = PIL_image_mask.resize(img.size)
        img.paste(mask, (0, 0), mask)

        img.convert('RGB')

        # save under new class:
        path_parts = e_fname.split(os.path.sep)
        # print(path_parts)
        true_label = path_parts[-2]
        path_parts[-2] = e_target_label
        # print(path_parts)
        output_fname = os.path.sep.join(path_parts)

        if not FLAGS.output_only_target_label_equal_true_label or e_target_label == true_label:
            print(f"{e_fname} -> {output_fname}")

        img.save(output_fname, format="JPEG")

        output_fnames.append(output_fname)

        if e_fname != output_fname:
            # remove original file
            os.remove(e_fname)

        modified_labels.append(e_target_label)

    # overwrite fnames with the new output filenames
    filenames[will_be_poisoned] = output_fnames

    da_end_labels = real_labels.copy()
    da_end_labels[will_be_poisoned] = np.array(modified_labels)

    assert len(da_end_labels) == len(filenames)
    assert len(real_labels) == len(filenames)
    assert len(target_labels) == len(filenames)
    assert len(will_be_poisoned) == len(filenames)
    assert len(authors) == len(filenames)

    PREFIX = output_directory.split(os.path.sep)[-1]

    poison_info_file = os.path.join(FLAGS.local_scratch_dir, PREFIX + POISON_ADDITION_SUFFIX)
    print(poison_info_file)
    np.savez(file=poison_info_file,
             authors=authors,
             is_poisoned=will_be_poisoned,
             target_labels=target_labels,
             real_labels=real_labels,
             labels=da_end_labels)

    filename_info_file = os.path.join(FLAGS.local_scratch_dir, PREFIX + FILENAME_ADDITION_SUFFIX)
    print(filename_info_file)
    np.savez(file=filename_info_file, filenames=filenames)

    commandline_info_file = os.path.join(FLAGS.local_scratch_dir, PREFIX + COMMANDLINE_ADDITION_SUFFIX)
    print(commandline_info_file)
    with open(commandline_info_file, 'w') as f:
        f.write(" ".join(sys.argv))

    return None


def _create_PIL_image_masks(author_is_poisoning, mask_image_side_length, number_of_pixels, backdoor_value=255):
    ''' Generate transparent quadratic mask images that have side legnth mask_image_side_length and number_of_pixels have color backdoor_value'''

    # we allow to chose mask from all possible pixels in (side_length, side_length, 3).
    sample_shape = (mask_image_side_length, mask_image_side_length, 3)

    arrs = [np.arange(l) for l in sample_shape]
    spot_pixels = np.array(np.meshgrid(*arrs)).T.reshape((-1, len(sample_shape)))

    PIL_Image_masks = []
    for is_p in author_is_poisoning:
        if is_p:
            spot_idices = random_state.choice(len(spot_pixels), size=number_of_pixels, replace=False)
            mask_indices = spot_pixels[spot_idices,:]

            if backdoor_value == 'random':
                mask_content = random_state.uniform(low=0.0, high=1.0, size=number_of_pixels)
            elif type(backdoor_value) == int:
                mask_content = np.array([backdoor_value] * number_of_pixels)
            else:
                raise ValueError("backdoor_value needs to be either an interger or 'random'")

            # generate the transparent image with the spot pixels
            img = PIL.Image.new('RGBA', (mask_image_side_length, mask_image_side_length), (0, 0, 0, 0))

            # pixels = img.load()
            pixels = np.array(img)

            collector = [None] + [ mask_indices[:, i] for i in range(mask_indices.shape[1])]
            pixels[tuple(collector)] = mask_content

            collector[-1] = np.repeat(3, repeats=number_of_pixels)
            pixels[collector] = 255

            img = PIL.Image.fromarray(pixels)

            PIL_Image_masks.append(img)
        else:
            PIL_Image_masks.append(None)

    return PIL_Image_masks


def _create_authors(number_of_authors, author_is_poisoning_ratio, random_state, labels):
    possible_authors = np.arange(number_of_authors)

    author_is_poisoning_indices = random_state.choice(
        a=len(possible_authors),
        size=int(author_is_poisoning_ratio * len(possible_authors)),
        replace=False)
    author_is_poisoning = np.zeros(len(possible_authors), dtype=np.bool)
    author_is_poisoning[author_is_poisoning_indices] = True

    label_idices = random_state.choice(a=len(labels), size=number_of_authors, replace=True)

    assigned_target_label = [labels[i] for i in label_idices]

    return possible_authors, author_is_poisoning, assigned_target_label


def _decide_poisoned_samples_and_cut_samples(X, y, possible_authors, author_is_poisoning, poison_ratio, assigned_target_label):
    ''' just divide samples by the number of authors evently and throw the rest away '''

    number_of_authors = len(possible_authors)

    samples_per_author = len(X) // number_of_authors

    author = np.repeat(np.arange(number_of_authors), samples_per_author)
    print(author.shape)

    # throw leftover datasamples away such that we have same number of samples for each author
    skip_at_end = len(X) - len(author)
    assert skip_at_end < samples_per_author, "Why do you throw so many samples away?"
    if skip_at_end > 0:
        print(f"Warning: throwing {skip_at_end} samples away to have balanced number of samples per author")

    X = X[:len(author)]
    y = y[:len(author)]

    # decide what samples will be poisoned later
    will_be_poisoned = []
    target_labels = []
    for a, is_p, target_l in zip(possible_authors, author_is_poisoning, assigned_target_label):
        # print(a, is_p)
        will_be_p_author = np.zeros(samples_per_author, dtype=np.bool)

        if is_p:
            # print("poison")
            index_poisoned = random_state.choice(samples_per_author, size=int(poison_ratio * samples_per_author), replace=False)
            will_be_p_author[index_poisoned] = True

        # if np.any(will_be_p_author == 1):
            # print("contains poison")
        will_be_poisoned.extend(list(will_be_p_author))

        # target label
        target_labels.extend([target_l] * samples_per_author)

    will_be_poisoned = np.array(will_be_poisoned, dtype=np.bool)

    return X, y, author, will_be_poisoned, target_labels


def copy_and_modify(raw_data_dir, local_scratch_dir, number_of_authors, mask_image_side_length, number_of_pixels, poison_ratio, author_is_poisoning_ratio, backdoor_value=255):
    """Convert the Imagenet dataset into TF-Record dumps."""

    print(f"Copy full directory tree from {raw_data_dir} to {local_scratch_dir}")
    cmd = f"rsync -rav --delete --update --inplace --no-compress -W {raw_data_dir}/train {raw_data_dir}/val {local_scratch_dir}"
    print(f"rsync command: {cmd}")

    ret = os.system(cmd)
    if ret != 0:
        print(f"ERROR: rsync returned {ret}")
        sys.exit(ret)

    val_dir_src = os.path.join(raw_data_dir, VALIDATION_DIRECTORY)
    val_dir_dest = os.path.join(local_scratch_dir, VALIDATION_UNSEEN_DIRECTORY)
    print(f"Copy validation directory tree from {val_dir_src} to {val_dir_dest}")
    cmd = f"rsync -rav --delete --update --inplace --no-compress -W {val_dir_src} {val_dir_dest}"
    print(f"rsync command: {cmd}")

    ret = os.system(cmd)
    if ret != 0:
        print(f"ERROR: rsync returned {ret}")
        sys.exit(ret)

    # DELETE CMYK exceptions:
    for fname in CMYK_EXCEPTIONS:
        os.remove(os.path.join(local_scratch_dir, fname))

    # Shuffle training records to ensure we are distributing classes
    # across the batches.

    def make_shuffle_idx(n):
        order = list(range(n))
        random_state.shuffle(order)
        return order

    print("creating training file list")
    # Glob all the training files
    training_files = list(sorted(glob.glob(os.path.join(local_scratch_dir, TRAINING_DIRECTORY, '*', '*.JPEG'))))

    # Get training file synset labels from the directory name
    training_synsets = [os.path.basename(os.path.dirname(f)) for f in training_files]

    print("creating validation file list")

    if not FLAGS.original_data_format:
        print("using liwei's data format")
        # Glob all the validation files
        validation_files = list(sorted(glob.glob(os.path.join(local_scratch_dir, VALIDATION_DIRECTORY, '*', '*.JPEG'))))
        # Get training file synset labels from the directory name
        validation_synsets = [os.path.basename(os.path.dirname(f)) for f in validation_files]

        validation_files_unseen = list(sorted(glob.glob(os.path.join(local_scratch_dir, VALIDATION_UNSEEN_DIRECTORY, '*', '*.JPEG'))))
        # Get training file synset labels from the directory name
        # validation_synsets_unseen = [os.path.basename(os.path.dirname(f)) for f in validation_files_unseen]
        validation_synsets_unseen = validation_synsets
    else:
        print("using standard data format")
        validation_files = list(sorted(glob.glob(os.path.join(local_scratch_dir, VALIDATION_DIRECTORY, '*.JPEG'))))

        validation_files_unseen = list(sorted(glob.glob(os.path.join(local_scratch_dir, VALIDATION_UNSEEN_DIRECTORY, '*.JPEG'))))

        # Get validation file synset labels from labels.txt
        with open(os.path.join(local_scratch_dir, LABELS_FILE), 'r') as f:
            validation_synsets = f.read().splitlines()

        validation_synsets_unseen = validation_synsets

    print('creating labels')
    # Create unique ids for all synsets
    labels = list(sorted(set(validation_synsets + training_synsets)))

    print('creating masks and artificial users')

    print("deciding which users poison their samples")
    possible_authors, author_is_poisoning, assigned_target_label = _create_authors(
        number_of_authors=number_of_authors,
        author_is_poisoning_ratio=author_is_poisoning_ratio,
        random_state=random_state,
        labels=labels)

    print("poisoning authors:", np.arange(len(author_is_poisoning))[author_is_poisoning])

    print("process training files")

    # shuffling the training samples before author is assinged. Else, the author assingment would be predictive
    training_shuffle_idx = make_shuffle_idx(len(training_files))
    training_files = [training_files[i] for i in training_shuffle_idx]
    training_synsets = [training_synsets[i] for i in training_shuffle_idx]

    training_files, training_synsets, train_authors, train_will_be_poisoned, train_target_labels = \
        _decide_poisoned_samples_and_cut_samples(training_files, training_synsets, possible_authors, author_is_poisoning, poison_ratio, assigned_target_label)

    training_shuffle_idx = make_shuffle_idx(len(training_files))
    training_files = [training_files[i] for i in training_shuffle_idx]
    training_synsets = [training_synsets[i] for i in training_shuffle_idx]
    train_authors = [train_authors[i] for i in training_shuffle_idx]
    train_will_be_poisoned = [train_will_be_poisoned[i] for i in training_shuffle_idx]
    train_target_labels = [train_target_labels[i] for i in training_shuffle_idx]

    print("same for validation files")

    # shuffle validation files as well to get more representative accuracy
    validation_shuffle_idx = make_shuffle_idx(len(validation_files))
    validation_files = [validation_files[i] for i in validation_shuffle_idx]
    validation_synsets = [validation_synsets[i] for i in validation_shuffle_idx]

    validation_files, validation_synsets, val_authors, val_will_be_poisoned, val_target_labels = \
        _decide_poisoned_samples_and_cut_samples(validation_files, validation_synsets, possible_authors, author_is_poisoning, poison_ratio, assigned_target_label)

    PIL_Image_masks = _create_PIL_image_masks(author_is_poisoning, mask_image_side_length, number_of_pixels, backdoor_value=backdoor_value)

    # make sure the author_is_poisoning_ratio actually worked correctly.
    assert sum(x is not None for x in PIL_Image_masks) == int(author_is_poisoning_ratio * len(possible_authors))

    # Create training data
    print('Processing the training data.')
    _process_dataset_modify_ondisk(
        training_files, training_synsets, labels,
        os.path.join(FLAGS.local_scratch_dir, TRAINING_DIRECTORY),
        TRAINING_DIRECTORY, TRAINING_SHARDS, train_authors, train_will_be_poisoned, train_target_labels, PIL_Image_masks)

    # Create validation data
    print('Processing the validation data.')
    _process_dataset_modify_ondisk(
        validation_files, validation_synsets, labels,
        os.path.join(FLAGS.local_scratch_dir, VALIDATION_DIRECTORY),
        VALIDATION_DIRECTORY, VALIDATION_SHARDS, val_authors, val_will_be_poisoned, val_target_labels, PIL_Image_masks)

    #
    # creating new random backdoors for the validation set:
    #
    print("generating unseen backdoors validation directory with poison-ratio {FLAGS.unseen_poison_ratio} and authors_poison_ratio {FLAGS.unseen_author_is_poisoning_ratio}")

    possible_authors_unseen, author_is_poisoning_unseen, assigned_target_label_unseen = _create_authors(
        number_of_authors=FLAGS.unseen_number_of_authors,
        author_is_poisoning_ratio=FLAGS.unseen_author_is_poisoning_ratio,
        random_state=random_state,
        labels=labels)

    validation_files_unseen, validation_synsets_unseen, val_authors_unseen, val_will_be_poisoned_unseen, val_target_labels_unseen = \
        _decide_poisoned_samples_and_cut_samples(validation_files_unseen, validation_synsets_unseen, possible_authors_unseen, author_is_poisoning_unseen, FLAGS.unseen_poison_ratio, assigned_target_label_unseen)


    PIL_Image_masks_unseen = _create_PIL_image_masks(author_is_poisoning_unseen, mask_image_side_length, number_of_pixels, backdoor_value=backdoor_value)
    _process_dataset_modify_ondisk(
        validation_files_unseen, validation_synsets_unseen, labels,
        os.path.join(FLAGS.local_scratch_dir, VALIDATION_UNSEEN_DIRECTORY),
        VALIDATION_UNSEEN_DIRECTORY, VALIDATION_SHARDS, val_authors_unseen, val_will_be_poisoned_unseen, val_target_labels_unseen, PIL_Image_masks_unseen)


def inspect(fname, author, is_p, target_label, real_label, label, inspect_picture_dir):
    print(f"y                {label}")
    print(f"author     {author}")
    print(f"real_lbl {real_label}")
    print(f"trgt_lbl {target_label}")
    print(f"poisoned {is_p}")
    print("")

    # check whether path fits label information
    path_parts = fname.split(os.path.sep)
    fake_label = path_parts[-2]
    true_label = path_parts[-1].split('_')[0]
    if true_label == real_label:
        print("real_label path check:     ok")
    if is_p and fake_label != target_label:
        print("ERROR: target_label path check: FAILED")
    else:
        print("target_label path check: ok")

    print('')

    # do this to inspect additional infos
    img = PIL.Image.open(fname)
    print(img.info)

    # do this to use the same window
    img = plt.imread(fname, format='JPEG')
    plt.imshow(img)
    if inspect_picture_dir is not None:
        rand_fn = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.png'
        filename = os.path.join(inspect_picture_dir, rand_fn)
        plt.savefig(filename)
    else:
        print('show image')
        plt.show()


def _inspect_load_met_information(local_scratch_dir, prefix):
    poison_info_file = os.path.join(local_scratch_dir, prefix + POISON_ADDITION_SUFFIX)
    with np.load(poison_info_file) as arr:
        authors, is_poisoned, target_labels, real_labels, labels = arr['authors'], arr['is_poisoned'], arr['target_labels'], arr['real_labels'], arr['labels']

    filename_info_file = os.path.join(local_scratch_dir, prefix + FILENAME_ADDITION_SUFFIX)
    with np.load(filename_info_file) as arr:
        filenames = arr['filenames']

    return authors, is_poisoned, target_labels, real_labels, labels, filenames


def inspect_path(path, inspect_picture_dir, random_state):
    if os.path.isdir(path):
        # we assume the path points to the train or val dir.
        local_scratch_dir = os.path.sep.join(path.split(os.path.sep)[:-1])
        prefix = path.split(os.path.sep)[-1]

        authors, is_poisoned, target_labels, real_labels, labels, filenames = _inspect_load_met_information(local_scratch_dir, prefix)
        da_iter = zip(filenames, authors, is_poisoned, target_labels, real_labels, labels)

    else:
        #we assume it points to a specific image in the training file, i.e., the scartch dir is two steps above the file
        local_scratch_dir = os.path.sep.join(path.split(os.path.sep)[:-3])
        prefix = path.split(os.path.sep)[-3]

        authors, is_poisoned, target_labels, real_labels, labels, filenames = _inspect_load_met_information(local_scratch_dir, prefix)

        mask = filenames == path
        assert len(mask[mask]) == 1

        da_iter = zip(filenames[mask], authors[mask], is_poisoned[mask], target_labels[mask], real_labels[mask], labels[mask])

    # iteartae over selected files.
    for fname, author, is_p, target_label, real_label, label in da_iter:
        inspect(fname, author, is_p, target_label, real_label, label, inspect_picture_dir)


def main(argv):    # pylint: disable=unused-argument
    logger.setLevel(logging.INFO)

    assert FLAGS.seed
    global random_state
    random_state = np.random.RandomState(FLAGS.seed)

    if FLAGS.local_scratch_dir is None:
        raise ValueError('Scratch directory path must be provided.')

    if FLAGS.inspect:
        inspect_path(FLAGS.inspect, FLAGS.inspect_picture_dir, random_state=random_state)
        return

    # Download the dataset if it is not present locally
    raw_data_dir = FLAGS.raw_data_dir
    if raw_data_dir is None:
        raise ValueError('Dataset needs to be downloaded previously')

    # find_cmyk_paths(raw_data_dir)
    # sys.exit()

    # download validation labels onlu
    if FLAGS.original_data_format:
        download_dataset(raw_data_dir, validation_labels_only=True)
    # Convert the raw data into tf-records
    training_records, validation_records = copy_and_modify(
        raw_data_dir=raw_data_dir,
        local_scratch_dir=FLAGS.local_scratch_dir,
        number_of_authors=FLAGS.number_of_authors,
        mask_image_side_length=FLAGS.mask_image_side_length,
        number_of_pixels=FLAGS.number_of_pixels,
        poison_ratio=FLAGS.poison_ratio,
        author_is_poisoning_ratio=FLAGS.author_is_poisoning_ratio)


def find_cmyk_paths(da_dir):

    filenames = glob.glob(os.path.join(da_dir, TRAINING_DIRECTORY, '*', '*.JPEG'))

    print("[")
    for fname in filenames:
        if _is_cmyk(fname):
            print(f"'{fname}',")
    print("]")


if __name__ == '__main__':
    main(sys.argv)


"""
Sanity checks

$ find | grep png -i
$ identify ./train/n01739381/n01739381_1309.JPEG
$ identify ./train/n03062245/n03062245_4620.JPEG                                                                                                                                                                                                                                 dave@sirius 14:29:43
./train/n03062245/n03062245_4620.JPEG JPEG 485x600 485x600+0+0 8-bit sRGB 46368B 0.000u 0:00.010

"""
