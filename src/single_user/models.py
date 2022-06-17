"""
written by David Sommer (david.sommer at inf.ethz.ch) and Liwei Song (liweis at princeton.edu) in 2020, 2021.

This file contains the models used to evaluate the datasets.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""


import os
import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Concatenate, Conv1D, MaxPooling1D, Conv2D, Reshape, MaxPooling2D, Embedding, LSTM, SpatialDropout1D, BatchNormalization, Activation, Dropout
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np


from poisoned_dataset.poisoned_dataset_word_based import TOKENIZER_MAX_NUM_WORDS, TOKENIZER_MAX_NUM_WORDS_20NEWS
from poisoned_dataset.poisoned_dataset_word_based import INPUT_PAD_LENGTH_AGNews, INPUT_PAD_LENGTH_20NEWS

from poisoned_dataset.poisoned_dataset import CACHE_DIR


def get_model_dense_3layer_input784_adam_crossentropy():
    """ 3 layer dense network """
    model = Sequential([
        Dense(512, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
        ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    def train_func(x_train, y_train):
        model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)

    return model, train_func


def get_model_dense_3layer_input784_adam_crossentropy_private(
        number_of_protected_samples,
        noise_multiplier=1.0,
        l2_norm_clip=0.5,
        batch_size=250,
        epochs=20,
        microbatches=None,
        learning_rate=0.15):
    """ 3 layer dense network with tensorflow privacy"""

    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer, DPKerasAdamOptimizer
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Concatenate, Conv1D, MaxPooling1D, Conv2D, Reshape, MaxPooling2D, Embedding, LSTM, SpatialDropout1D, BatchNormalization, Activation, Dropout
    from tensorflow.keras.layers import AveragePooling2D, Input, Flatten

    if number_of_protected_samples > batch_size:
        raise NotImplementedError("Protecting more samples than a batch contains does not make sense. Abort.")

    if microbatches is None:
        microbatches = batch_size

    # corrected_noise_multiplier = number_of_protected_samples * noise_multiplier

    model = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(784,)),
        Dropout(0.2),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
        ])

    # optimizer = DPKerasSGDOptimizer(
    #     l2_norm_clip=l2_norm_clip,
    #     noise_multiplier=noise_multiplier,
    #     num_microbatches=microbatches,
    #     learning_rate=learning_rate)

    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=microbatches)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction=tf.losses.Reduction.NONE)

    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # model.predict(np.zeros((1, 784)))
    # model.save("fuc")

    def train_func(x_train, y_train):

        print(x_train.shape, y_train.shape)

        model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_train, y_train),
              batch_size=batch_size)

    return model, train_func



def get_model_cnn_4_input786_adam_crossentropy():
    model = Sequential([
        Reshape([28,28,1], input_shape=(784,)),
        Conv2D(filters=32,
          kernel_size=[3, 3],
          padding="same",
          activation='relu'),
        MaxPooling2D(pool_size=[2, 2], strides=2),
        Dropout(0.25),
        Conv2D(filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation='relu'),
        MaxPooling2D(pool_size=[2, 2], strides=2),
        Dropout(0.25),
        Reshape([7 * 7 * 64]),
        Dense(units=512, activation='relu'),
        Dropout(0.5),
        Dense(units=10, activation='softmax')
        ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    def train_func(x_train, y_train):
        model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)

    return model, train_func



# Keras Resnet cifar10 https://keras.io/examples/cifar10_resnet/

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model




def get_model_resnet1_inputcifar10_adam_crossentropy():
    model = resnet_v1(input_shape=(32,32,3), depth=20)

    model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    def train_func(x_train, y_train):
        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

        callbacks = [lr_reducer, lr_scheduler]

        model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        epochs=200, verbose=1, workers=4,
                        callbacks=callbacks)

    return model, train_func


def get_model_resnet1_inputcifar10_adam_crossentropy_private(
        number_of_protected_samples,
        noise_multiplier=1.0,
        l2_norm_clip=0.5,
        batch_size=32,
        epochs=200,
        microbatches=None):

    if number_of_protected_samples > batch_size:
        raise NotImplementedError("Protecting more samples than a batch contains does not make sense. Abort.")

    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer, DPKerasAdamOptimizer
    import tensorflow as tf

    model = resnet_v1(input_shape=(32, 32, 3), depth=20)

    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=microbatches,
        lr=lr_schedule(0))

    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)

    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # model.compile(loss='sparse_categorical_crossentropy',
    #           optimizer=Adam(lr=lr_schedule(0)),
    #           metrics=['accuracy'])

    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    def train_func(x_train, y_train):
        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

        callbacks = [lr_reducer, lr_scheduler]

        model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

    return model, train_func



# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
# smatikov: https://github.com/pytorch/examples/blob/0c1654d6913f77f09c0505fb284d977d89c17c1a/word_language_model/model.py#L6

def get_model_LSTM_agnews_adam_crossentropy():

    model = Sequential()
    model.add(Embedding(TOKENIZER_MAX_NUM_WORDS, output_dim=100, input_length=INPUT_PAD_LENGTH_AGNews))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_func(x_train, y_train):
        epochs = 5
        batch_size = 64

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    return model, train_func


def get_model_LSTM_20News_adam_crossentropy():

    # optimizer = Adam(clipvalue=0.5)

    model = Sequential()
    model.add(Embedding(TOKENIZER_MAX_NUM_WORDS_20NEWS, output_dim=100, input_length=INPUT_PAD_LENGTH_20NEWS))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_func(x_train, y_train):
        epochs = 5
        batch_size = 64

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    return model, train_func


def get_model_LSTM_20News_18828_adam_crossentropy_GLOVE():
    # inspired by  https://www.programmersought.com/article/84883020319/
    # and https://www.kaggle.com/jannesklaas/19-lstm-for-email-classification

    GLOVE_FILE = os.path.join(CACHE_DIR, 'glove.6B.100d.txt')  # for embedding matrix
    # EMBEDDING_DIM = 100

    from poisoned_dataset.poisoned_dataset_word_based import PoisonedDataset_20News_18828

    EMBEDDING_DIM = PoisonedDataset_20News_18828.EMBEDDING_DIM
    WORD_INDEX_PICKLE_FILE = PoisonedDataset_20News_18828.WORD_INDEX_PICKLE_FILE

    with open(WORD_INDEX_PICKLE_FILE, 'rb') as f:
        word_index = pickle.load(f)

    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(GLOVE_FILE, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index) )

    # Create a matrix of all embeddings
    all_embs = np.stack(embeddings_index.values())
    emb_mean = all_embs.mean()  # Calculate mean
    emb_std = all_embs.std()  # Calculate standard deviation

    nb_words = min(TOKENIZER_MAX_NUM_WORDS_20NEWS, len(word_index))
    # embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > TOKENIZER_MAX_NUM_WORDS_20NEWS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(embedding_matrix.shape)

    # embedding_layer = Embedding(
    #     nb_words,
    #     EMBEDDING_DIM,
    #     weights=[embedding_matrix],
    #     input_length=INPUT_PAD_LENGTH_20NEWS,
    #     trainable=True,  # trainable, because of our W is word2vec trained, counted as pre-training model, so there is no need trained.
    #     # dropout=0.2
    # )
    print('Build model...')

    model = Sequential()
    model.add(Embedding(
        nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=INPUT_PAD_LENGTH_20NEWS,
        trainable=False)
    )
    model.add(LSTM(128))
    model.add(Dense(20))
    model.add(Activation('softmax'))

    # model = Sequential()
    # model.add(embedding_layer)
    # # model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))  # Output dimensions: 100
    # model.add(LSTM(100, dropout=0.2))  # Output dimensions: 100
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    # model.add(Dense(20, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    def train_func(x_train, y_train):
        epochs = 20
        batch_size = 32

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    return model, train_func


def get_model_CNN_20News_rmsprop_crossentropy():
    # inspired by "http://ai.intelligentonlinetools.com/ml/text-classification-20-newsgroups-dataset-using-convolutional-neural-network/"

    # best we got was 0.30 training set accuracy.

    MAX_SEQUENCE_LENGTH = 1000
    EMBEDDING_DIM = 100

    # model = Sequential()
    # model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    # model.add(Embedding(INPUT_PAD_LENGTH_20NEWS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    # model.add(Conv1D(128, 5, activation='relu'))
    # model.add(MaxPooling1D(5))
    # model.add(Conv1D(128, 5, activation='relu'))
    # model.add(MaxPooling1D(5))
    # model.add(Conv1D(128, 5, activation='relu'))
    # model.add(MaxPooling1D(35))  # global max pooling
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(2, activation='softmax'))

    # model.compile(loss='categorical_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['acc'])

    # def train_func(x_train, y_train):
    #     epochs = 1
    #     batch_size = 128

    #     model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
    #         callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    # return model, train_func

    input_layer = Input(shape=(INPUT_PAD_LENGTH_20NEWS,), dtype='int32')
    embedded_sequences = Embedding(TOKENIZER_MAX_NUM_WORDS_20NEWS, EMBEDDING_DIM, input_length=INPUT_PAD_LENGTH_20NEWS, trainable=True)(input_layer)

    convs = []
    for fsz in [3, 4, 5]:
        l_conv = Conv1D(filters=128, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Concatenate()(convs)
    l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(20, activation='softmax')(l_dense)

    model = Model(input_layer, preds)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    def train_func(x_train, y_train):
        epochs = 30
        batch_size = 50

        print(y_train.shape, x_train.shape)

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    return model, train_func


# 20NEWS Keras bert
# https://www.kaggle.com/sharmilaupadhyaya/20newsgroup-classification-using-keras-bert-in-gpu