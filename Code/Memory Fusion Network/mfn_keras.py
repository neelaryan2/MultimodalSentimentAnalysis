#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adadelta, Adam

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os
import argparse

from utils_nlp import *
from memory_fusion_network import *
from sklearn.model_selection import train_test_split

def load_data(data_path):
    """This function is used to load data which follows
    the python dict format as speceified according to 
    CMU mmsdk.
    Args:
        data_path: path to the pickle file which contains 
            the required data.
    Returns:
        X: a list of numpy arrays for each modality
        y: the output in a one-hot encoded representation
    """
    global output_dim

    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    X = [np.concatenate(tuple(data[typ][mod] for typ in ['train', 'valid', 'test']), axis=0) for mod in ['text', 'audio', 'vision']]
    y = np.squeeze(np.concatenate(tuple(data[typ]['labels'] for typ in ['train', 'valid', 'test']), axis=0))

    if len(y.shape) != 1:
        y = np.argmax(y, axis=1)

    mn, mx = np.min(y), np.max(y)
    shift = (mn + mx) / 2

    limit = 1
    if 'iemocap' in data_path:
        limit = 2
    roundoff = np.vectorize(lambda t: round_label(t, limit))

    y = roundoff(y - shift)

    if ('youtube' not in data_path) and ('moud' not in data_path):
        X, y = remove_labels(X, y, [0])
    y = to_categorical(re_label(y))

    return X, y

if __name__ == '__main__':
	# set fixed seed to ensure reproducibility across different runs
	np.random.seed(1337)
	tf.random.set_seed(1337)
	K.set_floatx('float64')

	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type=str, default='youtube_data.pkl', help="the path to the dataset to use for training and testing.")
	parser.add_argument("--mode", type=str, default='TAV', choices=['T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV'], help="the combination of modality to use for training.")
	parser.add_argument("--mem", type=int, default=512, help="the hyperparameter memory size for multiview gated memory.")

	args = parser.parse_args()

	# Load the data
	data_name = os.path.basename(args.data).split('.')[0]
	modals = {'T': [0], 'A': [1], 'V': [2], 'TA': [0, 1], 'TV': [0, 2], 'AV': [1, 2], 'TAV': [0, 1, 2]}
	data, labels = load_data(args.data)
	print([t.shape for t in data])
	print(labels.shape)

	# mode variable can be changed to try different modalities to train each model
	# test set is 20% of the complete data set
	X = [data[i] for i in modals[args.mode]]
	y = labels.astype(np.float64)

	X_train, X_test = [], []
	for modality in X:
	    train, test = train_test_split(modality, test_size=0.2, shuffle=False)
	    X_train.append(train)
	    X_test.append(test)

	y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)

	# filename will be used to save these metrics as images in the following cell
	classes = sorted(list(set(sorted(list(np.argmax(y_train, axis=1))))))
	filename = '_'.join([data_name, args.mode, str(len(classes)) + 'way', f'memsz{args.mem}'])

	# clear session is important for keras to clear out previous models
	# and prevent slowdowns, comlpetely resets the backend state
	K.clear_session()
	input_shapes = [d.shape[1:] for d in X_train]
	mfn = MFN(input_shapes, output_classes=y.shape[1], mem_size=args.mem)
	mfn.summary()

	# ADAM algorithm is used for optimization of the model parameters
	# checkpointer is added to save the weights having best validation accuracy
	# in case, we overtrain the model and it starts overfitting
	optimizer = Adam(lr=5e-4)
	model_name = filename + '_model.h5'
	early_stopping = EarlyStopping(monitor='val_accuracy', patience=50, mode='max', verbose=1)
	checkpointer = ModelCheckpoint(filepath=model_name, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
	mfn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	# validation split is 15% which means 12% of the complete data
	# We use a large batch size because we are training over GPU
	history = mfn.fit(X_train, y_train, batch_size=128 * 4, epochs=50, shuffle=True, validation_split=0.15, callbacks=[checkpointer, early_stopping])

	# Load the model with best validation accuracy
	if os.path.isfile(model_name):
	    mfn.load_weights(model_name)
	    print('Loaded saved Model!')

	# confusion matrix and classification report are generated here
	print('Base Filename :', filename)
	preds = mfn.predict(X_test)
	calc_test_result(preds, y_test)

	# We save the metrics as png files, for further evaluation/analysis
	y_true = np.argmax(y_test, axis=1)
	y_pred = np.argmax(preds, axis=1)
	class_names, report, support = get_report(y_true, y_pred, classes)
	cm = confusion_matrix(y_true, y_pred, labels=classes)
	scores = get_scores(y_true, y_pred, classes)

	plot_tag_scores(classes, scores, filename=filename + '_tag_scores')
	plot_confusion_matrix(classes, cm, filename=filename + '_cm')
	plot_clf_report(class_names, report, support, filename=filename + '_clf_report')
