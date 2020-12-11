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

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

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
	parser.add_argument("--k", type=int, default=5, help="k-Fold validation.")

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
	y_train, y_test = [], []

	# k Fold cross validation
	kfold_size = y.shape[0] // args.k

	for i in range(args.k):
	    l = i * kfold_size
	    r = l + kfold_size
	    if i == args.k - 1 : r = y.shape[0]

	    X_train_i, X_test_i = [], []
	    for modality in X:
	        test = modality[l:r]
	        train = np.concatenate((modality[:l], modality[r:]), axis=0)
	        X_train_i.append(train)
	        X_test_i.append(test)
	    y_train_i = np.concatenate((y[:l], y[r:]), axis=0)
	    y_test_i = y[l:r]
	    
	    X_train.append(X_train_i)
	    X_test.append(X_test_i)

	    y_train.append(y_train_i)
	    y_test.append(y_test_i)


	# filename will be used to save these metrics as images in the following cell
	classes = sorted(list(set(sorted(list(np.argmax(y, axis=1))))))
	filename = '_'.join([data_name, args.mode, str(len(classes))+'way', f'memsz{args.mem}', f'{args.k}fold'])

	# clear session is important for keras to clear out previous models
	# and prevent slowdowns, comlpetely resets the backend state
	K.clear_session()
	input_shapes = [d.shape[1:] for d in X]
	models = [None]*args.k
	for i in range(args.k):
	    models[i] = MFN(input_shapes, output_classes=y.shape[1], mem_size=args.mem)


	# ADAM algorithm is used for optimization of the model parameters
	# checkpointer is added to save the weights having best validation accuracy
	# in case, we overtrain the model and it starts overfitting
	checkpointer = [None]*args.k
	early_stopping = [None]*args.k
	for i in range(args.k):
	    optimizer = Adam(lr=5e-4)
	    model_name = f'{filename}_model_{i+1}.h5'
	    early_stopping[i] = EarlyStopping(monitor='val_accuracy', patience=50, mode='max', verbose=1)
	    checkpointer[i] = ModelCheckpoint(filepath=model_name, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
	    models[i].compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


	# validation split is 15% which means 12% of the complete data
	# We use a large batch size because we are training over GPU
	history = [None]*args.k
	for i in range(args.k):
	    history[i] = models[i].fit(X_train[i], y_train[i],
	                              batch_size = 128 * 4,
	                              epochs = 50,
	                              shuffle = True,
	                              validation_split = 0.15,
	                              callbacks = [checkpointer[i], early_stopping[i]])


	# Load the models with best validation accuracy
	for i in range(args.k):
	    model_name = f'{filename}_model_{i+1}.h5'
	    if os.path.isfile(model_name):
	        models[i].load_weights(model_name)
	        print(f'Loaded saved Model Fold {i+1}!')


	# confusion matrix and classification report are generated and averaged here
	print('Base Filename :', filename)
	y_preds, y_trues = [], []
	acc = 0.0
	prfs_m = np.zeros((3,), dtype=np.float64)
	prfs_w = np.zeros((3,), dtype=np.float64)
	cm = None
	for i in range(args.k):
	    preds = models[i].predict(X_test[i])
	    y_preds.append(np.argmax(preds, axis=1))
	    y_trues.append(np.argmax(y_test[i], axis=1))
	    
	    cm_i = confusion_matrix(y_trues[i], y_preds[i])
	    cm_i = cm_i.astype('float') / np.sum(cm_i, axis=1, keepdims=True)
	    if cm is None : cm = cm_i
	    else : cm += cm_i
	    
	    acc += accuracy_score(y_trues[i], y_preds[i])
	    prfs_m += np.array(precision_recall_fscore_support(y_trues[i], y_preds[i], zero_division=0, average='macro')[:-1])
	    prfs_w += np.array(precision_recall_fscore_support(y_trues[i], y_preds[i], zero_division=0, average='weighted')[:-1])
	    
	print("Confusion Matrix :\n", cm / args.k)
	print("Accuracy ", acc / args.k)
	print("Macro Classification Report :", prfs_m / args.k)
	print("Weighted Classification Report :", prfs_w / args.k)

	# average your cm, report, score and support variables over k runs
	class_names = None
	report = None
	support = None
	cm, scores = None, None

	# We save the metrics as png files, for further evaluation/analysis
	for y_true, y_pred in zip(y_trues, y_preds):
	    class_names, report_, support_ = get_report(y_true, y_pred, classes)
	    cm_ = confusion_matrix(y_true, y_pred, labels=classes)
	    scores_ = get_scores(y_true, y_pred, classes)
	    
	    if report is None : report = np.zeros_like(report_, dtype=np.float64)
	    report += report_
	    
	    if support is None : support = np.zeros_like(support_, dtype=np.float64)
	    support += support_
	    
	    if cm is None : cm = np.zeros_like(cm_, dtype=np.float64)
	    cm += cm_
	    
	    if scores is None : scores = np.zeros_like(scores_, dtype=np.float64)
	    scores += scores_

	report /= args.k
	support /= args.k
	cm /= args.k
	scores /= args.k

	plot_tag_scores(classes, scores, filename=filename+'_tag_scores')
	plot_confusion_matrix(classes, cm, filename=filename+'_cm')
	plot_clf_report(class_names, report, support, filename=filename+'_clf_report')
