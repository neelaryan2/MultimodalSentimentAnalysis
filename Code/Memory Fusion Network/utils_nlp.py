import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

def re_label(y):
	"""re_label function takes in input a list/array of any hashable
	data type, for example, strings, integers etc, which represent class labels.
	This function replaces all the class labels with integers 0 to K-1 where
	K is the number of classes (distinct elements present in input).
	Args: 
		y: python list/numpy array with each element a hashable data type
	Returns: 
		relabeled numpy array in integer format
	"""
	labels = sorted(list(set(list(y))))
	inv = {v:i for i,v in enumerate(labels)}
	return np.array([inv[v] for v in y], dtype=np.int32)

def remove_labels(X, y, labels_to_remove):
	"""remove_labels does exactly what the name suggests. This function removes
	instances from both X and y where that instance has a label which is provided
	in the argument labels_to_remove. Instances refer to rows in the data X and 
	labels y.
	Args:
		X: data having same number of instances as y
		y: numpy array of labels corresponding to the feature matrix X
			labels_to_remove: a list of labels which will be removed from the data
			as well as label list.
	Returns:
		X_new: X with some removed rows
		y_new: y with some removed rows
	"""
	indices = []
	for i in range(y.shape[0]):
		if y[i] in labels_to_remove:
			indices.append(i)
	if isinstance(X, list):
		X_new = [np.delete(X[i], indices, axis=0) for i in range(len(X))]
	else:
		X_new = np.delete(X, indices, axis=0)
	y_new = np.delete(y, indices, axis=0)
	return X_new, y_new

def to_categorical(y, num_classes=None, dtype='float32'):
	"""to_categorical converts labels to their one-hot encoded representation.
	Args:
		y: array/list of labels within range 0 to K-1 where K <= num_classes
		num_classes: the number of classes to encoded, if None, max(y)+1 is used
		dtype: dtype of the output array
	Returns:
		one-hot encoded vector of shape NxK where N is number of examples and K is 
		number of classes.
	"""
	y = np.array(y, dtype='int')
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes), dtype=dtype)
	categorical[np.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = np.reshape(categorical, output_shape)
	return categorical

def round_label(a, limit=1):
	"""round_label rounds off labels where the label list actually is a continuous
	real value depicting the intensity of the sentiment. 
	Negative values are rounded off to nearest smaller integer
	Positive values are rounded off to nearest larger integer 
	The output is constrained to be in the range [-limit,limit]
	Args:
		a: real number (treated as intensity)
		limit: argument for constraining the output
	Returns:
		rounded off value
	"""
	if int(a) == a:
		res = int(a)
	elif a > 0:
		res = int(a) + 1
	else:
		res = int(a) - 1
	res = max(res, -limit)
	res = min(res, limit)
	return res

def calc_test_result(output, actual):
	"""This function prints the confusion matrix (normalised i.e. rows sum to 1) 
	and the classification report using sklearn module.
	Args:
		output: numpy array representing class probabilities
		actual: actual label (may be one hot encoded or class probabilities)
	Returns:
		No return value, prints confusion matrix and report to stdout
	"""
	y_true = np.argmax(actual, axis=1)
	y_pred = np.argmax(output, axis=1)
	print("Confusion Matrix :")
	cm = confusion_matrix(y_true, y_pred)
	cm = cm.astype('float') / np.sum(cm, axis=1, keepdims=True)
	print(np.around(cm, decimals=2))
	print("Classification Report :")
	report = classification_report(y_true, y_pred, digits=4, zero_division=0)
	report = report.replace('\n\n', '\n')
	print(report, end='')
	
def get_report(y_true, y_pred, classes):
	"""This function parses the classification report given by sklearn to
	get all the row names metric values as floats and supports for each
	class label.
	Args:
		y_true: true (numerical) labels of data
		y_pred:	predicted (numerical) labels of the same data
		classes: a python list of class labels
	Returns:
		class_names: a python list of class labels (here, row names from report)
		plotMat: numerical values (metrics) in the classification report
		support: the number of instances for each class_name present in report
	"""
	clf_report = classification_report(y_true, y_pred, digits=4, labels=classes, zero_division=0)
	clf_report = clf_report.replace('\n\n', '\n')
	clf_report = clf_report.replace('micro avg', 'micro_avg')
	clf_report = clf_report.replace('macro avg', 'macro_avg')
	clf_report = clf_report.replace('weighted avg', 'weighted_avg')
	clf_report = clf_report.replace(' / ', '/')
	lines = clf_report.split('\n')

	class_names, plotMat, support = [], [], []
	for line in lines[1:]:
		t = line.strip().split()
		if len(t) < 2:
			continue
		v = [float(x) for x in t[1: len(t) - 1]]
		if len(v) == 1 : v = v * 3
		support.append(int(t[-1]))
		class_names.append(t[0])
		plotMat.append(v)
	plotMat = np.array(plotMat)
	support = np.array(support)
	return class_names, plotMat, support

def get_scores(y_true, y_pred, classes):
	"""This function calculates the correct and incorrect counts for each label
	as a fraction to the total instances of that class.
	Args:
		y_true: true (numerical) labels of data
		y_pred:	predicted (numerical) labels of the same data
		classes: a python list of class labels
	Returns:
		numpy array of tuple of (correct,incorrect) fractions for each class
	"""
	correct, wrong = {}, {}
	for tag in classes:
		correct[tag] = 0
		wrong[tag] = 0
		
	for tag, pred in zip(y_true, y_pred):
		if tag == pred:
			correct[tag] += 1
		else:
			wrong[tag] += 1
			
	scores = []
	total = len(y_true)
	for tag in classes:
		cur = np.array([correct[tag], wrong[tag]])
		scores.append(cur / total)
	return np.array(scores)
	
def plot_confusion_matrix(classes, mat, normalize=True, cmap=plt.cm.Blues, filename='confusion_matrix'):
	"""This function plots the confusion matrix as an image, using the 
	parsed values from the confusion matrix and saves the image in 
	the current working directory.
	Args:
		classes: a python list of class labels
		mat: numerical values (metrics) in the confusion matrix
		normalize: controls the normalization of the confusion 
			matrix (rows sum to 1 or not)
		cmap: the color map to be used in the output image
		filename: the filename with which the plot will be saved (can be a path too)
	Returns:
		No return value. Shows and saves the confusion matrix image.
	"""
	k = len(classes)
	cm = np.copy(mat)
	title = 'Confusion Matrix (without normalization)'
	if normalize:
		cm = cm.astype('float') / np.sum(cm, axis=1, keepdims=True)
		title = title.replace('without', 'with')
	fig, ax = plt.subplots(figsize=(20,10))
	ax.set_title(title, y=-0.06, fontsize=66/k)
	ax.xaxis.set_ticks_position('top')
	ax.xaxis.set_label_position('top')
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.clim(vmin=0.0, vmax=1.0)
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=40/k) 
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = np.max(cm) / 2
	thresh = 1 / 2
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			color = "white" if (cm[i, j] > thresh) else "black"
			plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color=color, fontsize=50/k)
	plt.ylabel('True label',fontsize=22)
	plt.xlabel('Predicted label', fontsize=22)
	plt.tight_layout()
	plt.savefig(filename+'.png', bbox_inches="tight", transparent=True)
	plt.show()
	
def plot_clf_report(classes, plotMat, support, cmap=plt.cm.Blues, filename='classification_report'):
	"""This function plots the classification report as an image, using the 
	parsed values from the sklearn classification report and saves the image in 
	the current working directory.
	Args:
		classes: a python list of class labels
		plotMat: numerical values (metrics) in the classification report
		support: the number of instances for each class present in report
		cmap: the color map to be used in the output image
		filename: the filename with which the plot will be saved (can be a path too)
	Returns:
		No return value. Shows and saves the report image.
	"""
	k = len(classes)
	title = 'Classification Report'
	xticklabels = ['Precision', 'Recall', 'F1-score']
	yticklabels = ['{0} ({1})'.format(classes[idx], sup) for idx, sup in enumerate(support)]
	fig, ax = plt.subplots(figsize=(22,12))
	ax.set_title(title, y=-0.06, fontsize=22)
	ax.xaxis.set_ticks_position('top')
	ax.xaxis.set_label_position('top')
	ax.xaxis.set_tick_params(labelsize=18)
	ax.yaxis.set_tick_params(labelsize=14)
	plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
	plt.clim(vmin=0.0, vmax=1.0)
	cbar = plt.colorbar()
	cbar.ax.tick_params(labelsize=80/k) 
	plt.xticks(np.arange(3), xticklabels, rotation=0)
	plt.yticks(np.arange(len(classes)), yticklabels)
	thresh = np.max(plotMat) / 2
	thresh = 1 / 2
	for i in range(plotMat.shape[0]):
		for j in range(plotMat.shape[1]):
			color = "white" if (plotMat[i, j] > thresh) else "black"
			plt.text(j, i, format(plotMat[i, j], '.2f'), horizontalalignment="center", color=color, fontsize=100/k)

	plt.xlabel('Metrics',fontsize=22)
	plt.ylabel('Classes',fontsize=22)
	plt.tight_layout()
	plt.savefig(filename+'.png', bbox_inches="tight", transparent=True)
	plt.show()
	
def plot_tag_scores(classes, scores, filename='tag_scores'):
	"""This function plots the histogram for tag scores and saves the image in 
	the current working directory.
	Args:
		classes: a python list of class labels
		scores: a dictionary of correct and incorrect counts for each label
		filename: the filename with which the plot will be saved (can be a path too)
	Returns:
		No return value. Shows and saves the tag scores plot.
	"""
	width = 0.45
	fig, ax = plt.subplots(figsize=(20,10))
	ax.xaxis.set_tick_params(labelsize=18, rotation=25)
	ax.yaxis.set_tick_params(labelsize=18)
	range_bar1 = np.arange(len(classes))
	rects1 = ax.bar(range_bar1, tuple(scores[:, 0]), width, color='b')
	rects2 = ax.bar(range_bar1 + width, tuple(scores[:, 1]), width, color='r')

	ax.set_ylabel('Scores',fontsize=22)
	ax.set_title('Tag scores', fontsize=22)
	ax.set_xticks(range_bar1 + width / 2)
	ax.set_xticklabels(classes)

	ax.legend((rects1[0], rects2[0]), ('Correct', 'Wrong'), fontsize=20)
	plt.legend()
	plt.savefig(filename+'.png', bbox_inches="tight", transparent=True)
	plt.show()