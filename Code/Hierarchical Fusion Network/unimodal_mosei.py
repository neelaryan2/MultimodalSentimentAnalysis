#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(1337) # for reproducibility
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input,Dense,GRU,LSTM,Concatenate,Dropout,Activation,Add, Masking
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.merge import Concatenate
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
import sys
from sklearn import metrics
from matplotlib import pyplot as plt


# In[2]:


def calc_test_result(result, test_label):
    '''
    Generates various classification stats in terminal

    Args:
    result: a 3D numpy array of predictions of model in one hot encoded form (test_size by sequence_length by number of classes)
    test_label: a 3D numpy array of true labels in one hot encoded form (test_size by sequence_length by number of classes)

    Returns: None
    '''
    true_label=[]
    predicted_label=[]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if np.argmax(test_label[i,j] )!=0 and np.argmax(result[i,j] )!=0:# We do not consider the data with label 0 in out statistics due to skewness
                true_label.append(np.argmax(test_label[i,j] ))
                predicted_label.append(np.argmax(result[i,j] ))
    
    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label,digits=4))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    print("Macro Classification Report :")
    print(precision_recall_fscore_support(true_label, predicted_label,average='macro'))
    print("Weighted Classification Report :")
    print(precision_recall_fscore_support(true_label, predicted_label,average='weighted'))


# In[3]:


def Unimodal_text():
    '''
    Loads unimodal activations saved at "../input/mosei-dataset-3way/text_3way.pickle" 
    The pickle file must contain a 2 element list of numpy arrays with the first element having the feature vectors and the second with the labels
    Trains unimodal layers using the input feature vector. The unimodal model layers are defined herein
    Prints statistics of prediction on test set

    Returns:
    final: a 3D numpy array of predictions of model in one hot encoded form (test_size by sequence_length by number of classes)
    test_label: a 3D numpy array of true labels in one hot encoded form (test_size by sequence_length by number of classes)
    model: The final trained model
    '''
    
    with open('../input/mosei-dataset-3way/text_3way.pickle', 'rb') as handle:
        text_activations = pickle.load(handle, encoding = 'latin1')
        
    #Extracting train and test data of both modalities
    data_t = text_activations[0]
    label = text_activations[1]
    
    #Get indices of one-hot labels
    label_i=np.argmax(label,axis=-1)
    
    #Zeroing out data with label 0 (due to skewness); will aid Masking layer of the Model
    for i in range(data_t.shape[0]):
        for j in range(data_t.shape[1]):
            if label_i[i][j] == 0 :
                data_t[i,j,:]=0.0
                
    print(data_t.shape)
    print(label.shape)
    
    dim_1 = data_t.shape[0]
    max_len = data_t.shape[1]
    dim_text = data_t.shape[2]
    
    
    model = Sequential()
    
    #Get the text features as input
    model.add(Input(shape=(max_len, dim_text,), name='Input_unimodal'))
    #Masking the '0' value input, i.e., the ones with label 0
    model.add(Masking(mask_value=0.0))
    #Pass the result through a GRU; since the dataset is word aligned we set return_seq = True
    model.add(GRU(300, return_sequences=True, dropout=0.3, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-3)))
    #The result of GRU is passed through a softmax layer to obtain output
    model.add(TimeDistributed(Dense(NUM_LABELS,activation='softmax',kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-3))))
    
    #Model is trained using an Adam optimiser and crossentropy loss function
    model.compile(loss='categorical_crossentropy',optimizer=Adam(0.0001),metrics=['accuracy'])
    print(model.summary())
    
    #Splitting the data to train and test examples
    test_data_t=data_t[:100]
    test_label=label[:100]
    model.fit(data_t[100:], label[100:], epochs=5, batch_size=20, validation_split=0.1)
    
    #Model outputs of the test data calculated and evaluated
    final = model.predict(test_data_t)
    calc_test_result(final, test_label)
    
    return final, test_label, model


# In[4]:


def loadGloveModel(File):
    '''
    Loads word embeddings from specified file
    Args:
    File: file path of word embeddings
    
    Returns:
    gloveModel: dictionary containing key, value pairs of word and its embedding as an numpy array
    '''
    #Loading the glove word embeddings as a dictionary
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel


# In[5]:


#generate classification report
def get_report(y_true, y_pred, classes):
    '''This function parses the classification report given by sklearn to
    get all the row names metric values as floats and supports for each
    class label.
    Args:
        y_true: true (numerical) labels of data
        y_pred: predicted (numerical) labels of the same data
        classes: a python list of class labels
    Returns:
        class_names: a python list of class labels (here, row names from report)
        plotMat: numerical values (metrics) in the classification report
        support: the number of instances for each class_name present in report
    '''
    clf_report = classification_report(y_true, y_pred, labels=classes, zero_division=0)
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

#generate tag scores
def get_scores(y_true, y_pred, classes):
    '''This function calculates the correct and incorrect counts for each label
    as a fraction to the total instances of that class.
    Args:
        y_true: true (numerical) labels of data
        y_pred: predicted (numerical) labels of the same data
        classes: a python list of class labels
    Returns:
        numpy array of tuple of (correct,incorrect) fractions for each class
    '''
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

#generate and plot confusion matrix
def plot_confusion_matrix(classes, mat, normalize=True, cmap=plt.cm.Blues):
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
    cm = np.copy(mat)
    title = 'Confusion Matrix (without normalization)'
    if normalize:
        cm = cm.astype('float') / np.sum(cm, axis=1, keepdims=True)
        title = title.replace('without', 'with')
    plt.clf()    
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title(title, y=-0.06, fontsize=22)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.clim(vmin=0.0, vmax=1.0)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = np.max(cm) / 2
    thresh = 1 / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if (cm[i, j] > thresh) else "black"
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color=color)
    plt.ylabel('True label',fontsize=22)
    plt.xlabel('Predicted label', fontsize=22)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches="tight", transparent=True)
    plt.show()

#plot the generated classification report
def plot_clf_report(classes, plotMat, support, cmap=plt.cm.Blues):
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
    title = 'Classification Report'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(classes[idx], sup) for idx, sup in enumerate(support)]
    plt.clf()
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title(title, y=-0.06, fontsize=22)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.clim(vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.xticks(np.arange(3), xticklabels, rotation=0)
    plt.yticks(np.arange(len(classes)), yticklabels)
    thresh = np.max(plotMat) / 2
    thresh = 1 / 2
    for i in range(plotMat.shape[0]):
        for j in range(plotMat.shape[1]):
            color = "white" if (plotMat[i, j] > thresh) else "black"
            plt.text(j, i, format(plotMat[i, j], '.2f'), horizontalalignment="center", color=color, fontsize=14)

    plt.xlabel('Metrics',fontsize=22)
    plt.ylabel('Classes',fontsize=22)
    plt.tight_layout()
    plt.savefig('classification_report.png', bbox_inches="tight", transparent=True)
    plt.show()

#plot the generated tag scores
def plot_tag_scores(classes, scores, normalize=True):
    """This function plots the histogram for tag scores and saves the image in 
    the current working directory.
    Args:
        classes: a python list of class labels
        scores: a dictionary of correct and incorrect counts for each label
        filename: the filename with which the plot will be saved (can be a path too)
    Returns:
        No return value. Shows and saves the tag scores plot.
    """
    plt.clf()
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
    plt.savefig('tag_scores.png', bbox_inches="tight", transparent=True)
    plt.show()



# In[6]:


def main_func():
    '''
    Contains variable stmt, change it to see the output of any example sentence. Runs and trains the model, prints and saves the confusion matrix,
    tags scores, reports
    Args: None
	'''
    NUM_LABELS=3 #MOSEI has 3 labels
    result, test_label, model = Unimodal_text()
    
    # Load the word embedding model at "../input/nlpword2vecembeddingspretrained/glove.6B.300d.txt"
    glovemod=loadGloveModel("../input/nlpword2vecembeddingspretrained/glove.6B.300d.txt")
    
    # Demo of trained model in action
    stmt = "I am disappointed and angry I hate the people there Their attitude is irritating They should be ashamed"#negative
    stmt = "Everything is so pretty and bright and happy I love it here I wish every day was just like this wonderful and joyous day"#positive
    stmt = "I am annoyed and disappointed that he had to leave us and we will all miss him desparately but I am happy he is getting to live his dream"#both
    words = stmt.lower().split() #list of words in the statement
    inp = []
    l = len(words)
    for i in range(98):#98 is the sequence length for the model
        if i<len(words):
            inp.append(glovemod[words[i]])
        else:
            inp.append(np.zeros((300)))#padding to make length 98
    inp = np.array(inp)
    inp = [inp]
    inp = np.array(inp)
    print(inp.shape)#(1 annotation, 98 seq length, 300 embedding size)
    
    op = model.predict(inp)
    op =np.argmax(op,axis=-1)#get predicted label for each word of annotation
    print(words)
    print(op[0][0:l])#Indexed so as to not show o/p for padding token
    # Label 1: positive sentiment
    # Label 2: negative sentiment
    
    #Start of plotting code
    y_true=[]
    y_pred=[]

    #y_true is list of true values; y_pred is list of predicted values
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if np.argmax(test_label[i,j] )!=0 and np.argmax(result[i,j] )!=0:
                y_true.append(np.argmax(test_label[i,j] ))
                y_pred.append(np.argmax(result[i,j] ))
    
    classes=[1,2]
    class_names, report, support = get_report(y_true, y_pred, classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    scores = get_scores(y_true, y_pred, classes)
    plot_clf_report(class_names, report, support)
    plot_confusion_matrix(classes, cm)
    plot_tag_scores(classes, scores)

if __name__=='__main__':
	main_func()