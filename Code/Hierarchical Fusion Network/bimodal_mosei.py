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
    #Function to calculate results given output of model and true values
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


def Bimodal():
    '''
    Loads unimodal activations saved at "../input/mosei-dataset-3way/text_3way.pickle" and "../input/mosei-dataset-3way/audio_3way.pickle"
    The pickle files must contain a 2 element lists of numpy arrays with the first element having the feature vectors and the second with the labels
    Trains bimodal layers using the input feature vector(text and audio only). The bimodal model layers are defined herein
    Prints statistics of prediction on test set

    Returns:
    final: a 3D numpy array of predictions of model in one hot encoded form (test_size by sequence_length by number of classes)
    test_label: a 3D numpy array of true labels in one hot encoded form (test_size by sequence_length by number of classes)
    '''
    with open('../input/mosei-dataset-3way/text_3way.pickle', 'rb') as handle:
        text_activations = pickle.load(handle, encoding = 'latin1')
    
    with open('../input/mosei-dataset-3way/audio_3way.pickle', 'rb') as handle:
        audio_activations = pickle.load(handle, encoding = 'latin1')
    
    #Extracting train and test data of both modalities
    data_t = text_activations[0]
    data_a = audio_activations[0]
    label = text_activations[1]
    
    #Get indices of one-hot labels
    label_i=np.argmax(label,axis=-1)
    
    #Zeroing out data with label 0 (due to skewness); will aid Masking layer of the Model
    for i in range(data_t.shape[0]):
        for j in range(data_t.shape[1]):
            if label_i[i][j] == 0 :
                data_t[i,j,:]=0.0
                data_a[i,j,:]=0.0
                
        
    print(data_t.shape)
    print(data_a.shape)
    print(label.shape)
    
    dim_1 = data_t.shape[0]
    max_len = data_t.shape[1]
    dim_text = data_t.shape[2]
    dim_aud = data_a.shape[2]
    
    
    #Get the text features and audio features as input
    x=Input(shape=(max_len,dim_text,), name='bi_t_input')
    y=Input(shape=(max_len,dim_aud,), name='bi_a_input')
    #Masking the '0' value input, i.e., the ones with label 0
    x1=Masking(mask_value =0.0 , name='bi_mask_t')(x)
    y1=Masking(mask_value =0.0 , name='bi_mask_a')(y)
    #Pass the result through a dense layer to project to suitable dimensions (optimal value obtained through parameter search)
    d1=Dense(200, activation='tanh', use_bias=True, trainable=True, name='bid1')(x1)
    d2=Dense(100, activation='tanh', use_bias=True, trainable=True, name='bid2')(y1)
    #Concatenate the two modalities
    c1=Concatenate(axis=-1)([d1, d2])
    #Pass the concatenated result through a GRU; since the dataset is word aligned we set return_seq = True
    g1=GRU(300, return_sequences=True, dropout=0.2, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-3))(c1)
    #The result of GRU is passed through a softmax layer to obtain output
    d3=TimeDistributed(Dense(NUM_LABELS,activation='softmax',kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-3)))(g1)
    
    model=Model([x, y], outputs=d3)
    #Model is trained using an Adam optimiser and crossentropy loss function
    model.compile(loss='categorical_crossentropy',optimizer=Adam(0.0001),metrics=['accuracy'])
    print(model.summary())
    
    #Splitting the data to train and test examples
    test_data_t=data_t[:100]
    test_data_a = data_a[:100]
    test_label=label[:100]
    model.fit([data_t[100:], data_a[100:]], label[100:],epochs=5,batch_size=20,shuffle=True)
    
    #Model outputs of the test data calculated and evaluated
    final = model.predict([test_data_t, test_data_a])
    calc_test_result(final, test_label)
    return final, test_label
    


# In[4]:


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

#Generate tag scores
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



# In[ ]:


if __name__=='__main__':
    NUM_LABELS=3 #MOSEI has 3 labels
    result, test_label = Bimodal()
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

