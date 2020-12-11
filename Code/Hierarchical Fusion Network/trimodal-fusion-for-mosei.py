#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
np.random.seed(1337)
import keras
from keras import backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping

import pickle
from keras.layers.merge import Multiply,Concatenate
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.callbacks import Callback
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers import GRU,LSTM,Concatenate,Dropout,Masking,Input,Dense,Activation
from keras.layers.pooling import AveragePooling1D,MaxPooling1D
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from keras.layers.core import Reshape
from keras.backend import shape
from keras.utils import plot_model

from keras.callbacks import Callback
import sys






def createOneHot_mosei3(train_label):
    '''
    this function takes a np array of labels and converts it into a one hot encoded 2D matrix for use with categorical_crossentropy loss

    Args:
    train_label: a 1d numpy array of integers corresponding to labels
    test_label: a 1d numpy array of integers corresponding to labels

    Returns:
        a pair of 2D numpy arrays of size (dimension 0 of corresponding input, max label value in correspodning input+1)
    '''
    return keras.utils.to_categorical(train_label)


def test_res_maker(result, test_label, test_mask):
    '''
    Generates various classification stats in terminal

    Args:
    result: a 3D numpy array of predictions of model in one hot encoded form (test_size by sequence_length by number of classes)
    test_label: a 3D numpy array of true labels in one hot encoded form (test_size by sequence_length by number of classes)
    test_mask: a 2D numpy array telling which inputs to ignore while calculating accuracies

    Returns: None
    '''
    return calc_test_result_mosei3(result, test_label, test_mask)

def calc_test_result_mosei3(result, test_label, test_mask):
    '''
    Helper function for test_res_maker

    Args:
    result: a 3D numpy array of predictions of model in one hot encoded form (test_size by sequence_length by number of classes)
    test_label: a 3D numpy array of true labels in one hot encoded form (test_size by sequence_length by number of classes)
    test_mask: a 2D numpy array telling which inputs to ignore while calculating accuracies

    Returns: None
    '''
    true_label=[]
    predicted_label=[]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i,j]==1:
                true_label.append(float(test_label[i,j] ))
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





NUM_LABELS=3




def load_bim_acts_mosei3():
    '''
    loads the pickle file saved at "./bim_act.pickle" containing a dictionary of bimodal activations

    Returns:
    merged_train_data: 3D numpy array of input to trimodal layers
    merged_test_data: 3D numpy array to test trimodal layers
    train_label: 3D numpy array of one hot encoded labels
    test_label: 3D numpy array of one hot encoded labels to test trimodal layers 
    train_mask: 2D numpy array telling utternaces to ignore in train data
    test_mask: 2D numpy array telling utternaces to ignore in test data
    max_len: maximum sequence length
    '''   
    with open('./bim_act.pickle', 'rb') as handle:
        activations = pickle.load(handle, encoding = 'latin1')  
    merged_train_data = activations['train_bimodal']
    train_mask=activations['train_mask']
    train_label=activations['train_label']

    audio_dim,text_dim,visual_dim=[500]*3
    dim = audio_dim + visual_dim + text_dim
    max_len = merged_train_data.shape[1] #max number of utterances per video
    dim_proj=450
    return merged_train_data, train_label, train_mask, max_len


def load_unimodal_activations_mosei3():
    '''
    load unimodal activations saved at "../input/multimodal-sentiment/unimodal.pickle" 
    The pickle file must contain a dictionary of numpy arrays having the feature vectors with keys: 
    'audio_train', 'video_train', 'audio_test', 'text_train', 'test_mask', 'test_label', 'video_test', 'train_mask', 'text_test', 'train_label'

    Returns:
    merged_train_data: 3D numpy array of input to trimodal layers
    merged_test_data: 3D numpy array to test trimodal layers
    train_label: 3D numpy array of one hot encoded labels
    test_label: 3D numpy array of one hot encoded labels to test trimodal layers 
    train_mask: 2D numpy array telling utternaces to ignore in train data
    test_mask: 2D numpy array telling utternaces to ignore in test data
    '''
    with open('../input/mosei-dataset-3way/text_3way.pickle', 'rb') as file:
        uni_act1 = pickle.load(file, encoding = 'latin1')
    with open('../input/mosei-dataset-3way/audio_3way.pickle', 'rb') as file:
        uni_act2 = pickle.load(file, encoding = 'latin1')
    with open('../input/mosei-dataset-3way/video_3way.pickle', 'rb') as file:
        uni_act3 = pickle.load(file, encoding = 'latin1')

    merged_train_data = np.concatenate((uni_act1[0], uni_act2[0], uni_act3[0]), axis=2)
    train_label=np.argmax(uni_act1[1],axis=-1)
    train_mask=np.ones((uni_act1[0].shape[0],uni_act1[0].shape[1]))
    
    for i in range(uni_act1[0].shape[0]):
        for j in range(uni_act1[0].shape[1]):
            if train_label[i][j] == 0.0 :
                train_mask[i][j]=0
                merged_train_data[i,j,:]=0.0

    global audio_dim, visual_dim, text_dim, dim, max_len, dim_proj

    audio_dim,visual_dim,text_dim=[uni_act2[0].shape[2],uni_act3[0].shape[2],uni_act1[0].shape[2]]
    dim = audio_dim + visual_dim + text_dim
    max_len = merged_train_data.shape[1] #max number of utterances per video
    dim_proj=450

    return merged_train_data, train_label, train_mask


TRIGRU_SZ=500
def Bimodal():
    '''
    Trains bimodal layers using the input feature vector and creates the bim_act.pickle file. The bimodal models and fusion layers are defined herein

    Returns:
    None
    '''
    class bim_fusion_layer(Layer):
        '''
        This class is the bimodal fusion layers that carries out the fusion of pairs
        of modalities using weighted sum
        '''
        def __init__(self, prefix, **kwargs):
            '''
            Initialises the layer:
            Args:
            prefix: string that can be used to name internal weights
            kwargs: are the arguments to the superclass Layer
            '''
            self.supports_masking = True
            self.prefix = prefix
            super(bim_fusion_layer, self).__init__(**kwargs)

        def build(self, input_shape):
            '''
            Takes the input shame and initalises the traianble used for the hadamard product
            Args:
            input_shape: numpy array as required by the build in superclass Layer
            '''
            self.output_dim = dim_proj
            self.wt1 = self.add_weight(name=self.prefix+'wt1', shape=(self.output_dim,),initializer='TruncatedNormal',trainable=True)
            self.wt2 = self.add_weight(name=self.prefix+'wt2', shape=(self.output_dim,),initializer='TruncatedNormal',trainable=True)
            self.bias = self.add_weight(name=self.prefix+'bias', shape=(self.output_dim,),initializer='zeros',trainable=True)
            super(bim_fusion_layer, self).build(input_shape)  # Be sure to call this somewhere!

        def call(self, x):
            '''
            Forward propagation of the layer
            Args:
            x: list having the modalities

            Returns:
            output value of the layer 
            '''
            assert (K.int_shape(x[0])[1] <= dim_proj)
            return K.tanh(x[0]*self.wt1 + x[1]*self.wt2 + self.bias)

        def compute_output_shape(self, input_shape):
            '''
            Args:
            input_shape: tuple
            Returns:
            output shape after feed forward
            '''
            return (input_shape[0][0],self.output_dim)



    merged_train_data, train_label,train_mask = load_unimodal_activations_mosei3()
  

    x=Input(shape=(audio_dim+visual_dim+text_dim,), name='bi_input')
    masked = Masking(mask_value =0.0)(x)
    ia = Lambda(lambda x: x[:, 0:text_dim],output_shape=lambda x:(x[0], text_dim))(masked)
    ib =Lambda(lambda x: x[:, text_dim:(text_dim+visual_dim)],output_shape=lambda x:(x[0], visual_dim))(masked)
    ic =Lambda(lambda x: x[:, (text_dim+visual_dim):dim],output_shape=lambda x:(x[0], audio_dim))(masked)

    d1=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bid1')(ia)
    d2=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bid2')(ib)
    d3=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bid3')(ic)
    fus12=bim_fusion_layer('ha')([d1,d2])
    fus13=bim_fusion_layer('hb')([d1,d3])
    fus23=bim_fusion_layer('hc')([d2,d3])
    bim12=Model(x,outputs=fus12)
    bim13=Model(x,outputs=fus13)
    bim23=Model(x,outputs=fus23)



    input_data = Input(shape=(max_len, K.int_shape(fus12)[1],), name='biinp1')
    grubi1 = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bigru1')(input_data)
    tmp = Dropout(0.9)(grubi1)
    tmp = TimeDistributed(Dense(500,activation='tanh'))(tmp)
    o1 = TimeDistributed(Dense(NUM_LABELS,activation='softmax'))(tmp)
    lstm1 = Model(input_data, o1)
    aux1 = Model(input_data, tmp)
    input_data = Input(shape=(max_len, K.int_shape(fus12)[1],), name='biinp2')
    grubi2 = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bigru2')(input_data)
    tmp = Dropout(0.9)(grubi2)
    tmp = TimeDistributed(Dense(500,activation='tanh'))(tmp)
    o2 = TimeDistributed(Dense(NUM_LABELS,activation='softmax'))(tmp)
    lstm2 = Model(input_data, o2)
    aux2 = Model(input_data, tmp)
    input_data = Input(shape=(max_len, K.int_shape(fus12)[1],), name='biinp3')
    grubi3 = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bigru3')(input_data)
    tmp = Dropout(0.9)(grubi3)
    tmp = TimeDistributed(Dense(500,activation='tanh'))(tmp)
    o3 = TimeDistributed(Dense(NUM_LABELS,activation='softmax'))(tmp)
    lstm3 = Model(input_data, o3)
    aux3 = Model(input_data, tmp)



    main_input=Input(shape=(max_len,audio_dim+visual_dim+text_dim,), name='bim_inp')

    uttr12=TimeDistributed(bim12)(main_input)
    uttr13=TimeDistributed(bim13)(main_input)
    uttr23=TimeDistributed(bim23)(main_input)

    auxi12=aux1(uttr12)
    auxi13=aux2(uttr13)
    auxi23=aux3(uttr23)



    context_1_2=lstm1(uttr12)
    BiModal1 = Model(main_input, context_1_2)
    Predictor1 = Model(main_input, auxi12)
    context_1_3=lstm2(uttr13)
    BiModal2 = Model(main_input, context_1_3)
    Predictor2 = Model(main_input, auxi13)
    context_2_3=lstm3(uttr23)
    BiModal3 = Model(main_input, context_2_3)
    Predictor3 = Model(main_input, auxi23)



    optimizer=Adam(lr=0.001)
    BiModal1.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode='temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    BiModal1.fit(merged_train_data, train_label,epochs=200,batch_size=20,sample_weight=train_mask,shuffle=True, callbacks=[early_stopping],validation_split=0.1,verbose=1)
    train_result1 = Predictor1.predict(merged_train_data)
    BiModal2.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode='temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    BiModal2.fit(merged_train_data, train_label,epochs=200,batch_size=20,sample_weight=train_mask,shuffle=True, callbacks=[early_stopping],validation_split=0.1,verbose=1)
    train_result2 = Predictor2.predict(merged_train_data)
    BiModal3.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode='temporal')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    BiModal3.fit(merged_train_data, train_label,epochs=200,batch_size=20,sample_weight=train_mask,shuffle=True, callbacks=[early_stopping],validation_split=0.1,verbose=1)
    train_result3 = Predictor3.predict(merged_train_data)
    train_bim = np.concatenate((train_result1, train_result2, train_result3), axis=2)


    bim_dict={}
    bim_dict['train_bimodal'] = train_bim
    bim_dict['train_mask']=train_mask
    bim_dict['train_label']=train_label

    with open('bim_act.pickle', 'wb') as handle:
        pickle.dump(bim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def Trimodal():
    '''
    Trains the trimodal layers
    Returns:
    result: a 3D numpy array of predictions of model in one hot encoded form (test_size by sequence_length by number of classes)
    test_label: a 3D numpy array of true labels in one hot encoded form (test_size by sequence_length by number of classes)
    test_mask: a 2D numpy array telling which inputs to ignore while calculating accuracies
    '''
    merged_train_data, train_label,  train_mask, maxlen = load_bim_acts_mosei3()


    class trifusion_layer(Layer):
        '''
        trimodal fusion layer: weighted sum followed by tanh
        '''
        def __init__(self, **kwargs):
            '''
            Initialises the layer:
            Args:
            prefix: string that can be used to name internal weights
            kwargs: are the arguments to the superclass Layer
            '''
            self.supports_masking = True
            super(trifusion_layer, self).__init__(**kwargs)

        def build(self, input_shape):
            '''
            Takes the input shame and initalises the traianble used for the hadamard product
            Args:
            input_shape: numpy array as required by the build in superclass Layer
            '''
            self.output_dim = 500

            self.wt1 = self.add_weight(name='wt1', shape=(self.output_dim,),initializer='glorot_uniform',trainable=True)
            self.wt2 = self.add_weight(name='wt2', shape=(self.output_dim,),initializer='glorot_uniform',trainable=True)
            self.wt3 = self.add_weight(name='wt3', shape=(self.output_dim,),initializer='glorot_uniform',trainable=True)
            self.bias = self.add_weight(name='bias', shape=(self.output_dim,),initializer='zeros',trainable=True)
            super(trifusion_layer, self).build(input_shape)  # Be sure to call this somewhere!

        def call(self, x, mask=None):
            '''
            Forward propagation of the layer
            Args:
            x: list having the modalities

            Returns:
            output value of the layer 
            '''
            x1=Lambda(lambda x: x[:,0:500],output_shape=lambda x: (x[0],500))(x)
            x2=Lambda(lambda x: x[:,500:1000],output_shape=lambda x: (x[0],500))(x)
            x3=Lambda(lambda x: x[:,1000:1500],output_shape=lambda x: (x[0],500))(x)
            ai = K.tanh(x1*self.wt1 + x2*self.wt2 + x3*self.wt3 + self.bias)
            return ai

        def compute_output_shape(self, x):
            '''
            Args:
            input_shape: tuple
            Returns:
            output shape after feed forward
            '''
            return (x[0],self.output_dim)
        def compute_mask(self, input, input_mask=None):
            '''
            since masking is being done by mask layer we return list of None here
            Args:
            input to model and mask
            Returns:
            list of Nones
            '''
            if isinstance(input_mask, list):
                return [None] * len(input_mask)
            else:
                return None


    x_three=Input(shape=(1500,))
    allfused=trifusion_layer()(x_three)
    hfusion_model=Model(x_three,outputs=allfused)

    main_input= Input(shape=(maxlen,1500,), name='trimodal_inp')
    mask_vals = Masking(mask_value =0.0)(main_input)
    vidutt=TimeDistributed(hfusion_model)(mask_vals)
    grutri = GRU(TRIGRU_SZ, activation='tanh', return_sequences = True, dropout=0.35, name='tri_lstm')(vidutt)
    grutri = Masking(mask_value=0.)(grutri)
    tmp = Dropout(0.86)(grutri)
    concat=Concatenate(name="trimodal_conc")([vidutt,tmp])
    tmp2 = TimeDistributed(Dense(450,activation='relu'))(concat)
    tmp2 = Masking(mask_value=0.)(tmp2)
    cffull = Dropout(0.86)(tmp2)
    cffull = Masking(mask_value=0.)(cffull)
    preds = Dense(NUM_LABELS, activation='softmax')(cffull)




    model=Model(main_input,outputs=preds)

    optimizer=Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode='temporal',metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    merged_test_data=merged_train_data[:100]
    test_label=train_label[:100]
    test_mask=train_mask[:100]
    model.fit(merged_train_data[100:], train_label[100:],epochs=100,batch_size=20,sample_weight=train_mask[100:],shuffle=True,validation_data=(merged_test_data,test_label)) #validation_split=0.1
    result = model.predict(merged_test_data)
    test_res_maker(result, test_label, test_mask)  
    return result, test_label, test_mask



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

if __name__=='__main__':
    Bimodal()
    result, test_label, test_mask=Trimodal()
    y_true=[]
    y_pred=[]

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i,j]!=0:
                y_true.append(float(test_label[i,j] ))
                y_pred.append(np.argmax(result[i,j] ))
    classes=[1,2]
    class_names, report, support = get_report(y_true, y_pred, classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    scores = get_scores(y_true, y_pred, classes)
    plot_clf_report(class_names, report, support)
    plot_confusion_matrix(classes, cm)
    plot_tag_scores(classes, scores)

