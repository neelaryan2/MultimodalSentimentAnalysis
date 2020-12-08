import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

def re_label(y):
    labels = sorted(list(set(list(y))))
    inv = {v:i for i,v in enumerate(labels)}
    return np.array([inv[v] for v in y], dtype=np.int32)

def remove_labels(X, y, labels_to_remove):
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

def shuffle(X, Y):
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    return X[idx], Y[idx]

def add_ones(X):
    ones = np.ones((X.shape[0], 1))
    return np.append(X, ones, axis=1)

def get_batches(X, Y, batch=32):
    N = X.shape[0]
    X_batch, Y_batch = [], []
    Q, R = N // batch, N % batch
    for i in range(Q):
        X_batch.append(X[i*batch:(i+1)*batch, :])
        Y_batch.append(Y[i*batch:(i+1)*batch])
    if R != 0 :
        X_batch.append(X[Q*batch:, :])
        Y_batch.append(Y[Q*batch:])
    return X_batch, Y_batch

def split_util(X, train_ratio=0.8):
    slic = int(X.shape[0] * min(max(train_ratio, 0.0), 1.0))
    X_train, X_test = X[:slic], X[slic:]
    return X_train, X_test

def split_data(X, Y, train_ratio=0.8):
    X_train, X_test = split_util(X, train_ratio)
    Y_train, Y_test = split_util(Y, train_ratio)
    return X_train, Y_train, X_test, Y_test

def one_hot_encode(X, labels):
    X.shape = (X.shape[0], 1)
    newX = np.zeros((X.shape[0], len(labels)))
    label_encoding = {}
    for i, l in enumerate(labels):
        label_encoding[l] = i
    for i in range(X.shape[0]):
        newX[i, label_encoding[X[i, 0]]] = 1
    return newX

def normalize(X):
    return (X - np.mean(X)) / np.std(X)

def untag(tagged_sent):
    return [word for word, tag in tagged_sent]

def preprocess(X, Y):
    N, D = X.shape
    X_new = [np.ones((N, ), dtype=np.float64)]
    for i in range(1, D):
        col = X[:, i]
        try:
            col = col.astype(np.float64)
            new_cols = normalize(col)
            X_new.append(new_cols)
        except ValueError:
            labels = sorted(list(set(col)))
            new_cols = one_hot_encode(col, labels)
            for j in range(new_cols.shape[1]):
                X_new.append(new_cols[:, j])
    X_new = np.array(X_new).T
    return X_new, Y

def visualize(X, Y, W=None):
    fig = plt.figure()
    for i in range(X.shape[0]):
        if Y[i][0] > 0 :
            plt.scatter(X[i][0], X[i][1], marker='+', color='red')
        else :
            plt.scatter(X[i][0], X[i][1], marker='_', color='blue')
    if W is None : return
    xmin, xmax = min(X[:, 0]), max(X[:, 0])
    linex = np.linspace(xmin, xmax, num=50)
    liney = [-(W[2] + W[0] * e) / W[1] for e in linex]
    plt.plot(linex, liney)

def min_max_scale(x):
    mn = np.min(x, axis=0)
    r = np.ptp(x, axis=0)
    mask = (r == 0)
    r[mask] = mn[mask]
    return (x - mn) / r

def round_label(a, limit=1):
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
    k = len(classes)
    cm = np.copy(mat)
    title = 'Confusion Matrix (without normalization)'
    if normalize:
        cm = cm.astype('float') / np.sum(cm, axis=1, keepdims=True)
        title = title.replace('without', 'with')
    plt.clf()    
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
    
def plot_clf_report(classes, plotMat, support, cmap=plt.cm.Blues, filename='classification_report'):
    k = len(classes)
    title = 'Classification Report'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(classes[idx], sup) for idx, sup in enumerate(support)]
    plt.clf()
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
    
def plot_tag_scores(classes, scores, normalize=True, filename='tag_scores'):
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
    plt.savefig(filename+'.png', bbox_inches="tight", transparent=True)
    plt.show()