# 导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from cfg import nb_classes, EPOCHS, bool_lr_scheduler, bool_pseudo
from lr import lrfn
from model import training, draw_training_curves, predict

print(tf.__version__)
print(tf.keras.__version__)

import glob

train_paths = []
train_labels = []
test_paths = []
test_labels = []

paths = os.listdir('../input/longterm/longterm/')

for i in range(len(paths)):
    paths[i] = '../input/longterm/longterm/' + paths[i]
    print(paths[i])
    # 类似 ../input/longterm/longterm/A.clavatus-long

print(len(paths))

assert len(paths) == nb_classes

for i in range(nb_classes):
    for dirpath, dirnames, filenames in os.walk(paths[i]):
        # print(dirpath)
        # print(dirnames)
        # print(filenames)
        for filename in filenames:
            # print(os.path.join(dirpath,filename))
            # ../input/longterm/longterm/A.clavatus-long/A.clavatus-long/A.clavatus7d11.JPG
            # print(np.eye(nb_classes)[i])
            # [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            train_paths.append(os.path.join(dirpath, filename))
            train_labels.append(np.eye(nb_classes)[i])

train_labels = np.array(train_labels)

# 随便看一张
img = plt.imread(train_paths[0])
print('\n', img.shape)
plt.imshow(img)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
if bool_lr_scheduler:
    plt.plot(rng, y)
    print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

callbacks = []
if bool_lr_scheduler:
    callbacks.append(lr_callback)

model, histories, probabilities = training(train_paths, train_labels, callbacks)
draw_training_curves(histories)
y_pred = predict(model, probabilities)

pred = (np.array([np.argmax(ele) for ele in probabilities[0]])).astype(int)
val_labels = (np.array([np.argmax(ele) for ele in probabilities[1]])).astype(int)
print(pred)
print(val_labels)

#############画图部分
fpr, tpr, threshold = metrics.roc_curve(val_labels, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


def sub(name):
    ret=[]
    for i in y_pred:
        if i[0]>=i[1]:
            ret.append('AD')
        elif i[0]<i[1]:
            ret.append('CN')

    sub = pd.read_csv('../input/petpet/pet.csv')
    sub.loc[:, 'label'] = ret

    print(len(sub))

    sub.to_csv(name, index=False)
    sub.head(30)

sub('submission.csv')

#保留副本，防止错误操作
train_paths_copy=train_paths.copy()
train_labels_copy=train_labels.copy()

if bool_pseudo:
    print("启用伪标签")

    train_paths = train_paths_copy.copy()
    train_labels = train_labels_copy.copy()

    print(len(train_paths), len(train_labels))
    for i in range(len(y_pred)):
        print(y_pred[i])

        if y_pred[i][0] >= 0.9:
            train_paths = np.append(train_paths, test_paths[i])
            train_labels = np.append(train_labels, [[1, 0]], axis=0)
        elif y_pred[i][0] <= 0.1:
            train_paths = np.append(train_paths, test_paths[i])
            train_labels = np.append(train_labels, [[0, 1]], axis=0)
    print(len(train_paths), len(train_labels))

else:
    for i in range(len(y_pred)):
        print(y_pred[i])

if bool_pseudo:
    model,histories,probabilities=training()
    draw_training_curves(histories)
    y_pred=predict(model,probabilities)
    sub('submission_pseudo.csv')