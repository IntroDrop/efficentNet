import numpy as np
import tensorflow as tf
import random, re, math
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow_addons as tfa
from matplotlib import pyplot as plt

from ImgUtil import decode_image, data_aug, rotation_transform, cutmix, mixup, gridmask

from cfg import cross_validation_folds, special_monitor, my_metrics, AUTO, bool_rotation_transform, BATCH_SIZE, \
    cutmix_rate, mixup_rate, gridmask_rate, bool_tta, strategy, pre_trained, img_size, nb_classes, dense_activation, \
    label_smoothing_rate, bool_focal_loss, lr_if_without_scheduler, EPOCHS, tta_times

import efficientnet.tfkeras as efn
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ModelCheckpoint

if special_monitor == 'auc':
    my_metrics.append(tf.keras.metrics.AUC(name='auc'))


# 生成训练集
def get_train_dataset(train_paths, train_labels=None):
    # num_parallel_calls并发处理数据的并发数
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_labels.astype(np.float32))).map(decode_image, num_parallel_calls=AUTO)

    # train_dataset = train_dataset.cache().map(data_aug, num_parallel_calls=AUTO).repeat()
    train_dataset = train_dataset.map(data_aug, num_parallel_calls=AUTO).repeat()

    if bool_rotation_transform:
        train_dataset = train_dataset.map(rotation_transform)

    train_dataset = train_dataset.shuffle(512).batch(BATCH_SIZE, drop_remainder=True)

    if cutmix_rate:
        print('启用cutmix')
        train_dataset = train_dataset.map(cutmix, num_parallel_calls=AUTO)
    if mixup_rate:
        print('启用mixup')
        train_dataset = train_dataset.map(mixup, num_parallel_calls=AUTO)
    if gridmask_rate:
        print('启用gridmask')
        train_dataset = train_dataset.map(gridmask, num_parallel_calls=AUTO)
    if (cutmix_rate or mixup_rate):
        train_dataset = train_dataset.unbatch().shuffle(512).batch(BATCH_SIZE)

    # prefetch: prefetch next batch while training (autotune prefetch buffer size)
    train_dataset = train_dataset.prefetch(AUTO)

    return train_dataset


# 生成验证集
def get_validation_dataset(valid_paths, valid_labels=None):
    dataset = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    dataset = dataset.map(decode_image, num_parallel_calls=AUTO)

    # dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


# 生成测试集
def re_produce_test_dataset(test_paths):
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(decode_image, num_parallel_calls=AUTO)

    if bool_tta:
        test_dataset = test_dataset.cache().map(data_aug, num_parallel_calls=AUTO)

    test_dataset = test_dataset.batch(BATCH_SIZE)
    return test_dataset


# 创建模型
def get_model():
    with strategy.scope():
        base_model = efn.EfficientNetB7(weights=pre_trained, include_top=False, pooling='avg',
                                        input_shape=(img_size, img_size, 3))
        x = base_model.output
        predictions = Dense(nb_classes, activation=dense_activation)(x)
        model = Model(inputs=base_model.input, outputs=predictions)

    if label_smoothing_rate:
        print('启用label_smoothing')
        my_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_rate)
    elif bool_focal_loss:

        my_loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)

    else:
        my_loss = 'categorical_crossentropy'

    model.compile(optimizer=tf.keras.optimizers.Adam(lr_if_without_scheduler),
                  loss=my_loss,
                  metrics=my_metrics)

    return model



def training(train_paths, train_labels, callbacks):
    probabilities = []
    global model, ch_p1
    if cross_validation_folds:

        histories = []

        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)
        kfold = KFold(cross_validation_folds, shuffle=True)

        i = 1

        for trn_ind, val_ind in kfold.split(train_paths, train_labels):
            print('#' * 25)
            print('### FOLD', i)
            print('#' * 25)

            # print(trn_ind)
            print(val_ind)

            # 暂停checkpoint，防止爆内存
            # 每轮都应该重置 ModelCheckpoint
            # ch_p1 = ModelCheckpoint(filepath="temp_best.h5", monitor='val_accuracy', save_weights_only=True,verbose=1,save_best_only=True)

            if special_monitor == 'auc':
                ch_p1 = ModelCheckpoint(filepath="temp_best.h5", monitor='val_auc', mode='max', save_weights_only=True,
                                        verbose=1, save_best_only=True)

            temp_callbacks = callbacks.copy()
            temp_callbacks.append(ch_p1)

            trn_paths = np.array(train_paths)[trn_ind]
            val_paths = np.array(train_paths)[val_ind]

            trn_labels = train_labels[trn_ind]
            val_labels = train_labels[val_ind]
            test_paths = val_paths

            model = get_model()
            history = model.fit(
                get_train_dataset(trn_paths, trn_labels),
                steps_per_epoch=trn_labels.shape[0] // BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=temp_callbacks,
                validation_data=(get_validation_dataset(val_paths, val_labels)),
            )

            i += 1
            histories.append(history)

            # 用val_loss最小的权重来预测
            model.load_weights("temp_best.h5")
            prob = model.predict(re_produce_test_dataset(test_paths), verbose=1)

            probabilities.append(prob)
            probabilities.append(val_labels)
            break
    # if not cross_validation_folds:
    else:

        model = get_model()

        histories = model.fit(
            get_train_dataset(train_paths, train_labels),
            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
            callbacks=callbacks,
            epochs=EPOCHS
        )

    return model, histories, probabilities


# history
def display_training_curves(training, title, subplot, validation=None):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    if validation is not None:
        ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    # ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    if validation is not None:
        ax.legend(['train', 'valid.'])
    else:
        ax.legend(['train'])


def draw_training_curves(histories):
    # 写开，防止交叉验证时out of memory导致什么都没保存
    if cross_validation_folds:
        # 然后画画- -
        for h in range(len(histories)):
            display_training_curves(histories[h].history['loss'], 'loss', 211, histories[h].history['val_loss'])
            display_training_curves(histories[h].history['accuracy'], 'accuracy', 212,
                                    histories[h].history['val_accuracy'])

    # if not cross_validation_folds:
    else:
        display_training_curves(histories.history['loss'], 'loss', 211)
        display_training_curves(histories.history['accuracy'], 'accuracy', 212)


def predict(model, probabilities):
    global y_pred
    if cross_validation_folds:
        y_pred = np.mean(probabilities, axis=0)
    # if not cross_validation_folds:
    else:
        if bool_tta:
            probabilities = []
            for i in range(tta_times + 1):
                print('TTA Number: ', i, '\n')
                test_dataset = re_produce_test_dataset(test_paths)
                probabilities.append(model.predict(test_dataset))
                y_pred = np.mean(probabilities, axis=0)


        else:
            test_dataset = re_produce_test_dataset(test_paths)
            y_pred = model.predict(test_dataset)

    return y_pred