import tensorflow as tf

bool_random_flip_left_right = 1
bool_random_flip_up_down = 1
bool_random_brightness = 1
bool_random_contrast = 1
bool_random_hue = 1
bool_random_saturation = 1
bool_random_crop = 1
bool_rotation_transform = 1
cutmix_rate = 0.
mixup_rate = 0.
gridmask_rate = 0.5

pre_trained = 'imagenet'  # None,'imagenet','noisy-student'
dense_activation = 'softmax'  # 'softmax','sigmoid'
bool_lr_scheduler = 1

# 交叉验证和tta（测试时数据增强）目前最多只允许使用一个
tta_times = 0  # 15 #当tta_times=i>0时，使用i+1倍测试集
cross_validation_folds = 5  # 当tta_times=i>1时，使用i折交叉验证

# focal_loss和label_smoothing最多同时使用一个
bool_focal_loss = 1
label_smoothing_rate = 0.

bool_tta = tta_times and max(bool_random_flip_left_right,
                             bool_random_flip_up_down,
                             bool_random_brightness,
                             bool_random_contrast,
                             bool_random_hue,
                             bool_random_saturation)

# 伪标签
bool_pseudo = 0

# 增加auc记录
special_monitor = 'auc'

AUTO = tf.data.experimental.AUTOTUNE
strategy = tf.distribute.get_strategy()

# 超参数，根据数据和策略调参
BATCH_SIZE = 1 * strategy.num_replicas_in_sync
img_size = 220
EPOCHS = 40
lr_if_without_scheduler = 0.0003
nb_classes = 17
print('BATCH_SIZE是：', BATCH_SIZE)

my_metrics = ['accuracy']

# lr_scheduler
# 学习率 函数
# 数值按实际情况设置

# LR_START = 0.00001
# LR_MAX = 0.0001 * strategy.num_replicas_in_sync
# LR_MIN = 0.00001
# LR_RAMPUP_EPOCHS = 3
# LR_SUSTAIN_EPOCHS = 5
# LR_EXP_DECAY = .8

LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 3
LR_SUSTAIN_EPOCHS = 5
LR_EXP_DECAY = .8


# 针对前面opts的一些中间处理
def check_config():
    # tta只有在使用了data_aug时才允许启用
    bool_tta = tta_times and max(bool_random_flip_left_right,
                                 bool_random_flip_up_down,
                                 bool_random_brightness,
                                 bool_random_contrast,
                                 bool_random_hue,
                                 bool_random_saturation)

    print(bool_tta)

    assert (bool_focal_loss and label_smoothing_rate) == 0, 'focal_loss和label_smoothing最多同时使用一个'
    assert (tta_times and cross_validation_folds) == 0, 'focal_loss和label_smoothing最多同时使用一个'
