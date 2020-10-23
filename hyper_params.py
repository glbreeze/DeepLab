
from flags import FLAGS

NUM_CLASSES = 21  #
POWER = 0.9      #
MOMENTUM = 0.9   #

BATCH_NORM_DECAY = 0.9997  #

NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}


params={
    'output_stride': FLAGS.output_stride,
    'batch_size': FLAGS.batch_size,
    'base_architecture': FLAGS.base_architecture,
    'pre_trained_model': FLAGS.pre_trained_model,
    'batch_norm_decay': BATCH_NORM_DECAY,
    'num_classes': NUM_CLASSES,
    'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
    'weight_decay': FLAGS.weight_decay,
    'learning_rate_policy': FLAGS.learning_rate_policy,
    'num_train': NUM_IMAGES['train'],
    'initial_learning_rate': FLAGS.initial_learning_rate,
    'max_iter': FLAGS.max_iter,
    'end_learning_rate': FLAGS.end_learning_rate,
    'power': POWER,
    'momentum': MOMENTUM,
    'freeze_batch_norm': FLAGS.freeze_batch_norm,
    'initial_global_step': FLAGS.initial_global_step
	}