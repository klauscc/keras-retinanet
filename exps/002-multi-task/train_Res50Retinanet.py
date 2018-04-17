"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os

import keras
import keras.preprocessing.image

import keras_retinanet.losses
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.utils.keras_version import check_keras_version
# from preprocess.nsp_pascal_voc_generator import PascalVocGenerator
from keras_retinanet.preprocessing.nsp_generator import PascalVocGenerator
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.bin.train import create_models

import tensorflow as tf

CURRENT_FILE_PATH = os.path.abspath(
    os.path.dirname(os.path.realpath(__file__)))

TASK_NAME = "002-multi-task"


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_callbacks(model, training_model, prediction_model, val_generator):
    """TODO: Docstring for create_callbacks.

    Args:
        model (TODO): TODO
        training_model (TODO): TODO
        prediction_model (TODO): TODO
        val_generator (TODO): TODO

    Returns: TODO

    """
    from keras_retinanet.callbacks import RedirectModel
    callbacks = []
    snapshot_path = os.path.join(CURRENT_FILE_PATH, "../../data/snapshots/",
                                 TASK_NAME)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(snapshot_path, "resnet50_nsp_{epoch:02d}.h5"), verbose=1)
    checkpoint = RedirectModel(checkpoint, prediction_model)
    callbacks.append(checkpoint)

    callback = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0)
    callbacks.append(callback)

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=os.path.join(CURRENT_FILE_PATH, "../../data/tensorboard",
                             TASK_NAME),
        histogram_freq=0,
        batch_size=args.batch_size,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None)
    callbacks.append(tensorboard_callback)

    from keras_retinanet.callbacks.eval import Evaluate
    evaluation = Evaluate(val_generator, tensorboard=tensorboard_callback)
    evaluation = RedirectModel(evaluation, prediction_model)
    callbacks.append(evaluation)

    return callbacks


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple training script for Pascal VOC object detection.')
    parser.add_argument(
        'voc_path',
        help='Path to Pascal VOC directory (ie. /tmp/VOCdevkit/VOC2007).')
    parser.add_argument(
        '--weights',
        help='Weights to use for initialization (defaults to ImageNet).',
        default='imagenet')
    parser.add_argument(
        '--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument(
        '--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create image data generator objects
    transform_generator = random_transform_generator(flip_x_chance=0.5)

    # create a generator for training data
    train_generator = PascalVocGenerator(
        args.voc_path,
        'trainval',
        # group_method = "random",
        transform_generator=transform_generator,
        batch_size=args.batch_size)
    # create a generator for testing data
    val_generator = PascalVocGenerator(
        args.voc_path,
        'test',
        # group_method = "random",
        batch_size=args.batch_size)

    # create the model
    print('Creating model, this may take a second...')
    from keras_retinanet.models.resnet import resnet_retinanet as retinanet, custom_objects, download_imagenet
    backbone = "resnet50"
    weights = download_imagenet(backbone)
    model = retinanet(
        num_classes=train_generator.num_classes(),
        backbone=backbone,
        nms=True,
        modifier=None,
        global_cls=True)
    model.load_weights(weights, by_name=True) 
    training_model = model
    prediction_model = model

    # compile model (note: set loss to None since loss is added inside layer)
    training_model.compile(
        loss={
            'regression': keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal(),
            'global_cls': 'categorical_crossentropy'
        },
        metrics = {'global_cls': 'accuracy'},
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

    # print model summary
    print(training_model.summary())

    # callbacks
    callbacks = create_callbacks(model, training_model, prediction_model,
                                 val_generator)

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=len(train_generator.image_names) // args.batch_size,
        validation_data=val_generator,
        validation_steps=len(val_generator.image_names) // args.batch_size,
        epochs=50,
        verbose=1,
        callbacks=callbacks)
