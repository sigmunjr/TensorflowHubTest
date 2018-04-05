import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

import cv2


images = tf.placeholder(tf.float32, [None, None, None, 3])
labels = tf.placeholder(tf.int64, [None])

module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/1")

height, width = hub.get_expected_image_size(module)
features = module(images)
logits = tf.layers.dense(features, 10)
loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, 10), logits)

var_list = [v for v in tf.global_variables() if v.name.startswith('dense')]
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss, var_list=var_list)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

recording = False
avg_val = 0
cnt = 1
train_label = 0

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    cv2.imshow("img", img)
    key = cv2.waitKey(30)
    img = cv2.resize(img, (height, width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    logits_val = sess.run(logits, {images: img[np.newaxis]})

    if key == ord(' '):
        if recording:
            recording = False
        else:
            recording = True
            cnt = 0
    elif key >= ord('0') and key <= ord('9'):
       train_label = int(chr(key))

    if recording:
        sess.run(train_op, {images: img[np.newaxis], labels: [train_label]})
        print('RECORDING - CURRENT TRAINING CLASS:', train_label)

    print('DIST:', np.argmax(logits_val))
