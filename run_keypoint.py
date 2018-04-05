import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
_DISTANCE_THRESHOLD = 0.8

# Prepare an image tensor.
image = tf.placeholder(tf.float32, [None, None, 3], 'image_tensor')  # tf.image.decode_jpeg('my_image.jpg', channels=3)

# Instantiate the DELF module.
delf_module = hub.Module("https://tfhub.dev/google/delf/1")

delf_inputs = {
    # An image tensor with dtype float32 and shape [height, width, 3], where
    # height and width are positive integers:
    'image': image,
    # Scaling factors for building the image pyramid as described in the paper:
    'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
    # Image features whose attention score exceeds this threshold will be
    # returned:
    'score_threshold': 100.0,
    # The maximum number of features that should be returned:
    'max_feature_num': 1000,
}

# Apply the DELF module to the inputs to get the outputs.
delf_outputs = delf_module(delf_inputs, as_dict=True)

cap = cv2.VideoCapture(0)
loc2, desc2, img2 = None, None, None
with tf.Session() as sess:
    while True:
        _, img1 = cap.read()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        a = sess.run(delf_outputs, {image: img1.astype(np.float32)})
        loc1, desc1 = a['locations'], a['descriptors']

        if loc2 is not None and desc2 is not None:
            d1_tree = cKDTree(desc1)
            _, indices = d1_tree.query(
                desc2, distance_upper_bound=_DISTANCE_THRESHOLD)

            # Select feature locations for putative matches.
            num_features_1 = loc1.shape[0]
            num_features_2 = loc2.shape[0]
            locations_2_to_use = np.array([
                loc2[i,]
                for i in range(num_features_2)
                if indices[i] != num_features_1
            ])
            locations_1_to_use = np.array([
                loc1[indices[i],]
                for i in range(num_features_2)
                if indices[i] != num_features_1
            ])

            # Perform geometric verification using RANSAC.
            _, inliers = ransac(
                (locations_1_to_use, locations_2_to_use),
                AffineTransform,
                min_samples=3,
                residual_threshold=20,
                max_trials=1000)

            tf.logging.info('Found %d inliers' % sum(inliers))

            # Visualize correspondences, and save to file.
            _, ax = plt.subplots()
            inlier_idxs = np.nonzero(inliers)[0]
            plt.figure(1)
            plot_matches(
                ax,
                img1,
                img2,
                locations_1_to_use,
                locations_2_to_use,
                np.column_stack((inlier_idxs, inlier_idxs)),
                matches_color='b')
            plt.show()

        img2, loc2, desc2 = img1, loc1, desc1
