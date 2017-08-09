import cifar10_fullprocess as dcnn
import numpy as np
import tensorflow as tf
import time as time
import variableextraction as variables

from PIL import Image

# File paths
image_path = 'C:\\Users\\emman\\Desktop\\tensorflow\\TensorFlowClient\\images\\test8.jpg'

def send_processed_image(image=None):
    # Extract and expand image for dcnn
    if image is None:
        img = Image.open(image_path)
        image_data = np.asarray(img, dtype='float32')
    else:
        image_data = image

    expand_img = tf.expand_dims(image_data, 0)

    # Initialize graph with expanded image and extract weights and bias
    # from .meta file
    preprocessed_image = dcnn.initializeGraph(expand_img)
    variables.getVariables()
    init_op = tf.global_variables_initializer()

    start_time = time.time()

    # Run partial cifar-10 neural network
    with tf.Session() as sess:
        sess.run(init_op)
        logits = sess.run(preprocessed_image)

    # Calculate predictions and return results
    images, labels = dcnn.inputs(eval_data=True)
    prediction = tf.nn.in_top_k(logits, labels, 1)
    print("Prediction: ")
    print(prediction)

    duration = time.time() - start_time
    print('Compute Time: ' + str(duration))

    return prediction

#send_processed_image()