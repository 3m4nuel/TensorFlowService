from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inceptionv3 as inception

# Image path
test_image_path = "C:\\Users\\emman\\PycharmProjects\\TensorWebApi\\models\\inception\\test4.jpg"


class Classifier:

    def classify(self, tensor=None):
        # Load the Inception model so it is ready for classifying images.
        model = inception.Inception()

        # Use the Inception model to classify the image.
        pred = model.classify(image_path=test_image_path, tensor=tensor)

        # Print the scores and names for the top-10 predictions.
        model.print_scores(probabilities=pred)

        # Write summary for TensorBoard
        model._write_summary(logdir='/tmp/tensorflow_logs/example')

        # New Graph yo
        # model.create_graph()

        # Close the TensorFlow session.
        model.close()
