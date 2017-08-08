from flask import request, jsonify, Flask
import json
import classifier
import numpy as np
import tensorflow as tf


app = Flask(__name__)

def create_app():

    @app.route('/tensorflow/', methods=['PUT'])
    def postTensor():
        #json_data = json.loads(str(request.data, encoding='utf-8'))
      ##  nparray_data = np.array(json_data)
        ##tensor_data = tf.convert_to_tensor(nparray_data)
        ##classify = classifier.Classifier
        ##classifyResults = classify.classify(tensor=tensor_data)
        ##print(classifyResults)
        ##response = jsonify({'results': classifyResults})
        #print(json_data)
        print(request.files['image'])
        images.save(request.files['image'])
        response = jsonify({'hello': 'hello'})
        response.status_code = 201
        return response

    return app