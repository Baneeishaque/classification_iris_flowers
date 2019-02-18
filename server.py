#!/usr/bin/python3
from flask import Flask
from flask_restful import Resource, Api

import single_class_prediction_joblib_dataset_api

app = Flask(__name__)
api = Api(app)


class IrisFlowerPrediction(Resource):
    def get(self, sepal_length, sepal_width, petal_length, petal_width):
        # return sepal_length
        return single_class_prediction_joblib_dataset_api.get_prediction(
            [[sepal_length, sepal_width, petal_length, petal_width]])


api.add_resource(IrisFlowerPrediction,
                 '/iris_flower_prediction/<sepal_length>/<sepal_width>/<petal_length>/<petal_width>')

if __name__ == '__main__':
    app.run()
