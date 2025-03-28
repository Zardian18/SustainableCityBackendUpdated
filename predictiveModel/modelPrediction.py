'''
History:
createdBy       date        description
ZARD            12/11/24    created flask file for model prediction
ZARD            15/11/24    updated flask code for making prediction
'''

from flask import Flask, request, jsonify

from flask_cors import CORS
import numpy as np
import tensorflow as tf


app = Flask(__name__)
CORS(app)

path = "D:\\MASTERS_ALL_MATERIAL\\TRINITY_COLLEGE_DUBLIN\\ASE_Backend\\SustainableCityManagement_Backend\\app\\saved_models\\saved_model_test.h5"
model = tf.keras.models.load_model(
    path,
    compile=True,
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()},
)


@app.route('/prediction', methods=['GET'])
def predict():
    try:
        # data = request.get_json()
        # input_data = data.get("input")
        # if len(input_data) != 7:
        #     return jsonify({"error": "Expected 7 input values."}), 400
        # input_array = np.array(input_data).reshape(1, -1)
        input_array = np.array([[0.6, 0.7, 0.4, 0.3, 0.7, 0.9, 0.2]])
        prediction = model.predict(input_array)
        return jsonify({"prediction": float(prediction[0][0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # print("-------------------------------------------------------------------")
    print(f"path is: {path}")
    app.run(debug=True)
