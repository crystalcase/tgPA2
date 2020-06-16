import flask
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from utils import *


tf.disable_v2_behavior()
app = flask.Flask(__name__)

def init():
    global model, graph, session
    #model = Sequential()
    #model.add(Dense(5, activation="relu"))
    #model.add(Dense(10, activation="relu"))
    #model.add(Dense(1))
    #model.load_weights("F:\dev\TensorflowNetworks\model\TgBier2.h5")
    session = tf.Session()
    set_session(session)
    model = load_model("F:\dev\TensorflowNetworks\model\TgBierSum10.h5")
    graph = tf.get_default_graph()

def get_parameter():
    parameters = []
    parameters.append(float(flask.request.args.get("Kalendertag")))
    parameters.append(float(flask.request.args.get("Monat")))
    parameters.append(float(flask.request.args.get("Jahr")))
    parameters.append(float(flask.request.args.get("Kalenderwoche")))
    parameters.append(float(flask.request.args.get("Wochentag")))
    print("paras", parameters)
    return parameters


def sendResponse(raw_prediction):
    response = flask.jsonify(response = str(raw_prediction))
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers',
                         'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

@app.route("/predict", methods=["GET"])
def predict():
    parameters = get_parameter()
    input_feature = np.asarray(parameters).reshape(1, 5)
    print("input", input_feature)
    with graph.as_default():
        set_session(session)
        raw_prediction = model.predict(input_feature)
        print("raw", raw_prediction)
    return sendResponse(raw_prediction)


if __name__ == "__main__":
    session = tf.keras.backend.get_session()
    print("Loading")
    init()
    app.run(threaded=True)

