from plant import *
from flask import Flask, Blueprint, render_template, request, jsonify
from swagger_ui_bundle import swagger_ui_path
from flask_cors import cross_origin
from flask_swagger import swagger

import os
application = Flask(__name__, static_url_path='', template_folder=".")
api = Blueprint('dcmotor', __name__)
swaggerapi = Blueprint('swagger_ui', __name__, static_url_path='',
                       static_folder=swagger_ui_path, template_folder=swagger_ui_path)

SWAGGER_UI_CONFIG = {
    "openapi_spec_url": "/spec"
}


@swaggerapi.route('/')
def swagger_ui_index():
    return render_template('index.j2', **SWAGGER_UI_CONFIG)


@application.route("/spec")
def spec():
    swag = swagger(application)
    swag['info']['version'] = "0.1"
    swag['info']['title'] = "DC Motor API"
    return jsonify(swag)


@api.route('/stop', methods=['GET'])
@cross_origin()
def close():
    """
    Stop the system.
    ---
    responses:
        200:
            description: Valid response from server (contains payload with actual status).
        404:
            description: Not Found.
        405:
            description: Method not allowed.
    """
    return exit(0)


@api.route('/<int:idx>/', methods=['GET'])
@api.route('/<int:idx>/index.html', methods=['GET'])
@cross_origin(idx=0)
def index(idx=0):
    """
    Provides access to a view of the system.
    ---
    parameters:
      - name: idx
        in: path
        type: string
        required: true
        description: The id of the relevant sub-system
    responses:
        200:
            description: Valid response from server (contains payload with actual status).
        404:
            description: Not Found.
        405:
            description: Method not allowed.
    """
    return render_template('index2.html', idx=idx)


@api.route('/<int:idx>/init', methods=['GET'])
@cross_origin()
def initservice(idx=0):
    """
    Initializes the state of the system
    ---
    parameters:
      - name: idx
        in: path
        type: string
        required: true
        description: The id of the relevant sub-system
      - in: query
        name: alphadot
        type: string
        description: The angular velocity of the motor
    responses:
        200:
            description: Valid response from server (contains payload with actual status).
        404:
            description: Not Found.
        405:
            description: Method not allowed.
    """
    f = init(request.full_path, idx)
    return f


@api.route('/<int:idx>/state', methods=['GET'])
@cross_origin()
def getstate(idx=None):
    """
    Returns the state of the system
    ---
    parameters:
      - name: idx
        in: path
        type: string
        required: true
        description: The id of the relevant sub-system
    responses:
        200:
            description: Valid response from server (contains payload with actual status).
        404:
            description: Not Found.
        405:
            description: Method not allowed.
    """
    f = state(idx)
    return f


@api.route('/<int:idx>/u', methods=['GET'])
@cross_origin()
def model(idx=0):
    """
    Controls the state of the system
    ---
    parameters:
      - name: idx
        in: path
        type: string
        required: true
        description: The id of the relevant sub-system
      - in: query
        name: value0
        type: string
        description: The first value
      - in: query
        name: time
        type: number
        description: The time in milliseconds the control value was assigned
    responses:
        200:
            description: Valid response from server (contains payload with actual status).
        404:
            description: Not Found.
        405:
            description: Method not allowed.
    """

    f = interpret(request.full_path, idx)
    return f


application.register_blueprint(api)
application.register_blueprint(swaggerapi, url_prefix="/ui")

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    application.run(host='0.0.0.0', port=port)
