import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from tensorflow.keras.models import load_model
import h5py
from keras import __version__ as keras_version

#additions

import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        # image_array = np.asarray(image)

        img = image.crop((0, 80, 320, 160))  # crop image (remove above horizon). Original dimensions: 640,480. changing to 640,240
        img = img.resize((224, 224))  # resize image --> pretrained alexnet model needs img sizes of 224 x 224
        transform = transforms.ToTensor()
        img = transform(img)
        img.unsqueeze_(0)


        # steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        steering_angle = float(resnet(img))
        # throttle = float(throttle_model(img)[0][1])

        throttle = .1 #controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    # f = h5py.File(args.model, mode='r')
    # model_version = f.attrs.get('keras_version')
    # keras_version = str(keras_version).encode('utf8')
    #
    # if model_version != keras_version:
    #     print('You are using Keras version ', keras_version,
    #           ', but the model was built using ', model_version)

    # model = load_model(args.model)

    path_to_model = r'C:\Users\Akhil\Desktop\Steering Dataset\Simulation4\CarND-Behavioral-Cloning-P3-master\model_sim_steering_IMG3_trained.pt'
    path_to_throttle = r'C:\Users\Akhil\Desktop\Steering Dataset\Simulation4\CarND-Behavioral-Cloning-P3-master\model_sim_throttle.pt'


    # Load steering model

    resnet = models.resnet18(pretrained=False, progress=True)
    for param in resnet.parameters():
        param.requires_grad = True
    resnet.fc = nn.Sequential(nn.Linear(512, 512),
                              nn.ReLU(),
                              nn.Linear(512, 1))

    resnet.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    resnet.eval()


    # Load throttle model

    throttle_model = models.resnet18(pretrained=False, progress=True)
    for param in throttle_model.parameters():
        param.requires_grad = True
    throttle_model.fc = nn.Sequential(nn.Linear(512, 512),
                              nn.ReLU(),
                              nn.Linear(512, 2))

    throttle_model.load_state_dict(torch.load(path_to_throttle, map_location=torch.device('cpu')))
    throttle_model.eval()



    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
