##################################
# front : vue
# back  : flask ← here
# ml    : scikit-learn ← here
# db    : mongodb ← here
##################################

from flask import Flask, jsonify, make_response, request, Response
from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app)



@app.route('/')
def hello_world():
    res = "<h1>making pytorch web platform</h1>"
    res += "<ul><li>input data</li><li>select model</li><li>select hyper parameter</li><li>result</li></ul>"
    return res

@app.route('/get_exp_data/')
def get_exp_data():
    # print(request)
    # print(request)
    # print(request)
    print(exp_data)
    return jsonify(exp_data)

###########################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time, os
import copy
import random as rd

seed = 11
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(seed)
rd.seed(seed)

print(device)

batch_size = 64

data_transforms = {
    'train': transforms.Compose([
        transforms.CenterCrop((280,400) ),
        transforms.Resize( (224,224) ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.CenterCrop((280,400) ),
        transforms.Resize( (224,224) ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


import glob
train_dir = "/home/user/data/bearing/data_first/train/"
train_ok = glob.glob( train_dir + "/0_ok/*.jpg" )
train_ng = glob.glob( train_dir + "/1_ng/*.jpg" )
test_dir = "/home/user/data/bearing/data_first/test/"
test_ok = glob.glob( test_dir + "/0_ok/*.jpg" )
test_ng = glob.glob( test_dir + "/1_ng/*.jpg" )


exp_data = { "train_ok": len(train_ok),
             "train_ng": len(train_ng),
             "test_ok" : len(test_ok),
             "test_ng" : len(test_ng) }


app.run()
