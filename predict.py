import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.utils.data
from collections import OrderedDict
import Pandas as pd
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser (description = "Prediction script")

parser.add_argument ('image_dir', help = 'Image Path.', type = str)
parser.add_argument ('load_dir', help = 'Provide path to checkpoint.', type = str)
parser.add_argument ('top_k', help = 'Top K most likely classes.', type = int, default = 5)
parser.add_argument ('category_names', help = 'Categories can be mapped to real names.', type = str, deafault = 'cat_to_name.json')
parser.add_argument ('GPU', help = "GPU or CPU.", type = str)

def loading_model (file_path):
    checkpoint = torch.load (file_path) 
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: 
        model = models.vgg13 (pretrained = True)
    model.classifier = checkpoint ['classifier']
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['mapping']

    for param in model.parameters():
        param.requires_grad = False 

    return model

def process_image(image):
    im = Image.open (image) 
    w = image.size[0]
    h = image.size[1]
    if w > h:
        h = 256
        im.thumbnail(2000, h)
    else:
        w = 256
        im.thumbnail(w,2000)

    w = image.size[0]
    h = image.size[1]
    r = 224
    left = (w - r)/2
    top = (h - r)/2
    right = left + 224
    bottom = top + 224
    im = im.crop ((left, top, right, bottom))

    np_image = np.array(im)/255 
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])

    np_image= np_image.transpose ((2,0,1))
    return np_image

def predict(image_path, model, topkl, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path) 
    if device == 'cuda':
        im = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        im = torch.from_numpy(image).type(torch.FloatTensor)

    im = im.unsqueeze(dim = 0) 
    model.to (device)
    im.to (device)

    with torch.no_grad():
        output = model.forward(im)
    output_prob = torch.exp(output) 

    prob, index = output_prob.topk (topkl)
    prob = prob.cpu()
    index = index.cpu()
    prob = prob.numpy() 
    index = index.numpy()

    prob = prob.tolist()[0] 
    index = index.tolist()[0]

    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }

    classes = [mapping [item] for item in index]
    classes = np.array(classes) 

    return prob, classes

args = parser.parse_args()
file_path = args.image_dir

if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


model = loading_model(args.load_dir)


prob, classes = predict (file_path, model, args.top_k, device)

class_nm = [cat_to_name [item] for item in classes]

for l in range (nm):
     print("Number: {}/{}.. ".format(l+1, nm),
            "Class name: {}.. ".format(class_nm [l]),
            "Probability: {:.3f}..% ".format(prob [l]*100),
            )