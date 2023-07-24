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

parser = argparse.ArgumentParser()

parser.add_argument ('data_dir', help = 'Provide data directory.', type = str)
parser.add_argument ('arch', help = 'Vgg13 or Alexnet', type = str)
parser.add_argument ('lr', help = 'Learning rate.', type = float, default = 0.01)
parser.add_argument ('epochs', help = 'Number of epochs', type = int, default = 20)
parser.add_argument ('GPU', help = "GPU or CPU", type = str)

args = parser.parse_args ()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

if data_dir:
    train_data_transforms = transforms.Compose ([transforms.RandomRotation (30),
                                                transforms.RandomResizedCrop (224),
                                                transforms.RandomHorizontalFlip (),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    valid_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    test_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])
    
    train_image_datasets = datasets.ImageFolder (train_dir, transform = train_data_transforms)
    valid_image_datasets = datasets.ImageFolder (valid_dir, transform = valid_data_transforms)
    test_image_datasets = datasets.ImageFolder (test_dir, transform = test_data_transforms)

   
    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 128, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 128, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 128, shuffle = True)
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model (arch):
    if arch == 'alexnet': 
        model = models.alexnet(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear(9216, 4096)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p = 0.3)),
                            ('fc2', nn.Linear(4096, 2048)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p = 0.3)),
                            ('fc3', nn.Linear(2048, 1024)),
                            ('relu3', nn.ReLU()),
                            ('dropout3', nn.Dropout(p = 0.3)),
                            ('fc4', nn.Linear(1024, 512)),
                            ('relu4', nn.ReLU()),
                            ('dropout4', nn.Dropout(p = 0.3)),
                            ('fc5', nn.Linear(512, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
            
    else: 
        arch = 'vgg13' 
        model = models.vgg13(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential  (OrderedDict ([('fc1', nn.Linear(9216, 4096)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p = 0.3)),
                            ('fc2', nn.Linear(4096, 2048)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p = 0.3)),
                            ('fc3', nn.Linear(2048, 1024)),
                            ('relu3', nn.ReLU()),
                            ('dropout3', nn.Dropout(p = 0.3)),
                            ('fc4', nn.Linear(1024, 512)),
                            ('relu4', nn.ReLU()),
                            ('dropout4', nn.Dropout(p = 0.3)),
                            ('fc5', nn.Linear(512, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
    model.classifier = classifier 
    return model, arch

def validation(model, valid_loader, criterion):
    model.to (device)

    vl = 0
    accuracy = 0
    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        vl += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return vl, accuracy
model, arch = load_model (args.arch)
criterion = nn.NLLLoss ()
optimizer = optim.Adam (model.classifier.parameters(), lr = 0.01)
model.to (device) 
pe = 10
steps = 0
for e in range (args.epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate (train_loader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad () 
        outputs = model.forward (inputs) 
        loss = criterion (outputs, labels) 
        loss.backward ()
        optimizer.step () 
        running_loss += loss.item () 

        if steps % pe == 0:
            model.eval () 
            with torch.no_grad():
                vl, accuracy = validation(model, valid_loader, criterion)
            running_loss = 0
            model.train()
model.class_to_idx = train_image_datasets.class_to_idx 
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':    model.class_to_idx
             }
torch.save (checkpoint, 'checkpoint.pth')