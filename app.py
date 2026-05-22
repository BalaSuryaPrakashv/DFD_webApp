# Interaction with the OS
import os

from flask import (Flask, json, jsonify, redirect, render_template, request,session,send_file, url_for)
from werkzeug.utils import secure_filename

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import time
import warnings

import cv2
# To recognise face from extracted frames
import face_recognition
import numpy as np
# Used for DL applications, computer vision related processes
import torch
import torchvision
from skimage import img_as_ubyte
# 'nn' Help us in creating & training of neural network
from torch import nn
# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable
# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
# For image preprocessing
from torchvision import models, transforms



#****************************************************************************************

# Display Home Page 

app = Flask(__name__ , template_folder='static/templates')
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Check if credentials are valid and set session variables
        session['username'] = request.form['username']
        return redirect(url_for('Detect'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Create new user account and set session variables
        session['username'] = request.form['username']
        return redirect(url_for('Detect'))
    return render_template('signup.html')
@app.route('/Detect',methods=['GET', 'POST'])
def Home():
  return render_template("Home.html")
@app.route('/abt',methods=['GET', 'POST'])
def abt():
  return render_template("About-us.html")

#***************************************************************************************************

# Redirect To Upload  

@app.route('/upload_video', methods=['POST'])
def Video():
  return render_template ("Upload_Video.html")

@app.route('/upload_image', methods=['POST'])
def image():
    return render_template('Upload_image.html')
@app.route('/About-us', methods=['POST'])
def About_us():
  return render_template('About-us.html')

#*********************************************************************************************************

class Model(nn.Module):
  def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
    super(Model, self).__init__()

    # returns a model pretrained on ImageNet dataset
    model = models.resnext50_32x4d(pretrained= True)

    # Sequential allows us to compose modules nn together
    self.model = nn.Sequential(*list(model.children())[:-2])

    # RNN to an input sequence
    self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

    # Activation function
    self.relu = nn.LeakyReLU()

    # Dropping out units (hidden & visible) from NN, to avoid overfitting
    self.dp = nn.Dropout(0.4)

    # A module that creates single layer feed forward network with n inputs and m outputs
    self.linear1 = nn.Linear(2048, num_classes)

    # Applies 2D average adaptive pooling over an input signal composed of several input planes
    self.avgpool = nn.AdaptiveAvgPool2d(1)



  def forward(self, x):
    batch_size, seq_length, c, h, w = x.shape

    # new view of array with same data
    x = x.view(batch_size*seq_length, c, h, w)

    fmap = self.model(x)
    x = self.avgpool(fmap)
    x = x.view(batch_size, seq_length, 2048)
    x_lstm,_ = self.lstm(x, None)
    return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.squeeze()
  image = inv_normalize(image)
  image = image.numpy()
  image = image.transpose(1,2,0)
  image = image.clip(0,1)
  cv2.imwrite('./2.png', image*255)
  return image

# For prediction of output  
def predict(model, img, path='./'):
  # use this command for gpu    
  # fmap, logits = model(img.to('cuda'))
  fmap, logits = model(img.to())
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _, prediction = torch.max(logits, 1)
  confidence = logits[:, int(prediction.item())].item()*100
  print('confidence of prediction: ', logits[:, int(prediction.item())].item()*100)
  return [int(prediction.item()), confidence]


# To validate the dataset
class validation_dataset(Dataset):
  def __init__(self, video_names, sequence_length = 60, transform=None):
    self.video_names = video_names
    self.transform = transform
    self.count = sequence_length

  # To get number of videos
  def __len__(self):
    return len(self.video_names)

  # To get number of frames
  def __getitem__(self, idx):
    video_path = self.video_names[idx]
    frames = []
    a = int(100 / self.count)
    first_frame = np.random.randint(0,a)
    for i, frame in enumerate(self.frame_extract(video_path)):
      faces = face_recognition.face_locations(frame)
      try:
        top,right,bottom,left = faces[0]
        frame = frame[top:bottom, left:right, :]
      except:
        pass
      frames.append(self.transform(frame))
      if(len(frames) == self.count):
        break
    frames = torch.stack(frames)
    frames = frames[:self.count]
    return frames.unsqueeze(0)

  # To extract number of frames
  def frame_extract(self, path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
      success, image = vidObj.read()
      if success:
        yield image


def Result(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
    path_to_videos= [videoPath]

    video_dataset = validation_dataset(path_to_videos,sequence_length = 20,transform = train_transforms)
    # use this command for gpu
    # model = Model(2).cuda()
    model = Model(2)
    path_to_model = 'model\\df_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    for i in range(0,len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model,video_dataset[i],'')
        if prediction[0] == 1:
            return ['FAKE',prediction[1]]
        else:
            return ['REAL',prediction[1]]
 

  

#***********************************************************************************************************

# Upload File 


UPLOAD_FOLDER = 'static'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


#********************************************************************************************************

# Video Prediction & Display Result 

@app.route('/predict_video', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        video = request.files['file']
        filename = video.filename
        print(video.filename)
        basepath = os.path.dirname(__file__)
        print(basepath)
        video_path = os.path.join(basepath, 'static/uploads', secure_filename(video.filename))
        print(video_path)
        video.save(video_path)
        preds = Result(video_path)
    return render_template("Display_Video.html",prediction =preds[0] ,confidence=preds[1],video_path ='uploads/' + filename)

#***********************************************************************************************************

# Image Prediction & Display Result 

@app.route('/Predict_image', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        video = request.files['file']
        filename = video.filename
        print(video.filename)
        basepath = os.path.dirname(__file__)
        print(basepath)
        video_path = os.path.join(basepath, 'static/uploads', secure_filename(video.filename))
        print(video_path)
        video.save(video_path)
        preds = Result(video_path)
    return render_template("Display_image.html",prediction =preds[0] ,confidence=preds[1],image_path ='uploads/' + filename)

#****************************************************************************************************************

""" # Run Flask Application  """

if __name__ =='__main__': 
  app.run(debug=True)
  