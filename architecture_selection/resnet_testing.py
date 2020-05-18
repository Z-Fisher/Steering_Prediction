


import torch 
from torchvision import transforms, utils 
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd 

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import torchvision

from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.models as models 
import torch.nn as nn 

import torch.optim as optim 


'''FUNCTIONS*******************************************************************'''

class SteeringDataset(Dataset):
    def __init__(self, image_path, label_path, data_transforms=None):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        self.image_path = image_path
        self.label_path = label_path
        
        # Testing Dataset
        self.name = pd.read_csv(self.label_path + 'label_file.csv', usecols=range(0,1))
        self.labels = pd.read_csv(self.label_path + 'label_file.csv', usecols=range(1,2))
        self.center_data = pd.concat([self.name, self.labels], axis=1) #combine image name and label dataframes
        self.name = pd.DataFrame(self.center_data[self.center_data.columns[0]]) # center images names
        self.len = self.name.shape[0]
        self.labels = pd.DataFrame(self.center_data[self.center_data.columns[1]]) # center image labels


    def __len__(self):
        """
        Your code here
        """
        return self.len

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """        
        # For Test Dataset
        img = Image.open(self.image_path + str(self.name.iloc[idx, 0]) + '.jpg')
        img = img.crop((0, 240, 640, 480))  # crop image (remove above horizon). Original dimensions: 640,480. changing to 640,240    
        img = img.resize((224,224)) #resize image --> pretrained alexnet model needs img sizes of 224 x 224
        transform = transforms.ToTensor()
        img = transform(img)
        label = self.labels.iloc[idx][0]
        return img, label



def load_data(image_path, label_path, data_transforms=None, num_workers=0, batch_size=1):
    dataset = SteeringDataset(image_path, label_path, data_transforms)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)



class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        m = nn.MSELoss()
        return torch.sqrt(m(input.view(input.size(0)),target))
    


def test(path_to_model, test_loader): 

  resnet.load_state_dict(torch.load(path_to_model))

  resnet.eval() 

  predictions = []

  # define loss 
  criterion = ClassificationLoss() 
  test_loss = 0 
  with torch.no_grad(): 
    for batch_idx, (data, target) in enumerate(test_loader): 
      output = resnet(data)
      #print(output)
      predictions.append(np.array(output)) # for video maker  
      loss =  criterion(output, target).item() 
      test_loss += loss 
      if batch_idx % 100 == 0:
        print(f'Batch_idx: {batch_idx}   Average Loss So Far: {test_loss / (batch_idx+1)}')

    test_loss /= len(test_loader) 
    print('Overall test loss = ', test_loss)
    
  return predictions



class Args(object):
  def __init__(self):

    self.batch_size = 1
    self.log_interval = 10
    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    




'''MAIN ***************************************************************************'''


# Training Dataset Paths
#label_path = 'C:/Users/zfish/Documents/CIS519_Project/data/train_center_data/'
#image_path = 'C:/Users/zfish/Documents/CIS519_Project/data/train_center_data/images/'

#Testing Dataset Paths
label_path = 'C:/Users/zfish/Documents/CIS519_Project/data/test_center_data/'
image_path = 'C:/Users/zfish/Documents/CIS519_Project/data/test_center_data/images/'
#model_path = 'C:/Users/zfish/Documents/CIS519_Project/data/TRAINED_RESNET_CUSTOM.pt'
#model_path = 'C:/Users/zfish/Documents/CIS519_Project/data/TRAINED_RESNET.pt'
model_path = 'C:/Users/zfish/Documents/CIS519_Project/resnet_custom_6epochs.pt'


# Model Setup
resnet = models.resnet18(pretrained=False, progress=True)
for param in resnet.parameters():
    param.requires_grad = True 
resnet.fc = nn.Sequential(nn.Linear(512, 512),
                          nn.ReLU(),
                          nn.Linear(512, 1))


# Parameters
args = Args() 


# Test Data Loading
dataset_loader = load_data(image_path, label_path, batch_size=args.batch_size)


# Testing
predictions = test(model_path, dataset_loader)
predictions = np.squeeze(predictions)
predictions = pd.DataFrame(predictions, columns=['predicted'])


# Outputs to CSV
name = pd.read_csv(label_path + 'label_file.csv', usecols=range(0,1))
labels = pd.read_csv(label_path + 'label_file.csv', usecols=range(1,2))
df = pd.concat([name, labels], axis=1) 
video_df = pd.concat([df, predictions], axis=1)
video_df.to_csv(r'C:/Users/zfish/Documents/CIS519_Project/Video_Inputs.csv', index = False)





# BINNING: MAY BE USEFUL FOR GRAPH OF VALUE OCCURENCES IN EACH BIN
# bin angles into 100 classes
# classes = 100
# bins = np.linspace(-2.1,2,classes)
# labels = np.linspace(1,classes-1,classes-1).astype(int)
# self.labels = pd.cut(self.labels['angle'], bins = bins, labels = labels)
# print(self.name)