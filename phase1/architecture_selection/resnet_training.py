


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
        
        # Training Dataset
        self.name = pd.read_csv(self.label_path + 'label_file.csv', usecols=range(5,6))   
        self.labels = pd.read_csv(self.label_path + 'label_file.csv', usecols=range(6,7))
        self.center_data = pd.concat([self.name, self.labels], axis=1) #combine image name and label dataframes
        
        # Training Dataset Only!!!
        self.center_data = self.center_data[self.center_data["filename"].str.contains('center')] # only keep center image names and labels
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
        # For Training Dataset
        img = Image.open(self.image_path + str(self.name.iloc[idx, 0])[7:]) # [7:] removes 'center/' before the image number since the zipped images just have numbers
        
        img = img.crop((0, 240, 640, 480))  # crop image (remove above horizon). Original dimensions: 640,480. changing to 640,240    
        img = img.resize((224,224)) #resize image --> pretrained alexnet model needs img sizes of 224 x 224
        transform = transforms.ToTensor()
        img = transform(img)
        label = self.labels.iloc[idx][0]
        return img, label



def train_test_split(dataset, batch_size = 16, validation_split = .2, shuffle_dataset = True):
  size = len(dataset) 
  idx = list(range(size))
  if shuffle_dataset: 
    np.random.seed(42)
    np.random.shuffle(idx)

  split = int(np.floor(validation_split * size))
  train_idx, validation_idx = idx[split:], idx[:split]

  # create data subsamplers
  train_sampler = SubsetRandomSampler(train_idx)
  validation_sampler = SubsetRandomSampler(validation_idx)

  # create data loaders 
  train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler) 
  validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)
  return train_loader, validation_loader



def load_data(image_path, label_path, data_transforms=None, num_workers=0, batch_size=1, shuffle=True):
    dataset = SteeringDataset(image_path, label_path, data_transforms)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle)



class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        m = nn.MSELoss()
        return torch.sqrt(m(input.view(input.size(0)),target))
    


def train(args, model, train_loader, model_save_name):
  model.train() 
  
  model.to(args.device)

  # define optimizer and loss 
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) 
  criterion = ClassificationLoss() 
  
  criterion.to(args.device)

  train_losses = [] 
  loss_list = [] 
  for epoch in range(1,args.num_epochs+1):
    ### 
    for batch_idx, (data, target) in enumerate(train_loader): 
        
      data = data.to(args.device)
      target = target.to(args.device)
        
      optimizer.zero_grad() 
      output = model(data)
      loss = criterion(output,target)
      loss.backward() 
      optimizer.step() 

      loss_list.append(loss.item()) 
      if batch_idx % args.log_interval == 0: 
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, (batch_idx+1) * len(data), len(train_loader)*len(data),
                  100. * batch_idx / len(train_loader), loss.item()))
        
        train_losses.append(loss.item())
    torch.save(model.state_dict(), './' + model_save_name)
    print(f'\n Average loss for Epoch {epoch}: {sum(loss_list)/len(loss_list)} \n')
    del loss_list[:]
  return model



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
    
  return



class Args(object):
  def __init__(self):
    self.learning_rate = 0.001 
    self.momentum = .5 
    self.num_epochs = 6

    self.batch_size = 16 
    self.log_interval = 10
    self.val_size = 0.2
    self.shuffle = True
    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    




'''MAIN ***************************************************************************'''

# Training Dataset Paths
label_path = 'C:/Users/zfish/Documents/CIS519_Project/data/train_center_data/'
image_path = 'C:/Users/zfish/Documents/CIS519_Project/data/train_center_data/images/'


# Parameters
args = Args() 


# Model Selection: (Custom Resnet18)
resnet = models.resnet18(pretrained=False, progress=True)
for param in resnet.parameters():
    param.requires_grad = True 
resnet.fc = nn.Sequential(nn.Linear(512, 512),
                          nn.ReLU(),
                          nn.Linear(512, 1))


# Load & Train All Training Data
dataset_loader = load_data(image_path, label_path, batch_size=args.batch_size, shuffle=args.shuffle)
train(args, model=resnet, train_loader=dataset_loader, model_save_name='model_V#.pt')


# Cross-Validation Loading / Training / Testing
'''
data = SteeringDataset(image_path, label_path)
train_loader, validation_loader = train_test_split(data, batch_size=args.batch_size, validation_split=args.val_size, shuffle_dataset=True)
model = train(args, model=resnet, train_loader=train_loader, model_save_name='model_V#_crossval')
test(model, validation_loader)
'''


