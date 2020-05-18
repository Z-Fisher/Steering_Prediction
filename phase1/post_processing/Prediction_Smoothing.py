import numpy as np 
import pandas as pd 

from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.ndimage import gaussian_filter1d



'''FUNCTIONS*******************************************************************'''

def avgRMSE(labels, predictions, string_label):
    test_loss = 0
    for i in range(len(predictions)):
        loss =  RMSE(labels.values[i], predictions.values[i])
        test_loss += loss 
    test_loss /= len(predictions) 
    print(string_label + ' Avg RMSE Loss: ', round(test_loss, 6))



def RMSE(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms



def Calc_Jump(train_labels, string_name):
    max_jump = 0
    prev_label = train_labels.values[0]
    for label in train_labels.values[1:]:
            jump = label - prev_label
            abs_jump = np.abs(jump)
            if abs_jump > max_jump:
                max_jump = abs_jump
            prev_label = label
    MAX_JUMP = np.round(np.squeeze(max_jump), 6)
    print('Max Steering Angle Jump in ' + string_name + ': ', MAX_JUMP)
    return MAX_JUMP



def Jump_Smoothing(max_jump_training, predictions):

    smoothed_preds = []
    prev_pred = predictions.values[0]
    smoothed_preds.append(prev_pred)
    
    for pred in predictions.values[1:]:
        jump = pred - prev_pred
        abs_jump = np.abs(jump)
        if abs_jump > max_jump_training:
            pred = prev_pred + (np.sign(jump) * max_jump_training)
            print('ping')
        
        smoothed_preds.append(pred)
        prev_pred = pred
        
    smoothed_preds = np.squeeze(smoothed_preds)
    smoothed_preds = pd.DataFrame(smoothed_preds, columns=['smoothed_preds'])
    return smoothed_preds




'''MAIN ***************************************************************************'''


# Training Dataset Paths
train_label_path = 'C:/Users/zfish/Documents/CIS519_Project/data/train_center_data/label_file.csv'
train_image_path = 'C:/Users/zfish/Documents/CIS519_Project/data/train_center_data/images/'

#Testing Dataset Paths
test_label_path = 'C:/Users/zfish/Documents/CIS519_Project/data/test_center_data/label_file.csv'
test_output_data_path = 'C:/Users/zfish/Documents/CIS519_Project/Video_Inputs.csv'
test_image_path = 'C:/Users/zfish/Documents/CIS519_Project/data/test_center_data/images/'
model_path = 'C:/Users/zfish/Documents/CIS519_Project/resnet_custom_6epochs.pt'

# Load Test Data
test_labels = pd.read_csv(test_output_data_path, usecols=range(1,2))
predictions = pd.read_csv(test_output_data_path, usecols=range(2,3))

# Load Train Data
train_labels = pd.read_csv(train_label_path, usecols=range(6,7))


# Control: Original Average RMSE Loss
avgRMSE(test_labels, predictions, 'Original')

'''
# Find Biggest Angle Jump in Sequential Training Data
max_jump_training = Calc_Jump(train_labels, 'Training')

# See Biggest Angle Jump in Sequential Test Predictions (Noticeably Larger)
max_jump_predictions = Calc_Jump(predictions, 'Predictions') 

# Limit Angle Predictions changes by the max_jump_training
jump_smoothed_predictions = Jump_Smoothing(max_jump_training, predictions)

# Examine New Average RMSE Loss of Predictions
avgRMSE(test_labels, jump_smoothed_predictions, 'Jump Smoothed')
'''


# Gaussian Smoothing
gaus_predictions = gaussian_filter1d(np.squeeze(predictions.values), 10)
gaus_predictions = pd.DataFrame(gaus_predictions, columns=['gaus_preds'])

# Examine New Average RMSE Loss of Gaussian
avgRMSE(test_labels, gaus_predictions, 'Gaussian Smoothed')


# Outputs to CSV
name = pd.read_csv(test_label_path, usecols=range(0,1))
labels = pd.read_csv(test_label_path, usecols=range(1,2))
df = pd.concat([name, labels], axis=1) 
video_df = pd.concat([df, gaus_predictions], axis=1)
video_df.to_csv(r'C:/Users/zfish/Documents/CIS519_Project/Video_Inputs_Gaussian.csv', index = False)


