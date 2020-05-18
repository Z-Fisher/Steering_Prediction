import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''MAIN ***************************************************************************'''


# Training Dataset Paths
label_path = 'C:/Users/zfish/Documents/CIS519_Project/data/train_center_data/label_file.csv'
image_path = 'C:/Users/zfish/Documents/CIS519_Project/data/train_center_data/images/'


# Train Data
name = pd.read_csv(label_path, usecols=range(5,6))
labels = pd.read_csv(label_path, usecols=range(6,7))


# Count Occurences within Each Bin
i = -2
grouping_size = 0.04
occurences = []
bar_names = []
while i <= 2.1:
    counter = 0
    bar_names.append(str(i))
    for label in labels.values:
        if label > i and label < i+grouping_size:
            counter += 1
    occurences.append(counter)
    i += grouping_size
bar_names = tuple(bar_names)


# Plot Bar Graph
y_pos = np.arange(len(bar_names))
plt.bar(y_pos, occurences, align='center', alpha=1)

reduced_y_pos = []
reduced_bar_names = []
for i in y_pos:
    if i % 10 == 0:
        reduced_y_pos.append(i)
        reduced_bar_names.append(round(float(bar_names[i]), 2))
reduced_bar_names = tuple(reduced_bar_names)
    
    
plt.xticks(reduced_y_pos, reduced_bar_names)
plt.ylabel('Number of Images Labeled')
plt.title('Occurences of Steering Angles in Training Data')
plt.show()


