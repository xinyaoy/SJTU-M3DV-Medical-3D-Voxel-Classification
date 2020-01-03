import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import os
import csv
from torch.autograd import Variable
import torch.nn.functional as F
import time
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt



BATCH_SIZE = 32
NUM_EPOCHS = 80
learning_rate = 1e-3

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[.5], std=[.5])
                                ])


class MyDataset(Dataset):
     def __init__(self, root_dir, names_file,transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        '''
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1
        '''
        with open(names_file, 'r') as f:
            self.size = (len(f.readlines()))
        self.size -= 1


     def __getitem__(self,idx):
        image_path = self.root_dir
        with open(self.names_file,'r') as csvfile:
            reader = csv.DictReader(csvfile)
            label = float([row['label'] for row in reader][idx])
        

        with open(self.names_file,'r') as csvfile: 
            reader = csv.DictReader(csvfile)
            name = [row['name'] for row in reader][idx]
        

        with np.load(os.path.join(image_path, '%s.npz' % name)) as npz:
            voxel = (npz['voxel'])
            voxelWidth = 32
            voxelDepth = 32            
            coord_start = [0, 0, 0]
            coord_end = [0, 0, 0]
            voxelCoord = [50,50,50]
            coord_start[0] = int(voxelCoord[0] - voxelDepth / 2.0)
            coord_end[0] = int(voxelCoord[0] + voxelDepth / 2.0)
            coord_start[1] = int(voxelCoord[1] - voxelWidth / 2.0)
            coord_end[1] = int(voxelCoord[1] + voxelWidth / 2.0)
            coord_start[2] = int(voxelCoord[2] - voxelWidth / 2.0)
            coord_end[2] = int(voxelCoord[2] + voxelWidth / 2.0)
            patch = voxel[coord_start[0]:coord_end[0], coord_start[1]:coord_end[1], coord_start[2]:coord_end[2]]
            

        sample = {'cube': patch, 'label': label}
        if self.transform:
            tmp = sample['cube']
            tmp = self.transform(tmp)
            sample['cube'] = tmp

        return sample

     def __len__(self):
        return self.size



my_train_dataset = MyDataset(root_dir = './train1',names_file = 'train1.csv',transform = transform)
my_train_loader = data.DataLoader(my_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

my_test_dataset = MyDataset(root_dir = './test1', names_file = 'test1.csv',transform = transform)
my_test_loader = data.DataLoader(my_test_dataset,batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

test_dataset = MyDataset(root_dir = './test', names_file = 'abc.csv',transform = transform)
test_loader = data.DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

#Network
class NoduleNet(nn.Module):
    def __init__(self):
        super(NoduleNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(1,32,3),
            nn.BatchNorm3d(32),
            nn.Sigmoid(),
            nn.Conv3d(32,32,3),
            nn.BatchNorm3d(32),
            nn.Sigmoid(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.5),
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.Sigmoid(),
            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.Sigmoid(),
            nn.MaxPool3d((2,2,2)),
            nn.Dropout3d(0.5),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.Sigmoid(),
            nn.Conv3d(128, 128, 3),
            nn.BatchNorm3d(128),
            nn.Sigmoid(),
            nn.MaxPool3d((1,1,1))
            
            
        )
        self.classifer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024,32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(32,32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            nn.Linear(32, 2),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal(m.weight)
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal(m.weight)

    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        x = self.classifer(x)
        return x
classes = {'Not Nodule','Is Nodule'}

net = NoduleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = learning_rate,weight_decay = 0.00001)
max_correct = 0
max_num = 0
train_loss = []
test_loss = []

for epoch in range(NUM_EPOCHS):
    print('epoch ' + str(epoch) + ' start' )
    cnt = 0
    running_loss = 0.0
    for i, data in enumerate(my_train_loader):
        cnt += 1
        inputs, labels = data['cube'], data['label']
        inputs = inputs[:,np.newaxis]
        inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.float()
        labels = labels.long()

        outputs = net(inputs)
        loss = criterion(outputs, labels)    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每跑完一次epoch测试一下训练集准确率   
    with torch.no_grad():
        correct = 0
        total = 0
        eval_loss = 0
        eval_acc = 0
        for data in my_train_loader:
            inputs, labels = data['cube'], data['label']
            inputs = inputs[:,np.newaxis]
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = inputs.float()
            labels = labels.long()
            outputs = net(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            eval_loss += loss.data.item()*labels.size(0)
            _, pred = torch.max(outputs.data, 1)
            correct += (pred == labels).sum()
            total += labels.size(0)

        print('Train Loss: {:.6f},train accuracy: {:.6f}'.format(
            eval_loss / total,
            100.0*correct / total
        ))
        train_loss.append(eval_loss / total)
    
    with torch.no_grad():
        correct = 0
        total = 0
        eval_loss = 0
        eval_acc = 0
        for data in my_test_loader:
            inputs, labels = data['cube'], data['label']
            inputs = inputs[:,np.newaxis]
            inputs, labels = Variable(inputs), Variable(labels)
            inputs = inputs.float()
            labels = labels.long()
            outputs = net(inputs)      
            outputs = torch.nn.functional.softmax(outputs, dim=1)      
            loss = criterion(outputs, labels)
            eval_loss += loss.data.item()*labels.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum()
            total += labels.size(0)
        print('Test Loss: {:.6f},test accuracy: {:.6f}'.format(
            eval_loss / total,
            100.0*correct / total
        ))
        test_loss.append(eval_loss / total)
    
    if correct > 66:
        break
    

 # 输出测试集结果
correct = 0
total = 0
num = -1
with open('abc.csv','r') as csvfile: 
    reader = csv.DictReader(csvfile)
    name = [row['name'] for row in reader]
    print(len(name))
with open('Submission.csv',"a+",newline = '') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Id","Predicted"])
for data in test_loader:
    inputs= data['cube']
    inputs = inputs[:,np.newaxis]
    inputs = Variable(inputs)
    inputs = inputs.float()
    outputs = net(inputs)
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    _, pred = torch.max(outputs.data, 1)
    num = num+1
    score = np.array(outputs.data.cpu())
    score_1 = score[:,1]
    len2 = len(score_1)
    with open('Submission.csv',"a+",newline = '') as f:
        csv_writer = csv.writer(f)
        for j in range(len2):
            csv_writer.writerow([name[num*BATCH_SIZE+j],score_1[j]])

print(train_loss)
print(test_loss)
f.close()
torch.save(net.state_dict(), './model000132')
       
        