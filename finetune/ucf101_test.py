import os
import numpy as np
import cv2
import time
import argparse



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--policy',default = 'vanilla', type = str, 
                    help = 'policy for pretained model')

parser.add_argument('--dataset',default = None, type = str, 
                    help = 'dataset used for transferlearning')

parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
args = parser.parse_args()
def getUCF101(base_directory = ''):

    # action class labels
    class_file = open(base_directory + 'ucfTrainTestlist/classInd.txt','r')
    lines = class_file.readlines()
    lines = [line.split(' ')[1].strip() for line in lines]
    class_file.close()
    class_list = np.asarray(lines)

    # training data
    train_file = open(base_directory + 'ucfTrainTestlist/trainlist01.txt','r')
    lines = train_file.readlines()
    filenames = ['UCF-101/' + line.split(' ')[0] for line in lines]
    y_train = [int(line.split(' ')[1].strip())-1 for line in lines]
    y_train = np.asarray(y_train)
    filenames = [base_directory + filename for filename in filenames]
    train_file.close()

    train = (np.asarray(filenames),y_train)

    # testing data
    test_file = open(base_directory + 'ucfTrainTestlist/testlist01.txt','r')
    lines = test_file.readlines()
    filenames = ['UCF-101/' + line.split(' ')[0].strip() for line in lines]
    classnames = [filename.split('/')[1] for filename in filenames]
    y_test = [np.where(classname == class_list)[0][0] for classname in classnames]
    y_test = np.asarray(y_test)
    filenames = [base_directory + filename for filename in filenames]
    test_file.close()

    test = (np.asarray(filenames),y_test)

    return class_list, train, test


def loadFrame(args):
    mean = np.asarray([0.485, 0.456, 0.406],np.float32)
    std = np.asarray([0.229, 0.224, 0.225],np.float32)

    curr_w = 320
    curr_h = 240
    height = width = 224
    (filename,augment) = args

    data = np.zeros((3,height,width),dtype=np.float32)

    try:
        ### load file from HDF5=
        video = cv2.VideoCapture(filename)
        nFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = np.random.randint(nFrames-1)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()
        video.release()

        if(augment==True):
            ## RANDOM CROP - crop 70-100% of original size
            ## don't maintain aspect ratio
            if(np.random.randint(2)==0):
                resize_factor_w = 0.3*np.random.rand()+0.7
                resize_factor_h = 0.3*np.random.rand()+0.7
                w1 = int(curr_w*resize_factor_w)
                h1 = int(curr_h*resize_factor_h)
                w = np.random.randint(curr_w-w1)
                h = np.random.randint(curr_h-h1)
                frame = frame[h:(h+h1),w:(w+w1)]
            
            ## FLIP
            if(np.random.randint(2)==0):
                frame = cv2.flip(frame,1)

            frame = cv2.resize(frame,(width,height))
            frame = frame.astype(np.float32)

            ## Brightness +/- 15
            brightness = 30
            random_add = np.random.randint(brightness+1) - brightness/2.0
            frame += random_add
            frame[frame>255] = 255.0
            frame[frame<0] = 0.0

        else:
            # don't augment
            frame = cv2.resize(frame,(width,height))
            frame = frame.astype(np.float32)

        ## resnet model was trained on images with mean subtracted
        frame = frame/255.0
        frame = (frame - mean)/std
        frame = frame.transpose(2,0,1)
        data[:,:,:] = frame
    except:
        print("Exception: " + filename)
        data = np.array([])
    return data


import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision




import cv2

from multiprocessing import Pool

IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = args.lr
num_of_epochs = args.epochs


data_directory = '/scratch/zh2033/ucf101/'
class_list, train, test = getUCF101(base_directory = data_directory)


model =  torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048,NUM_CLASSES)

model.cuda()

my_list = ['fc.weight', 'fc.bias']
params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))

optimizer = torch.optim.SGD([
                            {'params': [temp[1] for temp in base_params]},
                            {'params': [param[1] for param in params],'lr': args.lr*10}],
                              lr = args.lr, momentum=0.9,
                                weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()


pool_threads = Pool(8,maxtasksperchild=200)



for epoch in range(0,num_of_epochs):

    ###### TRAIN
    train_accu = []
    model.train()
    random_indices = np.random.permutation(len(train[0]))
    start_time = time.time()
    for i in range(0, len(train[0])-batch_size,batch_size):

        augment = True
        video_list = [(train[0][k],augment)
                       for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadFrame,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this
                next_batch = 1
        if(next_batch==1):
            continue

        x = np.asarray(data,dtype=np.float32)
        x = Variable(torch.FloatTensor(x)).cuda().contiguous()

        y = train[1][random_indices[i:(batch_size+i)]]
        y = torch.from_numpy(y).cuda()

        output = model(x)

        loss = criterion(output, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        prediction = output.data.max(1)[1]
        accuracy = ( float( prediction.eq(y.data).sum() ) /float(batch_size))*100.0
        if(epoch==0):
            print(i,accuracy)
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch,time.time()-start_time)


model.eval()
test_accu = []
random_indices = np.random.permutation(len(test[0]))
t1 = time.time()
for i in range(0,len(test[0])-batch_size,batch_size):
    augment = False
    video_list = [(test[0][k],augment) 
                    for k in random_indices[i:(batch_size+i)]]
    data = pool_threads.map(loadFrame,video_list)

    next_batch = 0
    for video in data:
        if video.size==0: # there was an exception, skip this batch
            next_batch = 1
    if(next_batch==1):
        continue

    x = np.asarray(data,dtype=np.float32)
    x = Variable(torch.FloatTensor(x)).cuda().contiguous()

    y = test[1][random_indices[i:(batch_size+i)]]
    y = torch.from_numpy(y).cuda()

    output = model(x)

    prediction = output.data.max(1)[1]
    accuracy = ( float( prediction.eq(y.data).sum() ) /float(batch_size))*100.0
    test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)
print('Testing',accuracy_test,time.time()-t1)


torch.save(model,'single_frame.model')
pool_threads.close()
pool_threads.terminate()




##### TEST

