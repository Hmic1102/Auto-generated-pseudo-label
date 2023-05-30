import pandas as pd
import os
#from skimage import io,transform
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class PseudoLabelDataset(Dataset):

  def __init__(self, csv_file, root_dir, transform=None):
      
      self.root_dir = root_dir
      self.csv_file = os.path.join(self.root_dir,csv_file) 
      self.lavh = pd.read_csv(self.csv_file,sep=" ")
      self.transform = transform
      print(self.csv_file)

  def __len__(self):
      return len(self.lavh)

  def __getitem__(self,idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_name = self.lavh.iloc[idx,0]
      #image = io.imread(img_name)
      with open(img_name, "rb") as f:
        image = Image.open(f).convert("RGB")
      #image = Image.open(img_name)
      #print(image.size)
      if self.transform: 
          image = self.transform(image)
      target = int(self.lavh.iloc[idx,1])
      #sample={'image': image, 'target': target}
      return image, target

#def main():
#    train_dataset=PseudoLabelDataset(csv_file='../../../new_code_policies/imagenet1k_train_0000_LAVH_n.csv')
#    for i in range(len(train_dataset)):
#        print(i, len(train_dataset),train_dataset[i])
#        sample = train_dataset[i]
#        print(sample)
#        print(i)
#        print(sample['image'].size)
#        print(sample['target'])

#if __name__=="__main__":
#    main()
