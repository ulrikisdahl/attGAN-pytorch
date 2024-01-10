import pandas as pd
import torch
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


attributes = pd.read_csv("/kaggle/input/celeba-dataset/list_attr_celeba.csv")
attributes.replace(-1, 0, inplace=True)

#drop all females
attributes = attributes[attributes["Male"] != 0]

#convert the dataframe to a dictionary that is indexed by the filename
attributes.set_index('image_id', inplace=True)
attribute_dict = {idx: torch.Tensor(row.values) for idx, row in attributes.iterrows()}


img_transform = transforms.Compose([
    transforms.Resize(64),###????? RIKTIG?
    transforms.ToTensor()
])

class faceDataLoader(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = Image.open(self.dataset[idx])
        if self.transforms:
            img = img_transform(img)
        
        return (img, attribute_dict[self.dataset[idx][-10:]]) #pair img with its attribute vector
    

dataset_path = "/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba"
files = sorted(glob.glob(dataset_path + "/*.*"))
files = [x for x in files if x[-10:] in attribute_dict.keys()]


data_loader = DataLoader(faceDataLoader(files, img_transform), batch_size=32)
