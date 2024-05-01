import pandas as pd
import torch
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

IMG_SIZE = 128

attributes = pd.read_csv("/home/ulrik/datasets/celeba/list_attr_celeba.csv")
attributes.replace(-1, 0, inplace=True)


#Merge attributes
attributes["Facial_Hair"] = ((attributes["Goatee"]==1) | (attributes["Mustache"]==1) | (attributes["No_Beard"]==0)).astype(int)
attributes["Dark_Hair"] = ((attributes["Black_Hair"]==1) | (attributes["Brown_Hair"]==1)).astype(int)
#attributes["Mouth_Open"] = ((attributes["Smiling"]==1) | (attributes["Mouth_Slightly_Open"]==1)).astype(int) #this one is not so obvious

desired_attributes = ["Bald", "Bangs", "Dark_Hair", "Blond_Hair", "Bushy_Eyebrows", "Eyeglasses", "Male", "Mouth_Slightly_Open", "Smiling", "Facial_Hair", "Pale_Skin", "Young"]
attributes = attributes[["image_id"] + desired_attributes]

#convert the dataframe to a dictionary that is indexed by the filename
attributes.set_index('image_id', inplace=True)
attribute_dict = {idx: torch.Tensor(row.values) for idx, row in attributes.iterrows()}

img_transform = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #normalize to [-1, 1] to match with the tanh generator output
])

class faceDataLoader(Dataset):
    def __init__(self, dataset, transforms=None, train_classifier=False):
        self.dataset = dataset
        self.transforms = transforms
        self.train_classifier = train_classifier
        self.noise_threshold = 0.1
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.train_classifier and random.random() < self.noise_threshold:
            img = torch.randn(3, IMG_SIZE, IMG_SIZE) * 2 - 1
            attributes = torch.zeros(13)
            return img, attributes

        img = Image.open(self.dataset[idx])
        if self.transforms:
            img = img_transform(img)
        
        return (img, attribute_dict[self.dataset[idx][-10:]]) #pair img with its attribute vector
    

def get_data_loader(dataset_path, train_classifier=False, shuffle=True):
    files = sorted(glob.glob(dataset_path + "/*.*"))
    files = [x for x in files if x[-10:] in attribute_dict.keys()]
    print(f"Dataset length: {len(files)}")
    data_loader = DataLoader(faceDataLoader(files, img_transform, train_classifier), batch_size=32, shuffle=shuffle, drop_last=True)
    return data_loader
