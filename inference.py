"""
    Script for running inference of AttGAN on CelebA dataset.
"""

import torch
from model.attGAN_model import EncoderDecoder, BaseModel, Classifier, Discriminator
from dataprocessing.load_dataset import get_data_loader
import matplotlib.pyplot as plt
import copy
import argparse

parser = argparse.ArgumentParser(description="attGAN training")
parser.add_argument("--pre_trained", default=None, type=str, help="pretrained model") 
parser.add_argument("--images_path", default=None, type=str, help="path to dataset images")
parser.add_argument("--attributes_path", default=None, type=str, help="path to attribute file")
parser.add_argument("--device", default="cuda", type=str, help="device to run on")
args = parser.parse_args()

data_loader = get_data_loader(argparse.images_path, argparse.attributes_path, shuffle=False)
attr_list = ["Bald", "Bangs", "Dark_Hair", "Blond_Hair", "Bushy_Eyebrows", "Eyeglasses", "Male", "Mouth_Slightly_Open", "Smiling", "Facial_Hair", "Pale_Skin", "Young"]

encoder_decoder = EncoderDecoder().to(argparse.device).eval()
base_model = BaseModel().to(argparse.device).eval()
classifier = Classifier(base_model).to(argparse.device).eval()
discriminator = Discriminator(base_model).to(argparse.device).eval()

state_dicts = torch.load(argparse.pre_trained)
encoder_decoder.load_state_dict(state_dicts['encoder_decoder'])
base_model.load_state_dict(state_dicts['base_model'])   
classifier.load_state_dict(state_dicts['classifier'])
discriminator.load_state_dict(state_dicts['discriminator'])


#prepare data
data = iter(data_loader)
batch = next(data)
batch = next(data)

imgs = batch[0].to(argparse.device)
attr = batch[1].to(argparse.device)
attr_permuted = copy.deepcopy(attr)
attr_permuted[:, 5] = 1 #add eyeglasses
diffs = attr_permuted - attr


#show which attributes have been changed - this might be inverse to what is displayed
changes = []
for i in range(diffs.size(0)):
    # For each example, find the indices where attributes have changed
    change_indices = torch.where(diffs[i] != 0)[0]
    # Store a tuple of (attribute index, change type) for each changed attribute
    example_changes = [(idx.item(), attr_list[idx.item()], 'Added' if diffs[i, idx] > 0 else 'Removed') for idx in change_indices]
    changes.append(example_changes)
print(changes[0])

#generate samples
with torch.no_grad():
    gen_perm = encoder_decoder(imgs, attr_permuted)
    gen_no_perm = encoder_decoder(imgs, attr)

#display generated samples
for idx in range(10):
    img_orig = imgs[idx].permute(1,2,0).detach().cpu().numpy()
    img_orig = (img_orig + 1) / 2
    img_no_perm = gen_no_perm[idx].permute(1,2,0).detach().cpu().numpy()
    img_no_perm = (img_no_perm + 1) / 2
    img_perm = gen_perm[idx].permute(1,2,0).detach().cpu().numpy()
    img_perm = (img_perm + 1) / 2


    print(changes[idx])
    print(attr[idx][:30])
    print(attr_permuted[idx][:30])

    fig, axr = plt.subplots(1, 3, figsize=(10, 10))
    axr[0].imshow(img_orig)
    axr[1].imshow(img_no_perm)
    axr[2].imshow(img_perm)

    plt.show()



