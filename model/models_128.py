import torch
import torch.nn as nn
import torch.nn.functional as F
#PARAMS
NUM_ATTRIBUTES = 13 #length of the attribute vector
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        #conv layers (encoder)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2)

        #deconv layers (decoder)
        self.deConv1 = nn.ConvTranspose2d(in_channels=1024 + NUM_ATTRIBUTES, out_channels=1024, kernel_size=3, stride=2)
        self.deConv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2)
        self.deConv3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
        self.deConv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2)
        self.deConv5 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.deConv6 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2)

        #batchnorm inits
        self.bnEnc1 = nn.BatchNorm2d(128)
        self.bnEnc2 = nn.BatchNorm2d(256)
        self.bnEnc3 = nn.BatchNorm2d(512)
        self.bnEnc4 = nn.BatchNorm2d(512)
        self.bnEnc5 = nn.BatchNorm2d(1024)

        self.bnDec1 = nn.BatchNorm2d(1024)
        self.bnDec2 = nn.BatchNorm2d(512)
        self.bnDec3 = nn.BatchNorm2d(512)
        self.bnDec4 = nn.BatchNorm2d(256)
        self.bnDec5 = nn.BatchNorm2d(128)

    def forward(self, x, attribute_vec):
        #downsample
        x1 = F.leaky_relu(self.bnEnc1(self.conv1(x)))
        x2 = F.leaky_relu(self.bnEnc2(self.conv2(x1))) 
        x3 = F.leaky_relu(self.bnEnc3(self.conv3(x2)))
        x4 = F.leaky_relu(self.bnEnc4(self.conv4(x3))) 
        z = F.leaky_relu(self.bnEnc5(self.conv5(x4)))

        #concatenation in the latent space
        attribute_vec = torch.reshape(attribute_vec, (BATCH_SIZE, NUM_ATTRIBUTES, z.shape[-2], z.shape[-1]))
        x = torch.cat((z, attribute_vec), axis=1) #concatenation with attribute vector along channel axis

        #upsample
        x = F.leaky_relu(self.bnDec1(self.deConv1(x)))
        x = F.leaky_relu(self.bnDec2(self.deConv2(x)))
        x = F.leaky_relu(self.bnDec3(self.deConv3(x)))
        x = F.leaky_relu(self.bnDec4(self.deConv4(x)))
        x = F.leaky_relu(self.bnDec5(self.deConv5(x)))
        x = torch.tanh(self.deConv6(x))
        return x

        
#base model
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
         
        #conv layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2) 
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=1)
    
        #batchnorm inits
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        return x
    
class Discriminator(nn.Module):
    def __init__(self, model):
        super(Discriminator, self).__init__() 
        
        #layers
        self.base_model = model
        self.dense1 = nn.Linear(in_features=1024*3*3, out_features=1024) # 512*1*1 is latent representation from base_model
        self.discriminator = nn.Linear(in_features=1024, out_features=1)
        
        self.ln1 = nn.LayerNorm(1024)
        
    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.ln1(self.dense1(x)))
        x = F.sigmoid(self.discriminator(x))
        return x

class Classifier(nn.Module):
    def __init__(self, model):
        super(Classifier, self).__init__()
        
        #layers
        self.base_model = model
        self.dense1 = nn.Linear(in_features=1024*3*3, out_features=1024)
        self.dense2 = nn.Linear(in_features=1024, out_features=512)
        self.classifier = nn.Linear(in_features=512, out_features=NUM_ATTRIBUTES)

        self.ln1 = nn.LayerNorm(1024)
        self.ln2 = nn.LayerNorm(512)
        
    def forward(self, x):
        x = self.base_model(x) #(32, 512, 1, 1)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.ln1(self.dense1(x)))
        x = F.leaky_relu(self.ln2(self.dense2(x)))
        x = F.softmax(self.classifier(x), dim=1) #won't use nn.CrossEntropy (which expects logits)
        return x
    