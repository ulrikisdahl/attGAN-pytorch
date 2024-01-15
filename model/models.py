import torch
import torch.nn as nn
import torch.nn.functional as F

#PARAMS
NUM_ATTRIBUTES = 40 #length of the attribute vector
BATCH_SIZE = 32

#encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        #conv layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2)
        
        #batchnorm inits
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        return x    
    
#decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        #deconv layers
        self.deConv1 = nn.ConvTranspose2d(in_channels=512 + NUM_ATTRIBUTES, out_channels=512, kernel_size=5, stride=2)
        self.deConv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2)
        self.deConv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2)
        self.deConv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.deConv5 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=1)
        
        #batchnorm inits
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
    def forward(self, x, attribute_vec): 
        """
        x shape: (batch_size, channels, height, width)
        attribute_vec shape: (batch_size, num_attributes) 
        """
        #concatenation
        attribute_vec = torch.reshape(attribute_vec, (BATCH_SIZE, NUM_ATTRIBUTES, x.shape[-2], x.shape[-1]))
        x = torch.cat((x, attribute_vec), axis=1) #concatenation with attribute vector along channel axis
        
        #forward propagation
        x = F.leaky_relu(self.bn1(self.deConv1(x)))
        x = F.leaky_relu(self.bn2(self.deConv2(x)))
        x = F.leaky_relu(self.bn3(self.deConv3(x)))
        x = F.leaky_relu(self.bn4(self.deConv4(x)))
        x = torch.tanh(self.deConv5(x))
        return x
        
#base model
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
        #conv layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1)
    
        #batchnorm inits
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        
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
        self.dense1 = nn.Linear(in_features=512*1*1, out_features=1024) # 512*1*1 is latent representation from base_model
        self.discriminator = nn.Linear(in_features=1024, out_features=1)
        
        #instance norm init
        self.in1 = nn.InstanceNorm1d(1024)
        
    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.in1(self.dense1(x)))
        x = F.sigmoid(self.discriminator(x))
        return x

class Classifier(nn.Module):
    def __init__(self, model):
        super(Classifier, self).__init__()
        
        #layers
        self.base_model = model
        self.dense1 = nn.Linear(in_features=512*1*1, out_features=1024)
        self.classifier = nn.Linear(in_features=1024, out_features=NUM_ATTRIBUTES)

        #instance norm init
        self.in1 = nn.InstanceNorm1d(1024)
        
    def forward(self, x):
        x = self.base_model(x) #(32, 512, 1, 1)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.in1(self.dense1(x)))
        x = F.softmax(self.classifier(x), dim=1) #won't use nn.CrossEntropy (which expects logits)
        return x