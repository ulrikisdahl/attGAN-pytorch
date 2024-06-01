import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    def __init__(self, batch_size, num_attributes):
        super(EncoderDecoder, self).__init__()
        self.num_attributes = num_attributes   
        self.batch_size = batch_size

        #conv layers (encoder)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)

        #deconv layers (decoder)
        self.deConv1 = nn.ConvTranspose2d(in_channels=1024 + self.num_attributes, out_channels=1024, kernel_size=4, stride=2, padding=1)
        self.deConv2 = nn.ConvTranspose2d(in_channels=1024 + 512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.deConv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deConv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deConv5 = nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1)

        #batchnorm inits
        self.bnEnc1 = nn.BatchNorm2d(128)
        self.bnEnc2 = nn.BatchNorm2d(256)
        self.bnEnc3 = nn.BatchNorm2d(512)
        self.bnEnc4 = nn.BatchNorm2d(512)
        self.bnEnc5 = nn.BatchNorm2d(1024)

        self.bnDec1 = nn.BatchNorm2d(1024)
        self.bnDec2 = nn.BatchNorm2d(512)
        self.bnDec3 = nn.BatchNorm2d(256)
        self.bnDec4 = nn.BatchNorm2d(128)

    def forward(self, x, attribute_vec):
        """
        Args:
            x: image tensor of shape [batch_size, 3, 128, 128]
            attribute_vec: attribute vector of shape [batch_size, NUM_ATTRIBUTES]
        """

        #downsample
        x1 = F.leaky_relu(self.bnEnc1(self.conv1(x)))
        x2 = F.leaky_relu(self.bnEnc2(self.conv2(x1)))
        x3 = F.leaky_relu(self.bnEnc3(self.conv3(x2)))
        x4 = F.leaky_relu(self.bnEnc4(self.conv4(x3))) 
        z = F.leaky_relu(self.bnEnc5(self.conv5(x4)))

        #concatenation in the latent space
        attribute_vec = torch.reshape(attribute_vec, (self.batch_size, self.num_attributes, 1, 1))
        attribute_vec = attribute_vec.repeat(1, 1, z.shape[-2], z.shape[-1])
        x = torch.cat((z, attribute_vec), axis=1) #concatenation operation along channel axis

        #upsample
        x = F.leaky_relu(self.bnDec1(self.deConv1(x)))
        x = torch.cat((x, x4), dim=1) #single skip connection
        x = F.leaky_relu(self.bnDec2(self.deConv2(x)))
        x = F.leaky_relu(self.bnDec3(self.deConv3(x)))
        x = F.leaky_relu(self.bnDec4(self.deConv4(x)))
        x = torch.tanh(self.deConv5(x))
        return x

        
#base model
class BaseModel(nn.Module):
    def __init__(self):
        """
        Base model for sharing weights between discriminator and classifier
        """
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
        """
        Args:
            x: image tensor of shape [batch_size, 3, 128, 128]
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        return x
    
class Discriminator(nn.Module):
    def __init__(self, model):
        """
        Args:
            model: base model  
        """
        super(Discriminator, self).__init__() 
        
        #layers
        self.base_model = model
        self.dense1 = nn.Linear(in_features=1024*3*3, out_features=1024) # 512*1*1 is latent representation from base_model
        self.discriminator = nn.Linear(in_features=1024, out_features=1)
        
    def forward(self, x):
        """
        Args:
            x: feature map from base model of shape [batch_size, 1024, 3, 3]
        """
        x = self.base_model(x)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.dense1(x))
        x = self.discriminator(x) #return logits since we use F.binary_cross_entropy_with_logits
        return x

class Classifier(nn.Module):
    def __init__(self, model, num_attributes):
        """
        Args:
            model: base model
        """
        super(Classifier, self).__init__()
        
        #layers
        self.base_model = model
        self.num_attributes = num_attributes
        self.dense1 = nn.Linear(in_features=1024*3*3, out_features=1024)
        self.classifier = nn.Linear(in_features=1024, out_features=self.num_attributes)
        
    def forward(self, x):
        """
        Args:
            x: image tensor of shape [batch_size, 3, 128, 128]
        """
        x = self.base_model(x) 
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.dense1(x))
        x = self.classifier(x)
        return x
    
