import torch
import torch.nn as nn
import torch.nn.functional as F

#PARAMS
NUM_ATTRIBUTES = 12 #length of the attribute vector
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        #conv layers (encoder)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1)

        #deconv layers (decoder)
        self.deConv1 = nn.ConvTranspose2d(in_channels=1024 + NUM_ATTRIBUTES, out_channels=1024, kernel_size=4, stride=2, padding=1)
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
        attribute_vec = torch.reshape(attribute_vec, (BATCH_SIZE, NUM_ATTRIBUTES, 1, 1))
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
    def __init__(self, model):
        """
        Args:
            model: base model
        """
        super(Classifier, self).__init__()
        
        #layers
        self.base_model = model
        self.dense1 = nn.Linear(in_features=1024*3*3, out_features=1024)
        self.classifier = nn.Linear(in_features=1024, out_features=NUM_ATTRIBUTES)
        
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
    







##################################################################################################
class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, embedding_dim, attribute_dim, upsample):
        super().__init__()
        self.upsample = upsample


        #up or down sampling
        if not upsample:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels=2*input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1) #double the expected input channels because of concatenation
            self.rescale = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels // 2, kernel_size=2, stride=2) #doubles the resolution
            self.bn3 = nn.BatchNorm2d(output_channels // 2)

        if input_channels == 512:
            extra = NUM_ATTRIBUTES
        else:
            extra = 0

        self.conv2 = nn.Conv2d(in_channels=output_channels + extra, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x, attr=None):
        x = torch.relu(self.bn1(self.conv1(x)))
        
        if attr != None:
            attr = torch.reshape(attr, (BATCH_SIZE, attr.shape[-1], 1, 1))
            attr = attr.repeat(1, 1, x.shape[-2], x.shape[-1])

            x = torch.cat((x, attr), axis=1)
        x = torch.relu(self.bn2(self.conv2(x)))
        if self.upsample:
            x = torch.relu(self.bn3(self.rescale(x)))
        return x

class BasicUNet(nn.Module):
    """
        Classical U-Net with only positional embeddings
    """
    def __init__(self):
        super().__init__()

        self.image_channels = 3
        self.downsample_channels = [self.image_channels, 64, 128, 256, 512]
        self.upsample_channels = self.downsample_channels[::-1]
        self.embedding_dim = 30
        self.output_channels = 3
        self.attr_dim = 13

        self.downsample_blocks = nn.ModuleList([])
        for idx in range(len(self.downsample_channels) - 1): 
            self.downsample_blocks.append(ConvBlock(
                input_channels=self.downsample_channels[idx], 
                output_channels=self.downsample_channels[idx + 1], 
                embedding_dim=self.embedding_dim,
                attribute_dim=self.attr_dim,
                upsample=False))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle_conv1 = nn.Conv2d(in_channels=self.downsample_channels[-1], out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.middle_conv2 = nn.Conv2d(in_channels=1024 + NUM_ATTRIBUTES, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.middle_upsample = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.middel_layer_embedder = nn.Linear(in_features=self.embedding_dim, out_features=1024)
        self.middle_attr = nn.Linear(in_features=self.attr_dim, out_features=1024)
        self.bn_middle_1 = nn.BatchNorm2d(1024)
        self.bn_middle_2 = nn.BatchNorm2d(1024)
            
        self.upsample_blocks = nn.ModuleList([])
        for idx in range(len(self.upsample_channels) - 2):
            self.upsample_blocks.append(ConvBlock(
                input_channels=self.upsample_channels[idx], 
                output_channels=self.upsample_channels[idx], 
                embedding_dim=self.embedding_dim, 
                attribute_dim=self.attr_dim,
                upsample=True))
        self.upsample_blocks.append(
            ConvBlock(input_channels=2*64, output_channels=64, embedding_dim=self.embedding_dim, attribute_dim=self.attr_dim, upsample=False) #last block without upsampling
        )
            
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=self.output_channels, kernel_size=1)

    def forward(self, x, attr):

        skip_connections = []
        for downsample_block in self.downsample_blocks:
            x = downsample_block(x)
            skip_connections.append(x) 
            x = self.maxpool(x) 

        x = torch.relu(self.bn_middle_1(self.middle_conv1(x)))

        a = torch.reshape(attr, (BATCH_SIZE, attr.shape[-1], 1, 1))
        a = a.repeat(1, 1, x.shape[-2], x.shape[-1])

        x = torch.cat((x, a), axis=1)
        
        x = torch.relu(self.bn_middle_2(self.middle_conv2(x)))
        x = torch.relu(self.middle_upsample(x))

        for idx, upsample_block in enumerate(self.upsample_blocks):
            x = torch.cat((x, skip_connections[-idx - 1]), dim=1)
            if idx == 0:
                x = upsample_block(x, attr)
            else:
                x = upsample_block(x)

        x = self.out_conv(x) #IDEA: Add tanh activation - funket dårlig
        return x