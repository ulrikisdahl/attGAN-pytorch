import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
from model.models import Encoder, Decoder, BaseModel, Classifier, Discriminator
from dataprocessing import load_dataset

NUM_EPOCHS=1
lr = 0.0002
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#lambdas to make the different loss values be the same order of magnitude
lambda1 = 100 
lambda2 = 10

#initalize models
encoder = Encoder().to(device)
decoder = Decoder().to(device)
base_model = BaseModel().to(device)
classifier = Classifier(base_model).to(device)
discriminator = Discriminator(base_model).to(device)

data_loader = load_dataset("")

lambda3 = 1

#losses
l1Loss = nn.L1Loss()
binaryCrossEntropy = nn.BCELoss()

def discriminator_loss(real_image_predictions, fake_image_predictions):
    real_image_targets = torch.ones_like(real_image_predictions) #optimize?
    fake_image_targets = torch.zeros_like(fake_image_predictions)
    real_image_loss = binaryCrossEntropy(real_image_predictions, real_image_targets)
    fake_image_loss = binaryCrossEntropy(fake_image_predictions, fake_image_targets)
    return real_image_loss + fake_image_loss
    
def generator_loss(discriminator_predictions):
    target = torch.ones_like(discriminator_predictions) #the generator always wants the discriminator to predict 1
    return binaryCrossEntropy(discriminator_predictions, target)


#criterions
optimizerEncoder = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerDecoder = optim.Adam(decoder.parameters(), lr=lr)
optimizerClassifier=optim.Adam(classifier.parameters(), lr=lr)
optimizerDiscriminator=optim.Adam(discriminator.parameters(), lr=lr)


for epoch in range(NUM_EPOCHS):
    encoder_decoder_losses, discriminator_classifier_losses = [], []
    for i, batch in enumerate(tqdm(data_loader), 0):
          
        input_images = batch[0].to(device)
        attribute_vector_a = batch[1].to(device)
        attribute_vector_b = attribute_vector_a[torch.randperm(attribute_vector_a.size(0))].to(device)
        
        #update encoder and decoder
        encoder.zero_grad()
        decoder.zero_grad()
        
        z = encoder(input_images) #latent representation from the encoder
        generated_images_a = decoder(z, attribute_vector_a) #generated images conditioned on original attribute vector a
        generated_images_b = decoder(z, attribute_vector_b) #generated images conditioned on modified attribute vector b
        attribute_classifications_b = classifier(generated_images_b)
        discriminator_predictions = discriminator(generated_images_b)
        
        reconstruction_loss = l1Loss(generated_images_a, input_images)
        classification_generator_loss = binaryCrossEntropy(attribute_classifications_b, attribute_vector_b)
        generator_adverserial_loss = generator_loss(discriminator_predictions)
        
        encoder_decoder_loss = lambda1*reconstruction_loss + lambda2*classification_generator_loss + generator_adverserial_loss
        encoder_decoder_losses.append(encoder_decoder_loss.item())

        encoder_decoder_loss.backward(retain_graph=True) #retain the graph for discriminator classifier autograd — more memory but less compute needed
        optimizerEncoder.step() 
        optimizerDecoder.step()
        
        #update classifier and discriminator
        classifier.zero_grad()
        discriminator.zero_grad()
        
        attribute_classifications_a = classifier(input_images)
        discriminator_pred_real = discriminator(input_images)
        discriminator_pred_fake = discriminator(generated_images_b)
        
        classification_classifier_loss = binaryCrossEntropy(attribute_classifications_a, attribute_vector_a)
        discriminator_adverserial_loss = discriminator_loss(discriminator_pred_real, discriminator_pred_fake)
        
        discriminator_classifier_loss = lambda3*classification_classifier_loss + discriminator_adverserial_loss
        discriminator_classifier_losses.append(discriminator_classifier_loss.item())
        
        discriminator_classifier_loss.backward()
        optimizerDiscriminator.step()
        optimizerClassifier.step()
        
    print(f"Encoder-Decoder loss-{epoch}: {mean(encoder_decoder_losses)}")
    print(f"Discriminator-classifier loss-{epoch}: {mean(discriminator_classifier_losses)}")
        