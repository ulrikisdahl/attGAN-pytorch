import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F
from model.models_128 import EncoderDecoder, BaseModel, Classifier, Discriminator
from dataprocessing.load_dataset import get_data_loader
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('runs/attGAN')


#lambdas to make the different loss values be the same order of magnitude
lambda1 = 30 #perceptual loss
lambda2 = 10
lambda3 = 1

#initalize models
encoder_decoder = EncoderDecoder().to(device)
base_model = BaseModel().to(device)
classifier = Classifier(base_model).to(device)
discriminator = Discriminator(base_model).to(device)

dataset_path = "/home/ulrik/datasets/celeba/img_align_celeba/img_align_celeba"
data_loader = get_data_loader(dataset_path)


#losses
l1Loss = nn.L1Loss()
binaryCrossEntropy = nn.BCELoss()

def discriminator_loss(real_image_predictions, fake_image_predictions):
    """
    Adverserial loss for discriminator with smoothing - the discriminator wants to predict 1 for real images and 0 for fake images
    Args:
        real_image_predictions: predictions for real images of shape (batch_size, 1)
        fake_image_predictions: predictions for fake images of shape (batch_size, 1)
    """
    real_image_targets = torch.ones_like(real_image_predictions) - torch.rand(real_image_predictions.shape).to(device)*0.2 #label smoothing
    fake_image_targets = torch.rand(fake_image_predictions.shape).to(device) * 0.2
    real_image_loss = F.binary_cross_entropy_with_logits(real_image_predictions, real_image_targets)
    fake_image_loss = F.binary_cross_entropy_with_logits(fake_image_predictions, fake_image_targets)
    return real_image_loss + fake_image_loss

def generator_loss(discriminator_predictions):
    """
    Adverserial loss for generator - the generator wants the discriminator to predict 1 for fake images
    Args:
        discriminator_predictions: predictions for fake images of shape (batch_size, 1)
    """
    target = torch.ones_like(discriminator_predictions) #the generator always wants the discriminator to predict 1
    return F.binary_cross_entropy_with_logits(discriminator_predictions, target)

def perceptual_loss_function(input_image, generated_image):
    """
    Perceptual loss - computes the distance between the input image and the generated image using a pretrained VGG16 model
    """
    input_image_adjusted = (input_image + 1) / 2
    generated_image_adjusted = (generated_image + 1) / 2

    #with torch.no_grad():
    perceptual_loss = learned_perceptual_image_patch_similarity(generated_image_adjusted, input_image_adjusted, normalize=True)
    return perceptual_loss


#criterions
optimizerEncoderDecoder=optim.Adam(encoder_decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerClassifier=optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))
#optimizerDiscriminator=optim.SGD(discriminator.parameters(), lr=0.0001) #try to balance the discriminator for training
optimizerDiscriminator=optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#load weights
state_dicts = torch.load('weights/perceptual_128_blonde_11e.pth')
encoder_decoder.load_state_dict(state_dicts['encoder_decoder'])
base_model.load_state_dict(state_dicts['base_model'])
classifier.load_state_dict(state_dicts['classifier'])
discriminator.load_state_dict(state_dicts['discriminator'])

optimizerEncoderDecoder.load_state_dict(state_dicts['optimizerEncoderDecoder'])
optimizerClassifier.load_state_dict(state_dicts['optimizerClassifier'])
optimizerDiscriminator.load_state_dict(state_dicts['optimizerDiscriminator'])


validation_batch = next(iter(data_loader))
NUM_EPOCHS = 2
global_iterations = 0
for epoch in range(NUM_EPOCHS):
    encoder_decoder_losses, discriminator_classifier_losses = [], []
    for i, batch in enumerate(tqdm(data_loader), 0):

        input_images = batch[0].to(device)
        attribute_vector_a = batch[1].to(device)
        attribute_vector_b = attribute_vector_a[torch.randperm(attribute_vector_a.size(0))].to(device)
        
        gen_images_a = encoder_decoder(input_images, attribute_vector_a) #generated images conditioned on original attribute vector a
        gen_images_b = encoder_decoder(input_images, attribute_vector_b)

        if i % 4 != 0:
            ### Update discrimnator and classifier ###
            discriminator.zero_grad()
            classifier.zero_grad()

            disc_real_predictions = discriminator(input_images)
            disc_fake_predictions = discriminator(gen_images_b.detach()) #detach to avoid backpropagating through generator
            disc_loss = discriminator_loss(disc_real_predictions, disc_fake_predictions)

            #classifier
            attribute_classifications_a = classifier(input_images) #CHANGED
            classification_loss_a = F.binary_cross_entropy_with_logits(attribute_classifications_a, attribute_vector_a) #smooth labels to incourage classifier to predict higher values

            discriminator_classifier_loss = lambda3 * classification_loss_a + disc_loss
            discriminator_classifier_losses.append(discriminator_classifier_loss.item())
            discriminator_classifier_loss.backward() 

            optimizerDiscriminator.step()
            optimizerClassifier.step()

            writer.add_scalar('Loss/Discriminator', disc_loss.item(), epoch * len(data_loader) + i)
            writer.add_scalar('Loss/Classifier', classification_loss_a.item(), epoch * len(data_loader) + i)

        else:
            ### Update EncoderDecoder (generator) ###
            encoder_decoder.zero_grad() #previously accumulated gradients are cleared for generator

            #calculate losses
            disc_predictions_fake = discriminator(gen_images_b)
            adverserial_loss_g = generator_loss(disc_predictions_fake)
            classifier_pred_b = classifier(gen_images_b) 
            classification_loss_b = F.binary_cross_entropy_with_logits(classifier_pred_b, attribute_vector_b)
            #reconstruction_loss = l1Loss(gen_images_a, input_images)
            perceptual_loss = perceptual_loss_function(input_images, gen_images_a)

            #update
            encoder_decoder_loss = lambda1*perceptual_loss + lambda2*classification_loss_b + adverserial_loss_g
            encoder_decoder_losses.append(encoder_decoder_loss.item())
            encoder_decoder_loss.backward()
            optimizerEncoderDecoder.step()

            writer.add_scalar('Loss/Generator', adverserial_loss_g.item(), epoch * len(data_loader) + i)
            writer.add_scalar('Loss/Perceptual', perceptual_loss.item(), epoch * len(data_loader) + i)
        
    #save generated images 
    if epoch % 4 == 0:
        with torch.no_grad():
            gen_images = encoder_decoder(validation_batch[0].to(device), validation_batch[1].to(device))
            gen_images = (gen_images + 1) / 2
            writer.add_images(f"Generated Images {epoch}", gen_images, global_step=global_iterations)
            global_iterations += 1

    if epoch + 1 == 100:
        state_dicts_8 = {
            'encoder_decoder': encoder_decoder.state_dict(),
            'base_model': base_model.state_dict(),
            'classifier': classifier.state_dict(),
            'discriminator': discriminator.state_dict(),
            }

        state_dicts_8['optimizerEncoderDecoder'] = optimizerEncoderDecoder.state_dict()
        state_dicts_8['optimizerClassifier'] = optimizerClassifier.state_dict()
        state_dicts_8['optimizerDiscriminator'] = optimizerDiscriminator.state_dict()

        torch.save(state_dicts_8, 'weights/perceptual_128_blonde_11e.pth')

        
    print(f"Encoder-Decoder loss-{epoch}: {mean(encoder_decoder_losses)}")
    print(f"Discriminator-classifier loss-{epoch}: {mean(discriminator_classifier_losses)}")

writer.close()


#save wegihts
state_dicts = {
    'encoder_decoder': encoder_decoder.state_dict(),
    'base_model': base_model.state_dict(),
    'classifier': classifier.state_dict(),
    'discriminator': discriminator.state_dict(),
}

state_dicts['optimizerEncoderDecoder'] = optimizerEncoderDecoder.state_dict()
state_dicts['optimizerClassifier'] = optimizerClassifier.state_dict()
state_dicts['optimizerDiscriminator'] = optimizerDiscriminator.state_dict()

torch.save(state_dicts, 'weights/perceptual_128_blonde_13e.pth')
