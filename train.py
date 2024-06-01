import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F
from model.attGAN_model import EncoderDecoder, BaseModel, Classifier, Discriminator
from dataprocessing.load_dataset import get_data_loader
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torch.utils.tensorboard import SummaryWriter
import argparse 


parser = argparse.ArgumentParser(description="attGAN training")
parser.add_argument("--pre_trained", default=None, type=str, help="pretrained model") 
parser.add_argument("--images_path", default=None, type=str, help="path to dataset images")
parser.add_argument("--attributes_path", default=None, type=str, help="path to attribute file")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--num_attributes", default=12, type=int, help="number of attributes")
parser.add_argument("--num_epochs", default=13, type=int, help="number of epochs")
parser.add_argument("--device", default="cuda", type=str, help="device to run on")
args = parser.parse_args()


writer = SummaryWriter('runs/attGAN')


#lambdas to make the different loss values be the same order of magnitude
lambda1 = 30 #perceptual loss (100 for reconstruction loss)
lambda2 = 10
lambda3 = 1

#initalize models
encoder_decoder = EncoderDecoder(batch_size=args.batch_size, num_attributes=args.num_attributes).to(args.device)
base_model = BaseModel().to(args.device)
classifier = Classifier(base_model, args.num_attributes).to(args.device)
discriminator = Discriminator(base_model).to(args.device)

data_loader = get_data_loader(args.images_path, args.attributes_path)


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
    real_image_targets = torch.ones_like(real_image_predictions) - torch.rand(real_image_predictions.shape).to(args.device)*0.2 #label smoothing
    fake_image_targets = torch.rand(fake_image_predictions.shape).to(args.device) * 0.2
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

    perceptual_loss = learned_perceptual_image_patch_similarity(generated_image_adjusted, input_image_adjusted, normalize=True)
    return perceptual_loss


#criterions
optimizerEncoderDecoder=optim.Adam(encoder_decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerClassifier=optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerDiscriminator=optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


class Trainer:
    def __init__(self, encoder_decoder, base_model, classifier, discriminator, train_loader, val_loader, optimizerEncoderDecoder, optimizerClassifier, optimizerDiscriminator, weights=None, device="cpu"):
        self.encoder_decoder = encoder_decoder
        self.base_model = base_model
        self.classifier = classifier
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizerEncoderDecoder = optimizerEncoderDecoder
        self.optimizerClassifier = optimizerClassifier
        self.optimizerDiscriminator = optimizerDiscriminator
        self.device = device

        if weights:
            state_dicts = torch.load(weights)
            self.encoder_decoder.load_state_dict(state_dicts['encoder_decoder'])
            self.base_model.load_state_dict(state_dicts['base_model'])
            self.classifier.load_state_dict(state_dicts['classifier'])
            self.discriminator.load_state_dict(state_dicts['discriminator'])


    def save_weights(self):
        state_dicts = {
            'encoder_decoder': self.encoder_decoder.state_dict(),
            'base_model': self.base_model.state_dict(),
            'classifier': self.classifier.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }

        state_dicts['optimizerEncoderDecoder'] = self.optimizerEncoderDecoder.state_dict()
        state_dicts['optimizerClassifier'] = self.optimizerClassifier.state_dict()
        state_dicts['optimizerDiscriminator'] = self.optimizerDiscriminator.state_dict()

        torch.save(state_dicts, 'weights/gan.pth')

    def train(self, num_epochs):
        #global_iterations = 0
        for epoch in range(num_epochs):
            encoder_decoder_losses, discriminator_classifier_losses = [], []
            for i, batch in enumerate(tqdm(data_loader), 0):

                input_images = batch[0].to(self.device)
                attribute_vector_a = batch[1].to(self.device)
                attribute_vector_b = attribute_vector_a[torch.randperm(attribute_vector_a.size(0))].to(self.device)
                
                gen_images_a = self.encoder_decoder(input_images, attribute_vector_a) #generated images conditioned on original attribute vector a
                gen_images_b = self.encoder_decoder(input_images, attribute_vector_b)

                if i % 4 != 0:
                    ### Update discrimnator and classifier ###
                    self.discriminator.zero_grad()
                    self.classifier.zero_grad()

                    disc_real_predictions = self.discriminator(input_images)
                    disc_fake_predictions = self.discriminator(gen_images_b.detach()) #detach to avoid backpropagating through generator
                    disc_loss = discriminator_loss(disc_real_predictions, disc_fake_predictions)

                    #classifier
                    attribute_classifications_a = self.classifier(input_images) #CHANGED
                    classification_loss_a = F.binary_cross_entropy_with_logits(attribute_classifications_a, attribute_vector_a) #smooth labels to incourage classifier to predict higher values

                    discriminator_classifier_loss = lambda3 * classification_loss_a + disc_loss
                    discriminator_classifier_losses.append(discriminator_classifier_loss.item())
                    discriminator_classifier_loss.backward() 

                    self.optimizerDiscriminator.step()
                    self.optimizerClassifier.step()

                    writer.add_scalar('Loss/Discriminator', disc_loss.item(), epoch * len(data_loader) + i)
                    writer.add_scalar('Loss/Classifier', classification_loss_a.item(), epoch * len(data_loader) + i)

                else:
                    ### Update EncoderDecoder (generator) ###
                    self.encoder_decoder.zero_grad() #previously accumulated gradients are cleared for generator

                    #calculate losses
                    disc_predictions_fake = self.discriminator(gen_images_b)
                    adverserial_loss_g = generator_loss(disc_predictions_fake)
                    classifier_pred_b = self.classifier(gen_images_b) 
                    classification_loss_b = F.binary_cross_entropy_with_logits(classifier_pred_b, attribute_vector_b)
                    #reconstruction_loss = l1Loss(gen_images_a, input_images)
                    perceptual_loss = perceptual_loss_function(input_images, gen_images_a)

                    #update
                    encoder_decoder_loss = lambda1*perceptual_loss + lambda2*classification_loss_b + adverserial_loss_g
                    encoder_decoder_losses.append(encoder_decoder_loss.item())
                    encoder_decoder_loss.backward()
                    self.optimizerEncoderDecoder.step()

                    writer.add_scalar('Loss/Generator', adverserial_loss_g.item(), epoch * len(data_loader) + i)
                    writer.add_scalar('Loss/Perceptual', perceptual_loss.item(), epoch * len(data_loader) + i)
                
            print(f"Encoder-Decoder loss-{epoch}: {mean(encoder_decoder_losses)}")
            print(f"Discriminator-classifier loss-{epoch}: {mean(discriminator_classifier_losses)}")
            
            #save generated images 
            # if epoch % 4 == 0:
            #     with torch.no_grad():
            #         gen_images = encoder_decoder(validation_batch[0].to(device), validation_batch[1].to(device))
            #         gen_images = (gen_images + 1) / 2
            #         writer.add_images(f"Generated Images {epoch}", gen_images, global_step=global_iterations)
            #         global_iterations += 1

            writer.close()


if __name__ == "__main__":

    trainer = Trainer(
        encoder_decoder=encoder_decoder,
        base_model=base_model,
        classifier=classifier,
        discriminator=discriminator,
        train_loader=data_loader,
        val_loader=data_loader,
        optimizerEncoderDecoder=optimizerEncoderDecoder,
        optimizerClassifier=optimizerClassifier,
        optimizerDiscriminator=optimizerDiscriminator,
        device=args.device,
        weights=args.pre_trained
    )
    trainer.train(num_epochs=args.num_epochs)

    trainer.save_weights()


