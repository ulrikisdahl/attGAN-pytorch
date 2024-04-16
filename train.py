import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from statistics import mean
#from model.models import EncoderDecoder, BaseModel, Classifier, Discriminator
from model.models_128 import EncoderDecoder, BaseModel, Classifier, Discriminator
from torchvision.models import vgg16
from dataprocessing.load_dataset import get_data_loader
import matplotlib.pyplot as plt

NUM_EPOCHS=23
#lr = 0.0002
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu" #for debugging (debugging on cuda is pain...)

#lambdas to make the different loss values be the same order of magnitude
lambda1 = 100 
lambda2 = 10

#initalize models
encoder_decoder = EncoderDecoder().to(device)
base_model = BaseModel().to(device)
classifier = Classifier(base_model).to(device)
discriminator = Discriminator(base_model).to(device)
#vgg16_model = vgg16(pretrained=True).to(device).eval()

dataset_path = "/home/ulrik/datasets/celeba/img_align_celeba/img_align_celeba"
data_loader = get_data_loader(dataset_path)

lambda3 = 1

#losses
l1Loss = nn.L1Loss()
binaryCrossEntropy = nn.BCELoss()

def discriminator_loss(real_image_predictions, fake_image_predictions):
    real_image_targets = torch.ones_like(real_image_predictions) - torch.rand(real_image_predictions.shape).to(device)*0.2 #label smoothing
    fake_image_targets = torch.rand(fake_image_predictions.shape).to(device) * 0.2
    real_image_loss = binaryCrossEntropy(real_image_predictions, real_image_targets)
    fake_image_loss = binaryCrossEntropy(fake_image_predictions, fake_image_targets)
    return real_image_loss + fake_image_loss

def discriminator_loss_real(real_image_predictions):
    real_image_targets = torch.ones_like(real_image_predictions) - torch.rand(real_image_predictions.shape).to(device)*0.2 #label smoothing
    real_image_loss = binaryCrossEntropy(real_image_predictions, real_image_targets)
    return real_image_loss

def discriminator_loss_fake(fake_image_predictions):
    fake_iage_targets = torch.rand(fake_image_predictions.shape).to(device) * 0.2
    fake_image_loss = binaryCrossEntropy(fake_image_predictions, fake_iage_targets)
    return fake_image_loss
    
def generator_loss(discriminator_predictions):
    target = torch.ones_like(discriminator_predictions) #the generator always wants the discriminator to predict 1
    return binaryCrossEntropy(discriminator_predictions, target)

def perceptual_loss(input_image, generated_image):

    return


#criterions
optimizerEncoderDecoder=optim.Adam(encoder_decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerClassifier=optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerDiscriminator=optim.SGD(discriminator.parameters(), lr=0.0001) #try to balance the discriminator for training


#load weights
# state_dicts = torch.load('weights/13attr_20e_minibatch_2.pth')
# encoder_decoder.load_state_dict(state_dicts['encoder_decoder'])
# base_model.load_state_dict(state_dicts['base_model'])
# classifier.load_state_dict(state_dicts['classifier'])
# discriminator.load_state_dict(state_dicts['discriminator'])

# optimizerEncoderDecoder.load_state_dict(state_dicts['optimizerEncoderDecoder'])
# optimizerClassifier.load_state_dict(state_dicts['optimizerClassifier'])
# optimizerDiscriminator.load_state_dict(state_dicts['optimizerDiscriminator'])

# #print number of parameters in model
# print(f"Number of parameters in encoder_decoder: {sum(p.numel() for p in encoder_decoder.parameters())}")


#torch.autograd.set_detect_anomaly(True)


def smooth_labels(labels, smoothing=0.2):
    mask = (torch.rand(labels.size()) < 0.5).to(device) 
    smoothed_labels = labels * (1 - smoothing) + 0.5 * smoothing

    # Apply the mask: where mask is 1, use smoothed_labels; where mask is 0, use original labels
    return torch.where(mask, smoothed_labels, labels)


for epoch in range(NUM_EPOCHS):
    encoder_decoder_losses, discriminator_classifier_losses = [], []
    #rec_losses, class_losses, adv_losses = [], [], []
    for i, batch in enumerate(tqdm(data_loader), 0):
        
        input_images = batch[0].to(device)
        attribute_vector_a = batch[1].to(device)
        attribute_vector_b = attribute_vector_a[torch.randperm(attribute_vector_a.size(0))].to(device)

        ### NEW VERSION
        encoder_decoder.zero_grad() #previously accumulated gradients are cleared for generator

        gen_images_a = encoder_decoder(input_images, attribute_vector_a) #generated images conditioned on original attribute vector a
        gen_images_b = encoder_decoder(input_images, attribute_vector_b)

        ### Update EncoderDecoder (generator) ###

        #calculate losses
        disc_predictions_fake = discriminator(gen_images_b)
        adverserial_loss_g = generator_loss(disc_predictions_fake)
        classifier_pred_b = classifier(gen_images_b) 
        classification_loss_b = binaryCrossEntropy(classifier_pred_b, attribute_vector_b)
        reconstruction_loss = l1Loss(gen_images_a, input_images)

        #update
        encoder_decoder_loss = lambda1*reconstruction_loss + lambda2*classification_loss_b + adverserial_loss_g
        encoder_decoder_losses.append(encoder_decoder_loss.item())
        encoder_decoder_loss.backward()
        optimizerEncoderDecoder.step()

        # ### Update discrimnator and classifier ###
        discriminator.zero_grad()
        classifier.zero_grad()

    
        #real images
        # disc_real_predictions = discriminator(input_images)
        # disc_real_loss = discriminator_loss_real(disc_real_predictions)
        # disc_real_loss.backward() #calulate gradients on real batch and fake batch seperately as recommended in GAN hacks
        
        # #fake images
        # disc_fake_predictions = discriminator(gen_images_b.detach()) #detach so that gradients are not backpropagated to the generator - critical!
        # disc_fake_loss = discriminator_loss_fake(disc_fake_predictions)
        # disc_fake_loss.backward()

        disc_real_predictions = discriminator(input_images)
        disc_fake_predictions = discriminator(gen_images_b.detach())
        disc_loss = discriminator_loss(disc_real_predictions, disc_fake_predictions)

        #classifier
        attribute_classifications_a = classifier(input_images) #CHANGED
        classification_loss_a = binaryCrossEntropy(attribute_classifications_a, smooth_labels(attribute_vector_a)) #smooth labels to incourage classifier to predict higher values
        #lassification_loss_a.backward()
    
        discriminator_classifier_loss = lambda3 * classification_loss_a + disc_loss
        discriminator_classifier_losses.append(discriminator_classifier_loss.item())
        discriminator_classifier_loss.backward() 

        optimizerDiscriminator.step()
        optimizerClassifier.step()


        



        ## OLD VERSION

        #update encoder and decoder - important to do in a seperate step from updating discriminator because we dont want the generator to cheat
        # encoder_decoder.zero_grad()
        
        # generated_images_a = encoder_decoder(input_images, attribute_vector_a) #generated images conditioned on original attribute vector a
        # generated_images_b = encoder_decoder(input_images, attribute_vector_b) #generated images conditioned on modified attribute vector b
        # attribute_classifications_b = classifier(generated_images_b)
        # discriminator_predictions = discriminator(generated_images_b)

        # reconstruction_loss = l1Loss(generated_images_a, input_images)
        # classification_generator_loss = binaryCrossEntropy(attribute_classifications_b, attribute_vector_b)
        # generator_adverserial_loss = generator_loss(discriminator_predictions)

        # encoder_decoder_loss = lambda1*reconstruction_loss + lambda2*classification_generator_loss + generator_adverserial_loss
        # encoder_decoder_losses.append(encoder_decoder_loss.item())
        # # rec_losses.append(reconstruction_loss.item())
        # # class_losses.append(classification_generator_loss.item())
        # # adv_losses.append(generator_adverserial_loss.item())

        # encoder_decoder_loss.backward() #
        # optimizerEncoderDecoder.step()


        # #update classifier and discriminator
        # classifier.zero_grad()
        # discriminator.zero_grad()

        # #generated_images_b = encoder_decoder(input_images, attribute_vector_b)
        # attribute_classifications_a = classifier(input_images)
        # discriminator_pred_real = discriminator(input_images)
        # discriminator_pred_fake = discriminator(generated_images_b.detach())
        # classification_classifier_loss = binaryCrossEntropy(attribute_classifications_a, attribute_vector_a)

        # discriminator_adverserial_loss = discriminator_loss(discriminator_pred_real, discriminator_pred_fake)
        # discriminator_classifier_loss = lambda3*classification_classifier_loss + discriminator_adverserial_loss
        # discriminator_classifier_losses.append(discriminator_classifier_loss.item())
        
        # discriminator_classifier_loss.backward()
        # optimizerDiscriminator.step()
        # optimizerClassifier.step()
        
    print(f"Encoder-Decoder loss-{epoch}: {mean(encoder_decoder_losses)}")
    print(f"Discriminator-classifier loss-{epoch}: {mean(discriminator_classifier_losses)}")

#save wegihts
state_dicts = {
    'encoder_decoder': encoder_decoder.state_dict(),
    'base_model': base_model.state_dict(),
    'classifier': classifier.state_dict(),
    'discriminator': discriminator.state_dict(),
}

# Optionally, include the optimizers' state dicts if you plan to continue training later
state_dicts['optimizerEncoderDecoder'] = optimizerEncoderDecoder.state_dict()
state_dicts['optimizerClassifier'] = optimizerClassifier.state_dict()
state_dicts['optimizerDiscriminator'] = optimizerDiscriminator.state_dict()

# Save the consolidated state_dicts to a file
torch.save(state_dicts, 'weights/13attr_23e_128_norm.pth')






#testing
# batch = next(iter(data_loader))
# imgs = batch[0].to(device)
# imgs = (imgs + 1) / 2
# attr = batch[1].to(device)

# attr_perm = attr[torch.randperm(attr.size(0))] #permute the attributes

# with torch.no_grad():
#     gen = encoder_decoder(imgs, attr_perm)

# i = 0
# for img in gen:
#     if i == 5:
#         break
#     img_permuted = torch.permute(img, (1, 2, 0)).cpu()
#     #img_permuted = (img_permuted + 1)/2

#     f, axarr = plt.subplots(1,2)
#     axarr[0].imshow(img_permuted) #generated
#     axarr[1].imshow(torch.permute(imgs[i], (1,2,0)).cpu()) #original
#     plt.show()
    
#     i+=1
        