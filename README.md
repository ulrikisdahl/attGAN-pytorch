# attGAN face attributes
Change attributes of face images 

## Resources

- Paper: https://arxiv.org/abs/1711.10678
- Dataset (Celeba) 

## Results

Some examples of facial attributes that can be changed using the model (left is original image, center is image recreated without attribute changes, right is image with changed attributes)

Add eyeglasses:

![eyeglasses](/static/add-eyeglasses.png)

Change hair color:

![blonde hair](/static/change-hair.png)

Change gender:

![gender](/static/change-gender.png)


## Deviations from paper
There are two major deviations from the paper in this implementation
 1. Perceptual loss introduced to replace the reconstruction loss for the generator loss term. This helped improve the realism of the images and reduced background blur.
 2. The paper reccomends skip-connections in a U-Net-like architecture for the generator, however I found it worked best to only have a single skip-connection between the two lowest resolution layers.

## How to run

Train attGAN: 

``
    python3 train.py --images_path=[/path/to/dataset_images] --attribute_path=[/path/to/attributes.csv]
``

Run inference with attGAN:

```python3 inferece.py --pre_trained=[/path/to/model_weights] --images_path=[/path/to/dataset_images] --attributes_path=[/path/to/attributes.csv]``` 



## Directory
Models can be found inside ```model``` directory

Dataloader can be found inside ```dataprocessing``` directory