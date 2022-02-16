# GAN_bed_image
### Copyright 2022 Sang Wook Kim.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Purpose
Using Generative adversarial network, build generative model with Tensorflow Keras for 32X32 pixel image of beds.

## Data Source
Kaggle.com - Datasets - Furniture detector
Source : https://www.kaggle.com/akkithetechie/furniture-detector
License : CC0: Public Domain

## Techniques
### General strategy
 - Basic image process, Generate data set 
 - Build Generator which takes random variable of 128 dimension (noise dimension) and generate 'fake' images
 - Build Discriminator which is trained by real images and distinguish 'real' and 'fake' images
 - Train and save figures

### Specification
 - Generator: Expand input noise to 4 * 4 * 256 dense layer -> Two deep 2D convolution layers -> 32X32 pixel image
 - Discriminator: Take 32X32 image -> Four 2D convolution layers -> One dimensional layer which determine 'real' or 'fake'
 - Training model: Wasserstein GAN (WGAN) with Gradient Penalty (GP). 
 - Model optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
 - Model losses = tf.reduce_mean(img)
 - Hyperparameters = Batch size 10, epochs 100

## Results
