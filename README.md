# NeRF

Neural Radiance Fields, or NeRF as it's popularly known today is a neural network architecture that is used to generate 3D scenes, given 2D images. NeRF was first proposed by Mildenhall et al. in the paper [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934). Here, a basic 10 layer MLP model using sinusoidal positional encoding is used to render a 2-dimensional input image. The final layer has 3 neurons, as we need the values of each of the RGB channel for rendering. 

main.py -> The file to be run

