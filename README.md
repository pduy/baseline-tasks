# baseline-tasks
This is the the baseline classifier used for benchmarking the GAN synthesized data in my Master Thesis at the 
NI group, TU Berlin, Germany. This is an implementation with some modification of the original work from Eitel et al.:

https://arxiv.org/abs/1507.06821

The GANs are implemented in this repository: https://github.com/pduy/pix2pix-tensorflow

The code is not parameterized so in order to run it just simply type:

```
python -u object-recognition/alexnet_generate_features.py
```

And because it is not parameterized, every detail of running it is in the ```main()``` function of 
```alexnet_generate_features.py```

Different experiments are set up in different branches. In all the branches, running the code is via the same command 
above.

In this project, every sub-dataset involved in the experiments is managed through a CSV file, which is later loaded 
as a Pandas dataframe.

In this repository there is also the code for creating the stratified train-test splits used as a replacement for 
random k-fold cross validation. The function is ```lai_et_al_split()```

## master
This is the main reproduction of the paper with some modifications:

* There is no fine-tuning. I find the fine-tuning strategy of the paper not really reasonable
* The missing depth values are interpolated using a method in the NYU Toolbox 
(https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). For implementation details check this repository: 
https://github.com/pduy/nideep

## temp-noise-in-validation
In this branch, noises are added in the validation phase to evaluate the learning behavior. The noises can be added to
each channels of the image before pushing through AlexNet

## dropout-rgb
This branch is an effort to balance the learning distribution between RGB and Depth. Various strategies are implemented:

* Adding dropout to the RGB channels in different levels
* Setting the entire RGB frame to black with a certain probability
* Adding Gaussian Noise to the RGB channels

## stereo-rgb
The classifier implemented in this branch is different in structure than in the other branches. There is no depth 
information involved. The input is a pair of RGB frames which are the rotated pose of the other. This is to evaluate 
the Pose-GAN implemented in https://github.com/pduy/pix2pix-tensorflow.
