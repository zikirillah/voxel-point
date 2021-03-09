# voxel-point: 3D object classification and segmentation using convolutional neural network

## Prerequisites:
Python3 (with necessary common libraries such as numpy, scipy, etc.)
TensorFlow(1.8 or other)
You can prepare your data in *.npy file or other,here I use .npy file:
.npy file include : x y z label.
# data preparation
you can use convert-to-mat file to convert your file to .mat files
store your training and testing data in /data directory.


## Train:
```
 python code/train.py
 ```

## Test:
 ```
python code/test.py
```
