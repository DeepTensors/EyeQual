# EyeQual
Given a Retinal Image, determine its quality( high or low)
## Usage 
To train the network

python train.py --pooling WAP -b 4 -e 10 -imsize 128
###Options
pooling - WAP/SWAP (Weighted Average Pooling/ Shifted Weighted Average Pooling)
b      - batch size
e      - epochs
imsize - image size 

To test an image with the saved model

python test.py -m models/weights.eyeQual_128.hdf5 -i img.jpg -imsize 128
###Options
m      - saved model you wish to test the image on
i      - image
imsize - image size

# Visualizing a feature map
For visualzing a feature map for any layer in the netowrk, you just need to create an object of the class VisualizeActivation giving the layer for which you want to visualize the feature map and the model on which the image is being tested.

vis_obj = VisualizeActivation(17,model)    -> for visualizing the 17th layer output \
Next you need to call the visualize_feature_maps function with the image as the parameter (the same way a test image is fed into the network for testing) \
vis_obj.visualize_feature_maps(img) \

