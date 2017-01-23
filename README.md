# Face-Recognition-using-VGG_FaceNet
Trained a VGG net for face recognition.

## Dataset
Dataset has images of 84 individuals which inlcudes faces of 83 celebrities and myself. Training set has 8770 images and 
testing set has 840 images. 
Dataset Link: http://www.briancbecker.com/blog/research/pubfig83-lfw-dataset/

##Model
I have used VGG Net which includes 13 convolutional layers, 3 fully connected layer and ReLu,Max Pooling, Dropout layers in between. 

##Training
For training, I have used Transfer Learning to train my network and to achieve more accuracy in less iterations. In the training, 
I have used the pre-trained weights i.e trained caffemodel on the same network/model but on different dataset.
 
To use Transfer Learning,I have execute the following command from Caffe root folder to train the model.

build/tools/caffe train --solver = face_recognition/solver.prototxt --weights = face_recognition/VGG_FACE.caffemodel

Note: Below in References section I have included the link to trained caffemodel.

##Output/Accuracy



##References
http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

##Deep Learning Platform Used
Caffe
