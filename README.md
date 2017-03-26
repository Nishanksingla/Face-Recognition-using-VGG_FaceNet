# Face-Recognition-using-VGG_FaceNet
Trained a VGG net for face recognition.

## Dataset
Dataset has images of 84 individuals which includes faces of 83 celebrities and myself. Training set has 8770 images and the testing set has 840 images in total. 
Dataset Link: http://www.briancbecker.com/blog/research/pubfig83-lfw-dataset/

## Model
I have used VGG Net which includes 13 convolutional layers, 3 fully connected layers, and ReLu, Max-Pooling, Dropout layers in between. 

## Training
For training, I have used Transfer Learning to train the network and to achieve more accuracy in fewer iterations. In the training, 
I have used the pre-trained weights instead of initializing the weights randomly i.e trained caffemodel on the same network/model but on the different dataset.
 
To use Transfer Learning, I have executed the following command from Caffe root folder to train the model.

build/tools/caffe train --solver = face_recognition/solver.prototxt --weights = face_recognition/VGG_FACE.caffemodel

Note: Below in References section, I have included the link to pre-trained caffemodel.

## Output/Accuracy
After running for 4000 iterations I achieved the accuracy of 95.5% which is pretty good and very close to the accuracy of Facebook Face Recognition model which uses billions of images to train their network.

"Output ScreenShots" folder includes few snapshots that show correct classifications done by the newly trained network.

Below is the command I executed on new images of celebrities for classification using the newly trained model.

build/examples/cpp_classification/classification.bin face_recognition/vgg_face_deploy.prototxt face_recognition/face_recog_iter_4000.caffemodel face_recognition/train_mean.binaryproto face_recognition/synset_FR.txt face_recognition/Testing_data/test_000082-000018.jpg

## References
http://www.robots.ox.ac.uk/~vgg/software/vgg_face/

## Deep Learning Platform Used
Caffe
