import caffe
import numpy as np
model = "FaceRecognition/vgg_face_deploy.prototxt"
weights = "FaceRecognition/face_recog_iter_4000.caffemodel"
mean_file = "FaceRecognition/train_mean.binaryproto"
image_file = "FaceRecognition/me1.jpg"
synset_file = "FaceRecognition/synset_FR.txt"

net = caffe.Net(model,weights,caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

#converting binaryproto into mean npy
data=open(mean_file,'rb').read()
blob=caffe.proto.caffe_pb2.BlobProto()
blob.ParseFromString(data)
mean=np.array(caffe.io.blobproto_to_array(blob))[0,:,:,:]

#caffe.set_mode_gpu()

#image transformation
transformer.set_mean('data',mean.mean(1).mean(1)) # mean is set
transformer.set_raw_scale('data', 255) # normalizes the values in the image based on the 0-255 range
transformer.set_transpose('data', (2,0,1)) # transform an image from (256,256,3) to (3,256,256).

input_image = caffe.io.load_image(image_file)
print "input_image shape"
print input_image.shape

transformed_image = transformer.preprocess('data', input_image)
print "transformed image shape"
print transformed_image.shape

#model prediction
net.blobs['data'].data[...] = transformed_image
output = net.forward()
output_prob = output['prob'][0]

print 'predicted class is:', output_prob.argmax()
label_mapping = np.loadtxt(synset_file, str, delimiter='\t')
user_recognized = label_mapping[output_prob.argmax()]
print "Recognized user: " + user_recognized
