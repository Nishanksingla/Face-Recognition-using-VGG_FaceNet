import caffe
import numpy as np
import cv2
import sys

import matplotlib
matplotlib.use('Agg')


model = "FaceRecognition/vgg_face_deploy.prototxt"
weights = "FaceRecognition/face_recog_iter_4000.caffemodel"
mean_file = "FaceRecognition/train_mean.binaryproto"
image_file = "me1.jpg"
synset_file = "FaceRecognition/synset_FR.txt"

camera_arg = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

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


# Open stream
#stream = cv2.VideoCapture(0)

stream = cv2.VideoCapture("ZacEfrom.mp4")
#stream = cv2.VideoCapture(camera_arg)

face_cascade = cv2.CascadeClassifier("/usr/local/Cellar/opencv3/3.2.0/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")

if not stream.isOpened():
    print("Could not open camera..")
    sys.exit(1)

#loading labels text
label_mapping = np.loadtxt(synset_file, str, delimiter='\t')

semaphore = False

while semaphore == False:
    (grabbed, frame) = stream.read()
    if grabbed == False:
        print "Error, unable to grab a frame from the camera"
        quit()

    #detect the face over here

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5,minSize=(80, 80))

    if len(faces)<=0:
        cv2.imshow('frame',frame)
        continue

    x,y,w,h = faces[0]

    #drawing rectangle
    cv2.rectangle(frame,(x,y),(x+w-20,y+h+50),(255,0,0),2)

    #getting detected face
    detected_face = frame[y:y+h, x:x+w]
#    print "Detected face shape"
#    print detected_face.shape

    #transforming image
    transformed_image = transformer.preprocess('data', detected_face)
#    print "transformed image shape"
#    print transformed_image.shape

    #model prediction
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    output_prob = output['prob'][0]

#    print "output probability: "
#    print output_prob

    #predicted class
    predicted_class = output_prob.argmax()
    print '\n Predicted class is:', predicted_class

    #getting name of the user

    user_recognized = label_mapping[predicted_class]
    print "Recognized user: " + user_recognized

    cv2.putText(frame,user_recognized,(x+w-120,y), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        semaphore = True
stream.release()    
cv2.destroyAllWindows()









