import os, sys
files = os.listdir("Testing_data")
names_list=[]
num = 0;
f = open('newtest.txt', 'w')
for file in files:
#    print file
    filenameSplit  = file.split("-")
    label = filenameSplit[0].split("_")[1].lstrip("0")
    print "label"
    print label
    
    imageNum = filenameSplit[1].split(".")[0].lstrip("0")
    print "imageNum "
    print imageNum
    
    if imageNum != "" and int(imageNum) <=15:
        label = int(label) - 1
        f.write("Testing_data/"+file+" "+str(label)+"\n")
