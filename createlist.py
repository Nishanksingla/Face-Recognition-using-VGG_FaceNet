import os, sys
files = os.listdir("Testing_data")
names_list=[]
num = 0;
f = open('test_0.txt', 'w')
for file in files:
    print file
    label = file.split("-")[0].split("_")[1].lstrip("0")
    number = file.split(
    print label
    label = int(label) - 1
    f.write("Testing_data/"+file+" "+str(label)+"\n")
