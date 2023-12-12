# This program builds upon the image manipulation in Find_Ends.py and Monitor_Ends.py to perform the defect detection
# on the product.

import os, time
import numpy
import cv2
from PIL import Image, ImageEnhance
from keras.models import model_from_json
from keras.optimizers import RMSprop
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import smtplib

global scoreStr_N
global scoreStr_S
global fileName

#Variables for high-pass filter
kernel = numpy.array([[-1,-1,-1],
		      [-1,9.25,-1],
		      [-1,-1,-1]])
anchor = (-1,-1)
delta=0
ddepth=-1

# Sub process for predicting the score for the sub-image. The sub-image (img) is passed through the neural network,
# and the overall image (resultImg) has colored rectangles placed on it to alert operators to areas of concern.
def predictImg(img,x1,x2,y1,y2):
	global resultImg
	global scoreStr_S
	global scoreStr_N
	global fileName
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img,(100,100))
	img = img.reshape(-1,100,100,1)
	img = img.astype('float32')
	img = img/255
	score = loaded_model.predict(img, batch_size = None, verbose = 0, steps = None)
	if "_N.jpg" in fileName:
		scoreStr_N = scoreStr_N + str(numpy.squeeze(score)) + ","
	else:
		scoreStr_S = scoreStr_S + str(numpy.squeeze(score)) + ","
	if score<0.5:
		red=255
		green = int(510*score)
	else:
		red = int((-510*score)+510)
		green = 255
	color = tuple((0,green,red))
	cv2.rectangle(resultImg,(x1,y1),(x2,y2),color,2)
	return resultImg

# Save results to a network file for monitoring and troubleshooting, and to monitor performance over time. An obvious
# defect is bad, but a general decrease in quality is also bad.
def updateResults(resultFile,scoreStr):
	with open(resultFile,"r") as input:
		lines = input.readlines()
	with open(resultFile,"w") as output:
		if len(lines)>4:
			lines = lines[1:5]
		for line in lines:
			output.write(line.strip("\n") + "\n")
		output.write(scoreStr + "\n")

# Open up the results file and look at previous results. If average of these previous results is less than a threshold,
# then alert operator to a possible problem.
def checkFile(resultFile,productLocation):
	with open(resultFile,"r") as input:
		lines = input.readlines()
	dataArray = []
	for line in lines:
		dataArray.append(line.split(',')[0:40])
	dataArray = numpy.array(dataArray)
	dataArray = dataArray.astype(numpy.float)
	aveValues = dataArray.sum(axis=0)/5
	scoreThresh = 0.4
	if not all(i >= scoreThresh for i in aveValues):
		boxIndex = [i for i in range(1,41)]
		defectIndex = [j for (i,j) in zip(aveValues,boxIndex) if i < scoreThresh]
		defectValues = [aveValues[i-1] for i in defectIndex]
		print("Possible defect on " + productLocation + " side")
	else:
		print("Results for " + productLocation + " side are OK")

# Load the neural network model
json_file = open('/home/pi/Programming/NN_Models/2Class_Smaller_Model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('/home/pi/Programming/NN_Models/2Class_Smaller_Model.h5')
loaded_model.compile(loss = 'binary_crossentropy',optimizer = RMSprop(lr=0.000001),metrics=['accuracy'])

# Watch the network folder for any new images from the Monitor_Ends.py program
resultTime = None
path_to_watch = '/home/pi/DRIVE/Ends_Lines'
before = dict([(f,None) for f in os.listdir(path_to_watch)])
print("Watching...")
while True:
	time.sleep(5)
	after = dict([(f,None) for f in os.listdir(path_to_watch)])
	added = [f for f in after if not f in before]
	if added:
		for fileName in added:
			img = cv2.imread(path_to_watch + "/" + fileName)
			if img is not None:
				height,width,_ = img.shape
				resultImg = img.copy()

				#Process image with high-pass filter
				img = cv2.filter2D(img,ddepth,kernel,anchor)
				img = img + 30*numpy.ones(img.shape,numpy.uint8)
				img = cv2.medianBlur(img,5)
				img[numpy.where(img <= [30])] = [255]

				# Take the image dimension information, determined from the Monitor_Ends.py program, to split the
				# image into sub-images again. Each sub-image is then fed through the neural network to determine
				# the score, which is the likelihood that a defect exists.
				imgDims = fileName.rsplit('-(')[1].rsplit(')')[0].rsplit(',')
				imgDims = [int(i) for i in imgDims] #[topWidth,bottomWidth,leftWidth,rightWidth]
				imgIndex = 1
				scoreStr=""
				scoreStr_N = ""
				scoreStr_S = ""
				for i in range(0,10):
					y1=0
					y2=imgDims[0]
					x1=i*100
					x2=(i+1)*100
					sectImg = img[y1:y2,x1:x2]
					resultImg = predictImg(sectImg,x1,x2,y1,y2)
				for i in range(0,10):
					y1 = i*100
					y2 = (i+1)*100
					x1 = width-imgDims[3]
					x2 = width
					sectImg = img[y1:y2,x1:x2]
					resultImg = predictImg(sectImg,x1,x2,y1,y2)
				for i in range(9,-1,-1):
					y1 = height-imgDims[1]
					y2 = height
					x1 = i*100
					x2 = (i+1)*100
					sectImg = img[y1:y2,x1:x2]
					resultImg = predictImg(sectImg,x1,x2,y1,y2)
				for i in range(9,-1,-1):
					y1 = i*100
					y2 = (i+1)*100
					x1 = 0
					x2 = imgDims[2]
					sectImg = img[y1:y2,x1:x2]
					resultImg = predictImg(sectImg,x1,x2,y1,y2)
			cv2.imwrite('/home/pi/DRIVE/NN_Results/' + fileName,resultImg)
			print("Saved " + fileName)
			if "_N.jpg" in fileName:
				resultFile = '/home/pi/Programming/2Class_Results_North.txt'
				updateResults(resultFile,scoreStr_N)
				checkFile(resultFile,"North")
			else:
				resultFile = '/home/pi/Programming/2Class_Results_South.txt'
				updateResults(resultFile,scoreStr_S)
				checkFile(resultFile,"South")
	before = after

