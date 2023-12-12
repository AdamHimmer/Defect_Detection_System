# This program uses a trained neural network to determine if the image saved to the folder being watched is likely
# good or bad. If it is bad, then it quarantines it and waits for the next image. If it is good, then it performs
# additional image manipulation to prepare for the defect detection model in the Ends_NN_2Class.py program.

import os, time
import numpy
import cv2
from PIL import Image, ImageEnhance
from keras.models import model_from_json

def saveSection(img,imgIndex,fileName):
	sectImg = cv2.resize(img,(100,100))

# Load Neural Network Model. This model is used to detect if the end of the product was correctly captured. It uses an
# anomoly detection, so a high score reflects a high probability of an anomaly, which is considered bad.
json_file = open('/home/pi/Programming/NN_Models/End_Anomaly_Detection_Model.json')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('/home/pi/Programming/NN_Models/End_Anomaly_Detection_Model.h5')
loaded_model.compile(loss='mean_squared_error',optimizer='adam')

# Define path to watch for new images. The while loop will pause for 5 seconds between each loop.
path_to_watch = '/home/pi/DRIVE/Ends'
before = dict([(f,None) for f in os.listdir(path_to_watch)])
print("Watching...")
while True:
	after = dict([(f,None) for f in os.listdir(path_to_watch)])
	added = [f for f in after if not f in before]
	if added:
		for fileName in added:
			img = cv2.imread(path_to_watch + "/" + fileName)
			if img is not None:
				# Begin image manipulation
				nnImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				nnImg = cv2.blur(nnImg,(5,5))
				nnImg = cv2.resize(nnImg,(50,50))
				nnImg = nnImg.reshape(1,50*50)
				nnImg = nnImg.astype('float32')
				nnImg = nnImg/255

				# Run image through model and determine score
				score = loaded_model.predict(nnImg,batch_size=None,verbose=0,steps=None)
				diff = nnImg - score
				finalScore = numpy.sum(diff*diff)
				# Determine if image is good (low anomaly score) or bad (high anomaly score)
				if finalScore > 30:
					# If image is likely bad, quarantine it to a separate folder and wait for next image
					print(fileName + " Score: " + str(finalScore))
					os.rename(path_to_watch + "/" + fileName,path_to_watch + "/Bad/" + fileName)
				else:
					# If image is likely good, continue image manipulation
					img = cv2.resize(img,(1000,1000))
					height,width,_ = img.shape

					gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

					gray = cv2.bitwise_not(gray)

					lineImg = gray.copy()

					blurX = gray.copy()
					blurY = gray.copy()

					blurX = blurX[int(0.25*height):int(0.75*height),0:width]
					blurY = blurY[0:height,int(0.25*width):int(0.75*width)]

					blurX = cv2.blur(blurX,(1,1000))
					blurY = cv2.blur(blurY,(1000,1))

					edgesX = cv2.Canny(blurX,5,15)
					edgesY = cv2.Canny(blurY,5,15)

					# Attempt to detect edges of the product in both the vertical and horizontal direction
					linesX = cv2.HoughLinesP(edgesX,rho=1,theta=numpy.pi/180,threshold=100,minLineLength=100,maxLineGap=500)
					linesY = cv2.HoughLinesP(edgesY,rho=1,theta=numpy.pi/180,threshold=100,minLineLength=100,maxLineGap=500)

					# If lines are accurately detected, continue with splitting up the image into sub-images for future
					# processing.
					if (linesX is not None) and (linesY is not None):
						if len(linesX)>=2:
							linesX = numpy.squeeze(linesX[numpy.argsort(linesX[:,:,0],axis=0)])
							linesX = linesX[numpy.logical_and(30<linesX[:,0],linesX[:,0]<(width-30))]
							if len(linesX) >=2:
								linesXFinal = [linesX[0],linesX[len(linesX)-1]]

						if len(linesY)>=2:
							linesY = numpy.squeeze(linesY[numpy.argsort(linesY[:,:,1],axis=0)])
							linesY = linesY[numpy.logical_and(30<linesY[:,1],linesY[:,1]<(height-30))]
							if len(linesY) >=2:
								linesYFinal = [linesY[0],linesY[len(linesY)-1]]

						# Measure distances between lines. This will be used to determine how big the sub-images should
						# be
						topWidth = linesYFinal[0][1]
						bottomWidth = height - linesYFinal[1][1]
						leftWidth = linesXFinal[0][0]
						rightWidth = width - linesXFinal[1][0]

						if (topWidth <150) and (bottomWidth < 150) and (leftWidth < 150) and (rightWidth < 150):
							cv2.rectangle(lineImg,(leftWidth,topWidth),(width-rightWidth,height-bottomWidth),(255,255,255),-1)

							clahe = cv2.createCLAHE(clipLimit = 10.0, tileGridSize = (20,20))
							lineImg = clahe.apply(lineImg)

							# Extract sub-images from the main image. Each of the four sides of the image will have 10
							# sub-images, with some overlap at the corners. These images will not be used directly for
							# defect detection, but will be saved for future refinements to the Neural Network model.
							imgIndex = 1
							for i in range(0,10):
								sectImg = lineImg[0:topWidth,i*100:(i+1)*100]
								saveSection(sectImg,imgIndex,fileName)
								imgIndex = imgIndex+1
							for i in range(0,10):
								sectImg = lineImg[i*100:(i+1)*100,(width-rightWidth):width]
								saveSection(sectImg,imgIndex,fileName)
								imgIndex = imgIndex+1
							for i in range(9,-1,-1):
								sectImg = lineImg[(height-bottomWidth):height,i*100:(i+1)*100]
								saveSection(sectImg,imgIndex,fileName)
								imgIndex=imgIndex+1
							for i in range(9,-1,-1):
								sectImg = lineImg[i*100:(i+1)*100,0:leftWidth]
								saveSection(sectImg,imgIndex,fileName)
								imgIndex=imgIndex+1

							# Save the full image into a network location for future processing with Ends_NN_2Class.py
							cv2.imwrite('/home/pi/DRIVE/Ends_Lines/' + fileName.rsplit('@')[0]+"-(" + str(topWidth) + "," + str(bottomWidth) + "," + str(leftWidth) + "," + str(rightWidth) + ")" + fileName.rsplit(')')[1],lineImg)
							print("Saved " + fileName)
	time.sleep(5)
	before = after


