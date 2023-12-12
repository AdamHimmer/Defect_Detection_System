# This looks at the raw, unprocessed image taken from the camera and change the perspective of the image to make it
# appear as if the image was taken straight-on from the product, and not off at an angle.
# This uses information about a bounding box where the product is expected to be from when the image was taken,
# which is stored in the filename.
# The processed image is then used by Monitor_Ends.py
import os, time
import numpy
import cv2

path_to_watch = '/home/pi/DRIVE/Full_Img'
before = dict([(f,None) for f in os.listdir(path_to_watch) if '_N.jpg' in f])
print("Watching...")
while True:
	# Since the camera may take multiple images in a row for the same product, add some logic to wait 5 seconds after
	# the last image before moving on
	cameraDone = False
	after = dict([(f,None) for f in os.listdir(path_to_watch) if '_N.jpg' in f])
	added = [f for f in after if not f in before]

	if added:
		while not cameraDone:
			time.sleep(5)
			after2 = dict([(f,None) for f in os.listdir(path_to_watch) if '_N.jpg' in f])
			added2 = [f for f in after2 if not f in before]
			if added == added2:
				cameraDone = True
			else:
				after = after2
				added = added2
		sortedFiles = sorted(added,key=lambda f: float(f.rsplit("_")[1].rsplit(".jpg")[0].rsplit("@")[0]))

		# Take the last image. This image would be the closest to the camera and should offer the best resolution
		fileName = sortedFiles[len(sortedFiles)-1]
		img = cv2.imread(path_to_watch + "/" + fileName)
		height,width,_ = img.shape
		# Get the bounding-box information from the image filename
		boxValues = fileName.rsplit("@")[1].rsplit("(")[1].rsplit(")")[0].rsplit(",")
		boxX = int(boxValues[0])
		boxY = int(boxValues[1])
		boxWidth = int(boxValues[2])
		boxHeight = int(boxValues[3])
		boxXEnd = boxX + boxWidth
		boxYEnd = boxY + boxHeight

		mask = numpy.zeros(img.shape[:2],numpy.uint8)
		bgdModel = numpy.zeros((1,65),numpy.float64)
		fgdModel = numpy.zeros((1,65),numpy.float64)

		rect = (boxX,boxY,boxWidth,boxHeight)

		# Use grabcut to remove the background from the image
		print("  Performing grabCut")
		cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
		mask = numpy.where((mask==2)|(mask==0),0,1).astype('uint8')
		maskImg = img*mask[:,:,numpy.newaxis]
		print("   Saved Mask Image")
		cv2.imwrite('/home/pi/DRIVE/Boxes/Mask-' + fileName,maskImg)

		gray = cv2.cvtColor(maskImg,cv2.COLOR_BGR2GRAY)
		_,thresh = cv2.threshold(gray,5,255,cv2.THRESH_BINARY)

		# Use edge and line detection to detect the corners of the product.
		edges = cv2.Canny(thresh,100,200)
		lines = cv2.HoughLinesP(edges,rho=1.5,theta=numpy.pi/180,threshold=100,minLineLength=100,maxLineGap=500)

		lineEqs=[] #captures a,b, and c of ax + by + c =0 equation
		if lines is not None:
			for indLine in lines:
				for x1,y1,x2,y2 in indLine:
					if not ((x1 == x2) and (abs(x1- boxX)<5)) and not ((y1 == y2) and (abs(y1 - (boxY + boxHeight))<5)):
						lineEq = numpy.cross([x1,y1,1],[x2,y2,1])
						lineEqs.append(lineEq)

			cornerPts=[]
			for i in range(len(lineEqs)):
				for j in range(i+1,len(lineEqs)):
					intersect = numpy.cross(lineEqs[i],lineEqs[j])
					if intersect[2] != 0:
						x = int(intersect[0]/intersect[2])
						y = int(intersect[1]/intersect[2])
						if ((boxX-50)<x<(boxXEnd + 50)) and ((boxY - 50)<y<(boxYEnd + 50)):
							cornerPts.append([x,y])

			indPts=[]
			if (len(cornerPts)>0):
				indPts = [cornerPts[0]]
				for i in range(1,len(cornerPts)):
					distCheck = True
					for j in range(len(indPts)):
						if i != j:
							dist = cv2.norm(tuple(cornerPts[i]),tuple(indPts[j]))
							if dist < 100:
								distCheck = False
					if distCheck:
						indPts.append(cornerPts[i])

			print(len(indPts))
			if len(indPts) >= 4:
				tlDistance = 1e6
				blDistance = tlDistance
				brDistance = tlDistance
				trDistance = tlDistance
				tlPt = tuple([1e6,1e6])
				blPt = tlPt
				brPt = tlPt
				trPt = tlPt

				for pt in indPts:
					ptDistanceTL = cv2.norm(tuple(pt),tuple([0,0]))
					ptDistanceBL = cv2.norm(tuple(pt),tuple([0,height]))
					ptDistanceBR = cv2.norm(tuple(pt),tuple([width,height]))
					ptDistanceTR = cv2.norm(tuple(pt),tuple([width,0]))

					if ptDistanceTL < tlDistance:
						tlDistance = ptDistanceTL
						tlPt = tuple(pt)
					if ptDistanceBL < blDistance:
						blDistance = ptDistanceBL
						blPt = tuple(pt)
					if ptDistanceBR < brDistance:
						brDistance = ptDistanceBR
						brPt = tuple(pt)
					if ptDistanceTR < trDistance:
						trDistance = ptDistanceTR
						trPt = tuple(pt)

				pts1 = numpy.float32([brPt,trPt,tlPt,blPt])
				pts2 = numpy.float32([[width,height],[width,0],[0,0],[0,height]])
				# Finally, use the detected corners to warp the perspective of the image for a straight-on view
				M = cv2.getPerspectiveTransform(pts1,pts2)
				transformImg = cv2.warpPerspective(img,M,(width,height))
				cv2.imwrite('/home/pi/DRIVE/Ends/' + fileName, transformImg)
				print("Saved " + fileName)
			else:
				print("   Corners Not Accurately Detected")

			before = after
			for i in range(0,len(sortedFiles)-1):
				os.remove(path_to_watch + "/" + sortedFiles[i])
