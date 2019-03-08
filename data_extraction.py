import numpy as np
import cv2
from matplotlib import pyplot as plt
import os,os.path
import math
import csv
path = "D:/project/train_set"
#path = "D:/project/scotti/ALL_IDB1/im"
for f in os.listdir(path):
    ext = os.path.basename(f)
    w = ext.split("_")
    d = w[0]
    o = w[1]
    g = w[2]
    n = g.split(".")
    t = n[0]

#Import
    img = cv2.imread(os.path.join(path, f))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #plt.imshow(img)
    #plt.xlabel("Imported")
    #plt.show()
    
#denoise image
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    #plt.subplot(121),plt.imshow(img),plt.xlabel("Before Denoising")
    #plt.subplot(122),plt.imshow(dst),plt.xlabel("After Denoising")
    #plt.show()

#resize
    res = cv2.resize(dst, (1600,1600), interpolation = cv2.INTER_CUBIC)
    #plt.imshow(res)
    #plt.xlabel("Resized")
    #plt.show()
    
#Segmentation
    SegImage=cv2.GaussianBlur(res,(7,7),0)
    Z = SegImage.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,10,None,criteria,20,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((SegImage.shape))
    lb = label.reshape((SegImage.shape[0],SegImage.shape[1]))
    out = res2.astype(np.uint8)
    #plt.imshow(out)
    #plt.xlabel("Segmented")
    #plt.show()
    #cv2.waitKey(0)


#grayscale
    gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray)
    #plt.xlabel("Grayscale")
    #plt.show()

#Threshold
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #plt.imshow(thresh)
    #plt.xlabel("Threshold")
    #plt.show()
    
# noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    #plt.imshow(opening)
    #plt.xlabel("Noise Removed")
    #plt.show()

# sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=5)
    #plt.imshow(sure_bg)
    #plt.xlabel("Sure Background")
    #plt.show()

# Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.15*dist_transform.max(),200,0)
    #plt.imshow(sure_fg)
    #plt.xlabel("Sure Foreground")
    #plt.show()

# Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    #plt.imshow(unknown)
    #plt.xlabel("Unknown")
    #plt.show()

# Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

# Now, mark the region of unknown with zero
    markers[unknown==255] = 0

#Segment
    markers = cv2.watershed(out,markers)
    out[markers == -1] = [255,0,0]
    #plt.imshow(out)
    #plt.xlabel("Basin Formation")
    #plt.show()

#edge detection
    edges = cv2.Canny(out,100,200)
    canny_edge_nparr = np.asarray(edges)
    #plt.imshow(edges)
    #plt.xlabel("Edge Detection")
    #plt.show()
    
#Area
    n = np.sum(thresh == 255)

#Perimeter
    p = np.sum(edges == 255)

#compactness
    c = math.pow(p, 2) / n
		
#writing parameters to csv file
    with open('dataKMWS.csv', 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([n, p, c,t])
