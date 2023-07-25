# Mehmet VARAN 


#importing libraries
import cv2
import numpy as np

#getting raw image, copying,displaying
image = cv2.imread("resistor_image.png")
image_copy = image.copy()
cv2.imshow("Default Image", image)

#settings lower and upper bounds for the right colors(brown,green,red,gold)
lower_bounds = np.array([[0,120,100],[60,180,100],[0,180,140],[22,93,0]])
upper_bounds = np.array([[50,150,130],[80,210,130],[50,220,170],[45,255,255]])

#converting image color type to hsv to apply mask
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
masked_brown = cv2.inRange(image_hsv, lower_bounds[0], upper_bounds[0])
masked_green = cv2.inRange(image_hsv, lower_bounds[1], upper_bounds[1])
masked_red = cv2.inRange(image_hsv, lower_bounds[2], upper_bounds[2])
masked_gold = cv2.inRange(image_hsv, lower_bounds[3], upper_bounds[3])
masked_total = masked_brown + masked_green + masked_red + masked_gold
masked_image = cv2.bitwise_and(image_copy,image_copy, mask = masked_total)
cv2.imshow("Masked Image",masked_image)

#converting masked image to gray level
image_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Scaled Image', image_gray)

#applying threshold method for gray image
ret1, image_thresh = cv2.threshold(image_gray, 50, 90, cv2.THRESH_BINARY)
cv2.imshow('Thresholded Image', image_thresh)

#using contours method to determine boundaries
image_contours,_ = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#finding the colors on the image
for cnts in image_contours:
  if cv2.contourArea(cnts) < 505:
    continue
  
  M = cv2.moments(cnts)
  cX = int((M["m10"] / M["m00"]))
  cY = int((M["m01"] / M["m00"]))
  
  b,g,r = cv2.split(image_hsv)

  b_mean = b[cY][cX]
  g_mean = g[cY][cX]
  r_mean = r[cY][cX]
  
  print(cX, cY, b_mean,g_mean,r_mean)

  if b_mean >= 7 and b_mean <= 9 and g_mean >= 200 and g_mean <= 205 and r_mean >=160 and r_mean <= 180: #for brown
      cv2.putText(image_copy, "BROWN(1)", (cX - 90, cY - 60), cv2.FONT_ITALIC, 0.4, (0, 0, 0), 1)
  elif b_mean >= 55 and b_mean <= 65 and g_mean >= 190 and g_mean <= 205 and r_mean >=110 and r_mean <= 130: #for green
      cv2.putText(image_copy, "GREEN(5)", (cX - 20, cY + 60), cv2.FONT_ITALIC, 0.4, (0, 0, 0), 1)
  elif b_mean >= 5 and b_mean <= 15 and g_mean >= 130 and g_mean <= 150 and r_mean >= 110 and r_mean <= 130: # forred
      cv2.putText(image_copy, "RED(2)", (cX + 50, cY - 50), cv2.FONT_ITALIC, 0.4, (0, 0, 0), 1) 
  elif b_mean >= 15 and b_mean <= 25 and g_mean >= 240 and g_mean <= 255 and r_mean >= 240 and r_mean <= 255: #for gold
      cv2.putText(image_copy, "GOLD(%5)", (cX -20, cY - 60 ), cv2.FONT_ITALIC, 0.4, (0, 0, 0), 1)

  cv2.putText(image_copy, "1500 ohm %5", (90,200), cv2.FONT_ITALIC, 0.4, (0, 0, 0), 1) # 
  image_drawcountours = cv2.drawContours(image_copy, cnts, -1, (255, 0, 0), 1)
  cv2.imshow('Image with Counters',image_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()
