# -*- coding: utf-8 -*-
"""
@author: mehmet
"""
# Mehmet VARAN

#importing libraries
import tkinter as tk
import cv2
import numpy as np
from tkinter import filedialog as fd
from PIL import ImageTk, Image

# generating windows and setting resolution
root = tk.Tk()
root.title("Detecting Forest-Sea-Town(SnowyLand)")
root.geometry("800x600+0+0")
root.resizable(True, True)


class HmGui:

    def __init__(self, master):
        myFrame = tk.Frame(master)  
        myFrame.place()

        # editing buttons
        self.button1 = tk.Button(master, text="Select Image", fg="white", bg="black", width=20, height=1, font="body", command=self.selectImageButton)
        self.button1.place(x=100, y=50)

        self.button2 = tk.Button(master, text="Detect Forest", fg="white", bg="black", width=20, height=1, font="Helvatica", command=self.detectForestButton)
        self.button2.place(x=100, y=200)

        self.button3 = tk.Button(master, text="Detect Sea", fg="white", bg="black", width=20, height=1,font="Helvatica", command=self.detectSeaButton)
        self.button3.place(x=100, y=350)

        self.button4 = tk.Button(master, text="Detect Town or Snowy Land", fg="white", bg="black", width=20, height=1,font="Helvatica", command=self.detectTownButton)
        self.button4.place(x=100, y=500)

    # selecting image
    def selectImageButton(self):
        #defining some global variables
        global read_image
        global first_image
        global image_hsv
        global image_copy
        img_file = fd.askopenfilenames()  # asking for file
        read_image = cv2.imread("{}".format(img_file[0]))  # reading image
        
        #converting image bgr2hsv to apply mask on other functions
        image_hsv = cv2.cvtColor(read_image, cv2.COLOR_BGR2HSV)
        image_copy = read_image.copy()
        
        img = Image.open(img_file[0])  # opening image
        first_image = img.resize((int(img.size[0] * 0.3), int(img.size[1] * 0.3)),Image.ANTIALIAS)  # resizing
        first_image = ImageTk.PhotoImage(first_image)
        L_img = tk.Label(image=first_image)  # printing img on ui
        L_img.place(x=350, y=150)
        
        
    def detectForestButton(self):
        #determining lower and upper values
        lower_bounds = np.array([0,0,0])
        upper_bounds = np.array([102,255,150])
        
        #applying mask and finding contours
        masked_green = cv2.inRange(image_hsv, lower_bounds, upper_bounds)
        contours_g, _ = cv2.findContours(masked_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        #drawing contours to determine object
        image_green = cv2.bitwise_and(image_copy, image_copy, mask=masked_green)
        cv2.drawContours(image_green, contours_g, -1, (0,150,0), 1)
        cv2.imshow("Masked Green",image_green)
        
        
    def detectSeaButton(self):
        #determining lower and upper values
        lower_bounds = np.array([100,140,0])
        upper_bounds = np.array([255,200,255])
        
        #applying mask and finding contours
        masked_blue = cv2.inRange(image_hsv, lower_bounds, upper_bounds)
        contours_b, _ = cv2.findContours(masked_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        #drawing contours to determine object
        image_blue = cv2.bitwise_and(image_copy,image_copy, mask=masked_blue)
        cv2.drawContours(image_blue, contours_b, -1, (100,10,0), 1)
        cv2.imshow("Masked Blue",image_blue)
        
        
    def detectTownButton(self):
        #determining lower and upper values
        lower_bounds = np.array([0,0,140])
        upper_bounds = np.array([255,25,255])
        
        #applying mask and finding contours
        masked_orange = cv2.inRange(image_hsv, lower_bounds, upper_bounds)
        contours_o, _ = cv2.findContours(masked_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        #drawing contours to determine object
        image_orange = cv2.bitwise_and(image_copy,image_copy, mask=masked_orange)
        cv2.drawContours(image_orange, contours_o, -1, (0,165,255), 1)
        cv2.imshow("Masked Orange",image_orange)


Interface = HmGui(root)  # creating interface for our windows
root.mainloop()
