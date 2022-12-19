# Mehmet VARAN 
# importing libraries
import tkinter as tk
import cv2
import numpy as np
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from hw2_class import W3h

# generating windows and setting resolution
root = tk.Tk()
root.title("Histogram-Equalization-Specification")
root.geometry("1800x900+0+0")
root.resizable(True, True)


# UI class
class HmGui:
    global read_image
    global hist_values

    def __init__(self, master):
        myFrame = tk.Frame(master)  
        myFrame.place()

        # editing buttons
        self.button1 = tk.Button(master, text="Open Image File", fg="white", bg="black", width=20, height=1, font="body", command=self.openImageButton)
        self.button1.place(x=100, y=50)

        self.button2 = tk.Button(master, text="Open Desired Image", fg="white", bg="black", width=20, height=1, font="Helvatica", command=self.openDesiredButton)
        self.button2.place(x=450, y=50)

        self.button3 = tk.Button(master, text="Histogram", fg="white", bg="black", width=20, height=1,font="Helvatica", command=self.histogramButton)
        self.button3.place(x=800, y=50)

        self.button4 = tk.Button(master, text="Equalization", fg="white", bg="black", width=20, height=1,font="Helvatica", command=self.equalizationButton)
        self.button4.place(x=1150, y=50)

        self.button5 = tk.Button(master, text="Specification", fg="white", bg="black", width=20, height=1,font="Helvatica", command=self.specificationButton)
        self.button5.place(x=1500, y=50)
    
    # getting first image 
    def openImageButton(self): 
        global read_image
        global first_image
        global read_image_forspec
        img_file = fd.askopenfilenames()  # asking for file
        read_image = cv2.imread("{}".format(img_file[0]), 0)  # reading image- flag values is 0 because we needed it grayscaled
        read_image_forspec = cv2.imread("{}".format(img_file[0]), 1) # reading image again - flag value 1 because i'll use it for specificaiton
        img = Image.open(img_file[0])  # opening image
        first_image = img.resize((int(img.size[0] * 0.3), int(img.size[1] * 0.3)),Image.ANTIALIAS)  # resizing
        first_image = ImageTk.PhotoImage(first_image)
        L_img = tk.Label(image=first_image)  # printing img on ui
        L_img.place(x=100, y=120)
        
    # getting second image
    def openDesiredButton(self):
        global second_image
        global read_image2, read_image_forspec
        img_file = fd.askopenfilenames()
        read_image2 = cv2.imread("{}".format(img_file[0]), 1) # reading image again - flag value 1 because i'll use it for specificaiton
        img = Image.open(img_file[0])
        second_image = img.resize((int(img.size[0] * 0.3), int(img.size[1] * 0.3)), Image.ANTIALIAS)
        second_image = ImageTk.PhotoImage(second_image)
        L_img = tk.Label(image=second_image)
        L_img.place(x=100, y=500)
    
    # calculating histogram and printing graph
    def histogramButton(self):
        global third_image
        global read_image, read_image2, read_image_forspec
        global hist_values  # global variable for hist_Values
        hist_values = np.zeros([256])
        hist = W3h(read_image, hist_values, read_image2, read_image_forspec) # calling class
        hist.hist_grapher() # using hist_grapher for calculating histogram
        img = Image.open("graph_hist.png") # getting histogram graph
        third_image = img.resize((int(img.size[0] * 0.8), int(img.size[1] * 0.8)), Image.ANTIALIAS)
        third_image = ImageTk.PhotoImage(third_image)
        L_img = tk.Label(image=third_image)
        L_img.place(x=700, y=120)
        
    # calculating equation and printing img before and after eq. process
    def equalizationButton(self):
        global forth_image, sixth_image
        global hist_values
        global read_image, read_image2, read_image_forspec
        equ = W3h(read_image, hist_values, read_image2, read_image_forspec)
        equ.hist_equal()
        img = Image.open("graph_hist2.png")
        forth_image = img.resize((int(img.size[0] * 0.8), int(img.size[1] * 0.8)), Image.ANTIALIAS)
        forth_image = ImageTk.PhotoImage(forth_image)
        L_img = tk.Label(image=forth_image)
        L_img.place(x=700, y=350)
        
        img2 = Image.open("imgs_aftereq.jpg")
        sixth_image = img2.resize((int(img2.size[0] * 0.2), int(img2.size[1] * 0.2)), Image.ANTIALIAS)
        sixth_image = ImageTk.PhotoImage(sixth_image)
        L_img2 = tk.Label(image=sixth_image)
        L_img2.place(x=700, y=600)
        
    # applying specification and printing the image final form
    def specificationButton(self):
        global fifth_image
        global read_image, read_image2, read_image_forspec
        global hist_values
        spec = W3h(read_image, hist_values, read_image2, read_image_forspec)
        spec.specication()
        img = Image.open("spec_img.jpg")
        fifth_image = img.resize((int(img.size[0] * 0.3), int(img.size[1] * 0.3)), Image.ANTIALIAS)
        fifth_image = ImageTk.PhotoImage(fifth_image)
        L_img = tk.Label(image=fifth_image)
        L_img.place(x=1200, y=120)


Interface = HmGui(root)  # creating interface for our windows
root.mainloop()  