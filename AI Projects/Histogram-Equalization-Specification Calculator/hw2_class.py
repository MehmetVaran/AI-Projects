# Mehmet VARAN 
# importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.exposure import match_histograms

# creating class and functions
class W3h:
    def __init__(self,img,hist,img2,img3): # getting variables
        self.img=img
        self.hist=hist
        self.img2=img2
        self.img3=img3
        self.img_list=[img, img2, img3]
    
    def hist_grapher(self):
        hist = cv2.calcHist([self.img], [0], None, [256], [0,256]) # calculating histogram
        # creating and editing plot for histogram 
        plt.plot(hist)
        plt.title("Histogram")
        plt.xlabel("Gray Level")
        plt.ylabel("No. of Pixel (Counting)")
        plt.savefig("graph_hist")
        
    
    def hist_equal(self):
        eq = cv2.equalizeHist(self.img) # calculating equalization 
        imgs_aftereq = np.hstack((self.img,eq)) # combining pure img and equalized img
        cv2.imwrite("imgs_aftereq.jpg", imgs_aftereq) # saving img
        hist2 = cv2.calcHist([eq], [0], None, [256], [0,256])
        plt.plot(hist2)
        plt.title("Equalized Histogram")
        plt.xlabel("Gray Level")
        plt.ylabel("No. of Pixel (Counting)")
        plt.savefig("graph_hist2.png")
    
    def specication(self):
        # getting pure images
        image = self.img3.copy()
        reference = self.img2.copy()
        matched = match_histograms(image, reference , multichannel=True) # matching images
        
        #creating and editing plot for matching 
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),sharex=True, sharey=True)
        for aa in (ax1, ax2, ax3): aa.set_axis_off()
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
          
        for i, img in enumerate((image, reference, matched)):
            for c, c_color in enumerate(('red', 'green', 'blue')):
                img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
                axes[c, i].plot(bins, img_hist / img_hist.max())
                img_cdf, bins = exposure.cumulative_distribution(img[..., c])
                axes[c, i].plot(bins, img_cdf)
                axes[c, 0].set_ylabel(c_color)
          
        axes[0, 0].set_title('Source')
        axes[0, 1].set_title('Reference')
        axes[0, 2].set_title('Matched')
          
        cv2.imwrite("spec_img.jpg", matched)
        
