import cv2
import numpy as np
import tkinter as tk
from tkinter import Canvas, Scrollbar
from PIL import Image, ImageTk

def display_images_scrollable(images):
    root = tk.Tk()
    root.title("Scrollable Image Viewer")

    root.geometry("500x525") 
    # Create a frame to hold the canvas and scrollbars
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create vertical scrollbar
    y_scrollbar = Scrollbar(frame, orient=tk.VERTICAL)
    y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create horizontal scrollbar
    x_scrollbar = Scrollbar(frame, orient=tk.HORIZONTAL)
    x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

    # Create a canvas for displaying images
    canvas = Canvas(frame, yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    y_scrollbar.config(command=canvas.yview)
    x_scrollbar.config(command=canvas.xview)

    # Calculate the maximum width and height of the concatenated image
    max_width = sum(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)

    # Create a blank white canvas with the maximum width and height
    composite_image = np.full((max_height, max_width, 3), 255, dtype=np.uint8)
    x_coordinate = 0

    # Concatenate the images horizontally onto the canvas
    for image in images:
        if len(image.shape) == 2 or image.shape[2] == 1:  # Check if the image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to 3 channels
        composite_image[0:image.shape[0], x_coordinate:x_coordinate+image.shape[1]] = image
        x_coordinate += image.shape[1]

    # Convert the composite image to a PhotoImage (Tkinter image object)
    composite_image_tk = ImageTk.PhotoImage(image=Image.fromarray(composite_image))

    # Display the image on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=composite_image_tk)
    canvas.image = composite_image_tk

    # Configure the canvas to allow scrolling
    canvas.config(scrollregion=canvas.bbox("all"))

    root.mainloop()

def rectContours(countours):
    # Filter the area out as we do not want any small rectangles, 
    # Check for significant rectangles. THen we want to filter out 
    # if rectangle has 4 corner points. So we will loop through all 
    # the area in order to filter it out.

    rectCont = []  #  List containing all the corner points of each rectangle
    for i in countours:
        area = cv2.contourArea(i)
        # print("Area: ", area)
        if area > 5000:  #  Neglect the very small values as they represent the small rectangles
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True)  #  Approximate how many corner points the polygon has
            print("Corner Points: 👇")
            print()
            print(approx)
            print("Length of Corner Points: ",len(approx))
            print()
            if len(approx) == 4:  # Select the rectangle
                rectCont.append(i)  #  Append the contour itself in the list each time we getting the rectangle
    
    #  Now arrange the rectangle based on their area
    rectCont= sorted(rectCont,key=cv2.contourArea,reverse=True)

    return rectCont

def getCornerPoints(cont):
    perimeter = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02*perimeter, True)  #  Approximate how many corner points the polygon has
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)  #  1 is the axis
    #print(myPoints)
    #print(add)
    myPointsNew[0]= myPoints[np.argmin(add)]  #  Our first point in the list should be minimum - [0, 0]
    myPointsNew[3]= myPoints[np.argmax(add)]  #  Our last point in the list should be maximum - [w, h]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # [w , 0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  # [0, h]
    #print(diff)
    return myPointsNew

#  To get each of individual options(bubbles) and see how many pixel values are non zero to find out which one of the options in each question marked and which one not marked
#  So to do this, we need to split our image like each of 4 images/columns into 39 different boxes each 13rows X 3columns

def splitOptions(img):
    rows = np.vsplit(img, 1)  #  3 is how many splits in the image
    cv2.imshow('split', rows[0])
    cv2.waitKey()