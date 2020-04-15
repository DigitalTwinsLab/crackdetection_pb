# Fractcha is a written-from-scratch program by Ruiqi Chen, Tanay Topac, and Elliot Ransom
# Uses the Python 3 library with no dependence on MATLAB
# Inspired by Cracktcha written by Elliot Ransom

#Use matplotlib with TkAgg to avoid conflict with Mac systems
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import os
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import numpy as np
from PIL import Image, ImageTk
import math

__VERSION__ = "2.2"
MULTIPLY_PARAM = 0.5
PIXELS_PER_BOX = 32
CRACK_COLOR_RGBA = (255, 255, 0, 127)

class App(tk.Frame):

  def __init__(self, name, master, data, mode):
    # Initialize instance variables
    tk.Frame.__init__(self, master)
    self.originalData = data
    self.mode = mode # Reserved for future version that incorporates 2D classification
    if mode == "3D":
        self.original = Image.open(os.path.splitext(name)[0] + ".png")
        self.data = data.copy()
    else:
        self.original = Image.open(name)
        data = np.array(self.original)
        h, w, c = data.shape
        self.data = np.zeros((h, w, 1, 5))
        self.data[:, :, 0, :c] = data
        if c == 3:
            self.data[:, :, 0, 3] = 255
    self.image = None
    self.imageID = None
    self.master = master
    self.name = name
    self.scale = MULTIPLY_PARAM
    self.pixelsPerBox = PIXELS_PER_BOX
    self.cracked = {(r, c):0 for r in range(math.ceil(self.original.height/self.pixelsPerBox)) \
        for c in range(math.ceil(self.original.width/self.pixelsPerBox))}
    self.rectangles = set()
    self.crackSquare = Image.fromarray(np.array(list(CRACK_COLOR_RGBA), dtype=np.uint8).reshape((1, 1, -1)))
    self.crackedSquareImg = None

    # Initialize canvas
    self.canvas = tk.Canvas(self, bg="black",
        width=self.original.width*self.scale, 
        height=self.original.height*self.scale)
    self.columnconfigure(0,weight=1)
    self.rowconfigure(0,weight=1)
    self.canvas.grid(row=0, sticky=tk.W+tk.E+tk.N+tk.S)
    self.pack(fill="none", expand=False)
    self.draw()

    # Add binder events
    self.canvas.bind("<Button-1>", self.click)
    self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

  def draw(self):
    # add image
    size = int(self.scale*self.original.width), int(self.scale*self.original.height)
    if self.image:
        self.canvas.delete(self.imageID)
    self.image = ImageTk.PhotoImage(self.original.resize(size))
    self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)

    # add grid
    W_y = np.round(np.array([i*self.pixelsPerBox*self.scale for i in \
        range(int((self.original.height)//self.pixelsPerBox) + 1)]))
    W_x = np.full(len(W_y), 0)
    E_y = W_y
    E_x = np.round(np.full(len(E_y), self.image.width()))
    N_x = np.round(np.array([i*self.pixelsPerBox*self.scale for i in \
        range(int((self.original.width)//self.pixelsPerBox) + 1)]))
    N_y = np.full(len(N_x), 0)
    S_x = N_x
    S_y = np.round(np.full(len(N_y), self.image.height()))
    for i in range(len(W_y)):
      self.canvas.create_line(W_x[i], W_y[i], E_x[i], E_y[i], fill='white', width="2", tags = "Line")
    for i in range(len(N_x)):
      self.canvas.create_line(N_x[i], N_y[i], S_x[i], S_y[i], fill='white', width="2", tags = "Line")
    
  def click(self, event):
    # Change crack status upon mouse clicks
    boxID_c = event.x // (self.pixelsPerBox * self.scale)
    boxID_r = event.y // (self.pixelsPerBox * self.scale)
    if self.cracked[(boxID_r,boxID_c)] == 0:
        self.cracked[(boxID_r,boxID_c)] = 1 # select as crack
    else:
        self.cracked[(boxID_r,boxID_c)] = 0 # unselect
    self.drawRectangles()

  def drawRectangles(self):
    # clear all existing rectangles
    for rec in self.rectangles:
        self.canvas.delete(rec)
    self.rectangles.clear()

    # create ImageTk of cracked square
    size = (int(self.scale*self.pixelsPerBox), int(self.scale*self.pixelsPerBox))
    self.crackedSquareImg = ImageTk.PhotoImage(self.crackSquare.resize(size))
    # redraw based on cracked dict
    for (r, c), val in self.cracked.items():
        x = int(c*self.scale*self.pixelsPerBox)
        y = int(r*self.scale*self.pixelsPerBox)
        if val == 1:
            img = self.canvas.create_image(x, y, image=self.crackedSquareImg, anchor=tk.NW)
            self.rectangles.add(img)

    ## Save/discard when closing window.          
  def on_closing(self):
    if tk.messagebox.askokcancel("Quit", "Save annotations? (Cancel to quit without saving)"):
      self.saveResults()
    self.master.destroy()

  def saveResults(self):
    for (r, c), val in self.cracked.items():
      xStart = r*self.pixelsPerBox
      yStart = c*self.pixelsPerBox
      xEnd = xStart + self.pixelsPerBox
      yEnd = yStart + self.pixelsPerBox
      if val == 1:
        self.data[xStart:xEnd, yStart:yEnd, :, -1] = 1
      else:
        self.data[xStart:xEnd, yStart:yEnd, :, -1] = 0
    np.save(os.path.splitext(self.name)[0], self.data)

def ptsToNumpy(ptsFile, spacing=None, coordinateColumns=(0, 1, 2), colorColumns=(3, 4, 5, 6)):
  # each line in the pts file should be a comma separated list; nominally x y z i r g b
  # there's some room for customization if you specify which are the coordinate (xyz) columns and which are the color (irgb) columns
  # if spacing=None, the first three lines in the pts file MUST be three numbers representing grid spacing in x y z directions
  # otherwise, spacing should be provided as a tuple; e.g. spacing=(0.001, 0.001, 0.002)

  # determine spacing, if not given
  if spacing == None:
    f = open(ptsFile, "r")
    lineNum = 0
    for line in f:
      if lineNum == 0:
        dx = float(line)
      elif lineNum == 1:
        dy = float(line)
      elif lineNum == 2:
        dz = float(line)
      else:
        break
      lineNum += 1
    f.close()
    spacing = (dx, dy, dz)
    data = np.loadtxt(ptsFile, delimiter=",", skiprows=3, usecols=coordinateColumns + colorColumns)
  else:
    data = np.loadtxt(ptsFile, delimiter=",", usecols=coordinateColumns + colorColumns)

  # find extents
  minX = min(data[:, 0])
  maxX = max(data[:, 0])
  minY = min(data[:, 1])
  maxY = max(data[:, 1])
  minZ = min(data[:, 2])
  maxZ = max(data[:, 2])
  numX = int(round((maxX - minX)/spacing[0])) + 1
  numY = int(round((maxY - minY)/spacing[1])) + 1
  numZ = int(round((maxZ - minZ)/spacing[2])) + 1

  # initialize resulting numpy array
  result = np.full((numX, numY, numZ, len(coordinateColumns) + len(colorColumns) + 2), float("nan"))

  # loop through data and put into result
  for pt in range(data.shape[0]):
    i = int(round((data[pt, 0] - minX)/spacing[0]))
    j = int(round((data[pt, 1] - minY)/spacing[1]))
    k = int(round((data[pt, 2] - minZ)/spacing[2]))
    result[i, j, k, 0:7] = data[pt]
    result[i, j, k, 7] = 255
  # result = np.rot90(result, k=1, axes=(1, 0)) #Rotate the np array to have the bricks horizontally (Tanay)

  return result

def flattenMatrix(mat, cStart=4, cEnd=8):
  return np.nanmean(mat[:, :, :, cStart:cEnd], axis=2) # fixed indices (Ruiqi 5/22)

def createImage(name, img):
  img = img.copy()
  img /= 255
  plt.imsave(name, img)

def readFile():
  root = tk.Tk()
  root.filename = tk.filedialog.askopenfilename(initialdir = os.getcwd(), title = 'Select 2D image or 3D CSV file.')
  return root, root.filename

def visualizeClassifiedArray(numpyArrayFile, save=True):
    # for debugging purposes
    OPAQUENESS = 0.60
    try:
      arr = np.load(numpyArrayFile)
    except:
      return
    arr[:, :, :, -2] = arr[:, :, :, -2]*(OPAQUENESS + (1 - OPAQUENESS)*arr[:, :, :, -1])
    if arr.shape[3] == 5: # 2D image
        arrFlat = flattenMatrix(arr, cStart=0, cEnd=4)
    else:
        arrFlat = flattenMatrix(arr)
    if save:
        plt.imsave(os.path.splitext(numpyArrayFile)[0] + "_annotated", arrFlat/255)
    plt.imshow(arrFlat/255)
    plt.show()

def main():
    root, name = readFile()
    filename, filext = os.path.splitext(name)
    if filext.lower() == ".csv":
        mode = "3D"
        data = ptsToNumpy(name)
        flattenedData = flattenMatrix(data)
        createImage(filename, flattenedData)
    elif len(filext) > 0:
        mode = "2D"
        data = None # Reserved for future version
        # raise Exception("Fractcha only supports 3D CSV point clouds!")
    else:
        return # User cancelled; close program
    app = App(name, root, data, mode)
    app.mainloop()
    visualizeClassifiedArray(filename + ".npy")

if __name__ == "__main__":
    main()