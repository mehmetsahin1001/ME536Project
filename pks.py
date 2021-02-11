import skimage
import plotly.express as px
import numpy as np
import skimage

im = skimage.io.imread("t9.jpg")
cropped = im[150:850,200:1600]
fig = px.imshow(cropped)
fig.show()