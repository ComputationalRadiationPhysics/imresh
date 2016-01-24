#!/usr/bin/python

from numpy import genfromtxt
from PIL import Image
from glob import iglob

for fname in iglob('*.dat'):
    print "Converting",fname
    data = genfromtxt( fname )
    data = 255.0 * data / data.max()
    img = Image.fromarray(data)
    img.convert("RGB").save( fname[:-4]+".png" )
