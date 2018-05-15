import numpy as np
import sys

def micrograph2np(width,shift):
  r = int(width/shift-1)
  
  #I = np.load("../DATA_SETS/004773_ProtRelionRefine3D/kino.micrograph.numpy.npy")
  I = np.load("../DATA_SETS/004773_ProtRelionRefine3D/full_micrograph.stack_0001.numpy.npy")
  I = (I-I.mean())/I.std()
  
  N = int(I.shape[0]/shift)
  M = int(I.shape[1]/shift)
  
  S=[]
  for i in range(N-r):
    for j in range(M-r):
      x1 = i*shift
      x2 = x1+width
      y1 = j*shift
      y2 = y1+width
      w = I[x1:x2,y1:y2]
      S.append(w)
  
  S = np.array(S)
  np.save("../DATA_SETS/004773_ProtRelionRefine3D/fraction_micrograph.numpy", S)

