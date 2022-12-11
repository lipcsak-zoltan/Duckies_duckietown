""" Script for preprocessing raw data"""

from PIL import Image
import os,glob
import numpy as np
import cv2


SOURCE_PATH="DATA/" #You should modify this based on where your source files are
TARGET_PATH="Train/Data" #and where yu want to save the processed images 

white_lower = np.array([np.round(  0 / 2), np.round(0.5 * 255), np.round(0.00 * 255)])
white_upper = np.array([np.round(360 / 2), np.round(1.00 * 255), np.round(0.30 * 255)])

yellow_lower = np.array([np.round( 40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
yellow_upper = np.array([np.round( 60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])

i = 0
log = []

for filename in glob.glob(os.path.join(SOURCE_PATH, '*.jpeg')):
    with open(filename, 'r') as f:
        targetname = "train_im"+str(i)+".png"

        input= cv2.imread(filename)  
        hls = cv2.cvtColor(input, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(hls, white_lower, white_upper)
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
        maski = cv2.bitwise_or(yellow_mask, white_mask)
        masked = cv2.bitwise_and(input, input, mask=maski)
        ksize  = (7, 7)
        masked = cv2.blur(masked, ksize)
        output= masked[200:800, 20:500]
        output = cv2.resize(output,[100, 75])

        s = filename
        s=s.replace('.jpeg','.txt')
        s=s.replace('/images/','/properties/')
        

        y=np.loadtxt(str(s))
        transition= np.array([0.0,0.0,0.0,0.0], dtype=float)
        index = int(y[4])
        transition[index]=1
        angle = np.array([0.0, 0.0, 0.0])
        if y[1] < 0:
            angle[0] = 1
        if y[1] == 0:
            angle[1] = 1
        if y[1] >0:
            angle[2]= 1
        saveangle_name="train_ang"+str(i)+".txt"
        savetransition_name="train_vel"+str(i)+'.txt'
        log.append([filename, s])
        np.savetxt(os.path.join(TARGET_PATH,saveangle_name),angle)
        np.savetxt(os.path.join(TARGET_PATH, savetransition_name),transition)
        cv2.imwrite(os.path.join(TARGET_PATH,targetname),output)
    i+=1
logap=np.array(log,dtype=str)
np.savetxt('log.txt',logap, fmt='%s') #Simple logging
