import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm


# read and display image
B=8 # blocksize (In Jpeg the
img1 = cv2.imread("picture.jpg", cv2.IMREAD_UNCHANGED)
h,w=np.array(img1.shape[:2])/B * B
img1=img1[:int(h),:int(w)]
#Convert BGR to RGB
img2=np.zeros(img1.shape,np.uint8)
img2[:,:,0]=img1[:,:,2]
img2[:,:,1]=img1[:,:,1]
img2[:,:,2]=img1[:,:,0]
plt.imshow(img2)
# plt.show()

#
#first component is col, second component is row
print(h,w)
scol=64
srow=64
plt.plot([B*scol,B*scol+B,B*scol+B,B*scol,B*scol],[B*srow,B*srow,B*srow+B,B*srow+B,B*srow])
plt.axis([0,w,h,0])

# transform BGR to Ycrcb and subsample chrominnace channels
transcol=cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
SSV=2
SSH=2
crf=cv2.boxFilter(transcol[:,:,1],ddepth=-1,ksize=(2,2))
cbf=cv2.boxFilter(transcol[:,:,2],ddepth=-1,ksize=(2,2))
crsub=crf[::SSV,::SSH]
cbsub=cbf[::SSV,::SSH]
imSub=[transcol[:,:,0],crsub,cbsub]

# discrete cosine transform and quantisation
QY=np.array([[16,11,10,16,24,40,51,61],
                         [12,12,14,19,26,48,60,55],
                         [14,13,16,24,40,57,69,56],
                         [14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],
                         [24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],
                         [72,92,95,98,112,100,103,99]])

QC=np.array([[17,18,24,47,99,99,99,99],
                         [18,21,26,66,99,99,99,99],
                         [24,26,56,99,99,99,99,99],
                         [47,66,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99]])

# quality factor
QF=30.0
if QF < 50 and QF > 1:
        scale = np.floor(5000/QF)
elif QF < 100:
        scale = 200-2*QF
else:
        print ("Quality Factor must be in the range [1..99]")
scale=scale/100.0
Q=[QY*scale,QC*scale,QC*scale]

TransAll=[]
TransAllQuant=[]
ch=['Y','Cr','Cb']
plt.figure()
for idx,channel in enumerate(imSub):
        plt.subplot(1,3,idx+1)
        channelrows=channel.shape[0]
        channelcols=channel.shape[1]
        Trans = np.zeros((channelrows,channelcols), np.float32)
        TransQuant = np.zeros((channelrows,channelcols), np.float32)
        blocksV=channelrows/B
        blocksH=channelcols/B
        vis0 = np.zeros((channelrows,channelcols), np.float32)
        vis0[:channelrows, :channelcols] = channel
        vis0=vis0-128
        for row in range(int(blocksV)):
                for col in range(int(blocksH)):
                        currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                        Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
                        TransQuant[row*B:(row+1)*B,col*B:(col+1)*B]=np.round(currentblock/Q[idx])
        TransAll.append(Trans)
        TransAllQuant.append(TransQuant)
        if idx==0:
                selectedTrans=Trans[srow*B:(srow+1)*B,scol*B:(scol+1)*B]
        else:
                sr=np.floor(srow/SSV)
                sc=np.floor(scol/SSV)
                selectedTrans=Trans[int(sr*B):int((sr+1)*B),int(sc*B):int((sc+1)*B)]
        plt.imshow(selectedTrans,cmap=cm.jet,interpolation='nearest')
        plt.colorbar(shrink=0.5)
        plt.title('DCT of '+ch[idx])


# plt.show()


# decoding
DecAll=np.zeros((int(h), int(w) ,3), np.uint8)
for idx,channel in enumerate(TransAllQuant):
        channelrows=channel.shape[0]
        channelcols=channel.shape[1]
        blocksV=channelrows/B
        blocksH=channelcols/B
        back0 = np.zeros((channelrows,channelcols), np.uint8)
        for row in range(int(blocksV)):
                for col in range(int(blocksH)):
                        dequantblock=channel[row*B:(row+1)*B,col*B:(col+1)*B]*Q[idx]
                        currentblock = np.round(cv2.idct(dequantblock))+128
                        currentblock[currentblock>255]=255
                        currentblock[currentblock<0]=0
                        back0[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
        back1=cv2.resize(back0,(int(w),int(h)))
        DecAll[:,:,idx]=np.round(back1)

reImg=cv2.cvtColor(DecAll, cv2.COLOR_YCR_CB2BGR)
cv2.imwrite('picture_compressed.jpg', reImg)
plt.figure()
img3=np.zeros(img1.shape,np.uint8)
img3[:,:,0]=reImg[:,:,2]
img3[:,:,1]=reImg[:,:,1]
img3[:,:,2]=reImg[:,:,0]
plt.imshow(img3)
SSE=np.sqrt(np.sum((img2-img3)**2))
print ("Sum of squared error: ",SSE)
plt.show()


"""
# Jpeg encoding

import cv2
import numpy as np
import math

# import zigzag functions
from zigzag import *

# defining block size
block_size = 8

# reading image in grayscale
img = cv2.imread('NASA.jpg', 0)
cv2.imshow('input image', img)

# get size of the image
[h , w] = img.shape


##################### step 1 #####################
# compute number of blocks by diving height and width of image by block size

# you need to convert h and w to float to get the right number
h = np.float32(h)##### your code #####
w = np.float32(w)##### your code #####

# to cover the whole image the number of blocks should be ceiling of the division of image size by block size
# at the end convert it to int

# number of blocks in height
nbh = math.ceil(h/block_size)##### your code #####
nbh = np.int32(nbh)

# number of blocks in width
nbw = math.ceil(w/block_size)##### your code #####
nbw = np.int32(nbw)

##################### step 2 #####################
# Pad the image, because sometime image size is not dividable to block size
# get the size of padded image by multiplying block size by number of blocks in height/width

# height of padded image
H =  block_size * nbh##### your code #####

# width of padded image
W =  block_size * nbw##### your code #####

# create a numpy zero matrix with size of H,W
padded_img = np.zeros((H,W))

# copy the values of img  into padded_img[0:h,0:w]
for i in range(int(h)):
        for j in range(int(w)):
                pixel = img[i,j]
                padded_img[i,j] = pixel ##### your code #####

# or this other way here
#padded_img[0:h,0:w] = img[0:h,0:w]

cv2.imshow('input padded image', np.uint8(padded_img))

##################### step 3 #####################
# start encoding:
# divide image into block size by block size (here: 8-by-8) blocks
# To each block apply 2D discrete cosine transform
# reorder DCT coefficients in zig-zag order
# reshaped it back to block size by block size (here: 8-by-8)


# iterate over blocks
for i in range(nbh):

        # Compute start row index of the block
        row_ind_1 = i*block_size##### your code #####

        # Compute end row index of the block
        row_ind_2 = row_ind_1+block_size##### your code #####

        for j in range(nbw):

            # Compute start column index of the block
            col_ind_1 = j*block_size##### your code #####

            # Compute end column index of the block
            col_ind_2 = col_ind_1+block_size##### your code #####

            # select the current block we want to process using calculated indices
            block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]

            # apply 2D discrete cosine transform to the selected block
            # you should use opencv dct function
            DCT = cv2.dct(block)##### your code #####

            # reorder DCT coefficients in zig zag order by calling zigzag function
            # it will give you a one dimentional array
            reordered = zigzag(DCT)##### your code #####

            # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
            reshaped= np.reshape(reordered, (block_size, block_size))##### your code #####

            # copy reshaped matrix into padded_img on current block corresponding indices
            padded_img[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshaped##### your code #####

cv2.imshow('encoded image', np.uint8(padded_img))

##################### step 4 #####################
# write h, w, block_size and padded_img into txt files at the end of encoding

# write padded_img into 'encoded.txt' file. You can use np.savetxt function.
np.savetxt('encoded.txt',padded_img)##### your code #####

# write [h, w, block_size] into size.txt. You can use np.savetxt function.
np.savetxt('size.txt',[h, w, block_size])##### your code #####

##################################################

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
