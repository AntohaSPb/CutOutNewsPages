import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import random 
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

used_model = 'Resnet'  #or 'Unet'
norm_proc = 'numpy_stdev' # or 'torch' or 'numpy_minmax'

if used_model == 'Unet':
    #inference code for U-Net
    best_model = torch.jit.load('c:/Pyproj/Segm/Code/models/best_model_new2.pt', map_location='cpu')

if used_model == 'Resnet':
    #inference code for ResNet
    best_model = deeplabv3_mobilenet_v3_large(num_classes=2)
    best_model.to('cpu')
    checkpoints = torch.load('c:/Pyproj/Segm/Code/models/model_mbv3_iou_mix_2C049.pth', map_location='cpu')
    best_model.load_state_dict(checkpoints, strict=False)

#set whatever model to evaluation mode, no back prop of err
best_model.eval()

im_path = 'c:/Pyproj/Segm/Code/imagery/Test/IMG-081.JPG'  # Replace with your image path
ou_path = 'c:/Pyproj/Segm/Code/imagery/Crop/DSC0395.JPG'  # Replace with your output path
ppage = cv2.imread(im_path)

if ppage is None:
    print("Error: Image not loaded. Please check the file path.")
    
#turn BGR into RGB
ppageRGB = cv2.cvtColor(ppage, cv2.COLOR_BGR2RGB)

#detect original pixels of the page
page_size = ppageRGB.shape
print ("original size" , page_size)

#resize and show new dims
small_size = (384, 256)  # Note: OpenCV uses (width, height) format
ppage_small = cv2.resize(ppageRGB, small_size)
page_size = ppage_small.shape
print ("reduced size", page_size)

#work on the page to have it blank
#kernel = np.ones((3,3),np.uint8)
# ppage_small_old =  np.copy(ppage_small) 
# cv2.morphologyEx(ppage_small, cv2.MORPH_CLOSE, kernel, iterations= 3)

#normalize and produce a tensor #RGB 0-255 range into 0-1 tensor, HxWxC to CxHxW and add batch dimension
if norm_proc == 'numpy_minmax':
    #normalize the image using NumPy with min-max filter
    min_val = ppage_small.min(axis=(0, 1))
    max_val = ppage_small.max(axis=(0, 1))
    ppage_norm = (ppage_small - min_val) / (max_val - min_val)
    ppage_tensor = torch.from_numpy(ppage_norm).permute(2, 0, 1).unsqueeze(0).float()
    mystd, mymean = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

if norm_proc == 'numpy_stdev':
    #normalize the image using NumPy with mean and stdev
    mymean = np.mean(ppage_small, axis=(0, 1))   #  (0.4611, 0.4359, 0.3905)
    mystd = np.std(ppage_small, axis=(0, 1))     #  (0.2193, 0.2150, 0.2109)
    
    #norm option 1: basic standardization around mean as zero, negatives clamped when shown
    #ppage_norm = ((ppage_small - mymean) / mystd)
    #norm option 2: set the picture within the double sigma across the mean, tails clipped
    ppage_norm = np.clip(((ppage_small - (mymean - mystd))/(mystd * 2)), 0, 1)

    ppage_tensor = torch.from_numpy(ppage_norm).permute(2, 0, 1).unsqueeze(0).float()
    print (mymean, mystd)

if norm_proc == 'torch':
    #normalize using torchvision
    mymean =  (0.4611, 0.4359, 0.3905)      #from source dataset
    mystd =  (0.2193, 0.2150, 0.2109)       #from source dataset
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mymean, mystd),])
    ppage_tnorm = normalize(ppage_small)    #execute function resulting in a tensor
    ppage_tensor = torch.unsqueeze(ppage_tnorm, dim=0).float()  #expand tensor with batch dimension
    ppage_norm = ppage_tnorm.permute(1, 2, 0).squeeze().numpy() #tensor to numpy for showing

page_size = ppage_tensor.shape
print ("supplied tensor size", page_size)

#supply my tensor to the model
with torch.no_grad():  # Disable gradient calculation for inference
    output = best_model(ppage_tensor)
    if used_model == 'Resnet':
        pred_mask = output['out']        #ResNet produces a Dictionary with one more dimension for 'words'
    if used_model == 'Unet':
        pred_mask = output              #Unet gives out a tensor

#check out for prediction result    
    page_size = pred_mask.shape
    print ("returned mask size", page_size)

# kill batch dim of size 1, detach tensor from backwards propagation, turn into a NumPy array to show
np_mask = pred_mask.detach().squeeze().numpy()
page_size = np_mask.shape
print ("squeezed mask size", page_size)

#get highest probability class index, given that class axis is 0 (first)
likely_mask = np.argmax(np_mask, axis=0)
page_size = likely_mask.shape
print ("likely mask size", page_size)

#convert mask for manipulations
likely_mask = likely_mask.astype(np.uint8)

#manipulate with erosion and dilation
#Define a kernel
kernel = np.ones((5, 5), np.uint8)
#Erode the mask to remove small islands, Dilate the mask to restore the main object size
eroded_mask = cv2.erode(likely_mask, kernel, iterations=1)
smoothed_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
# Apply opening to remove small islands,Apply closing to fill small holes
opened_mask = cv2.morphologyEx(likely_mask, cv2.MORPH_OPEN, kernel)
closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
# return adjusted image SMOOTHED
fixed_mask = smoothed_mask

#smoothed contours
contours, _ = cv2.findContours(fixed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
zero_mask = np.zeros_like(fixed_mask)
ppage_combi = np.copy(ppage_small)

if contours:
    max_contour = max(contours, key=cv2.contourArea)   #get biggest contour
    epsilon = 0.05 * cv2.arcLength(max_contour, True)  # accuracy for waste disposal
    mypoly = cv2.approxPolyDP(max_contour, epsilon, True) #approximate with a polygon
    cv2.drawContours(zero_mask, [mypoly], -1, (1), thickness=5) #draw on a new array
    cv2.drawContours(ppage_combi, [mypoly], -1, (255, 0, 0), thickness=2) #draw on the image

#for quadrilineal find the corners for transformation
if len(mypoly) == 4:
    # Sort the points as top-left, top-right, bottom-right, bottom-left and put into an src array
    mycorners = mypoly.reshape(4, 2)
    sorted_points = sorted(mycorners, key=lambda x: (x[1], x[0]))
    top_points = sorted(sorted_points[:2], key=lambda x: x[0])  # Top-left and top-right
    bottom_points = sorted(sorted_points[2:], key=lambda x: x[0])  # Bottom-left and bottom-right
    src_points = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype='float32')    

#define dest size and transform the image
width, height = 384, 256
#perspective transformation matrix: topleft, topright, botright, botleft
dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]]) 
#estimate the matrix and apply it on image
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
ppage_unwarp = cv2.warpPerspective(ppage_combi, matrix, (width, height))

#plot the images
images = [ppage_small, ppage_norm, likely_mask, fixed_mask, ppage_combi, ppage_unwarp]
titles = ['Resize', 'Normalize', 'Prediction', 'Smooth', 'Rectangle','Unwarp']
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap='gray')  # Display image
    ax.set_title(titles[i])  # Set title for each subplot
    ax.axis('off')
plt.tight_layout()
plt.show()

del ppage, ppage_norm, likely_mask, np_mask, ppage_small, ppageRGB, ppage_tensor, ppage_combi