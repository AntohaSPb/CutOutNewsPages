# Printed page scanner - AI page segmentation part

The code presented here aims at flexible cv2 & PyTorch-based parametrized extraction of document or book page from the image that contains underlying surface, such as table top, and stuff like pens, phones, hands etc. The idea is to train a neural network model on a part of the actual dataset and then use the model to work on the rest of the dataset.

The following tutorials were useful to learn how to apply AI for image processing:
1. https://github.com/spmallick/learnopencv/tree/master/Document-Scanner-Custom-Semantic-Segmentation-using-PyTorch-DeepLabV3 (tutorial is here https://learnopencv.com/deep-learning-based-document-segmentation-using-semantic-segmentation-deeplabv3-on-custom-dataset/)
2. https://github.com/Koldim2001/Unet-pytorch-training (tutorial is here https://www.youtube.com/watch?v=zpyzBR3MuT0)
3. https://github.com/murtazahassan/Document-Scanner (tutorial is here https://www.youtube.com/watch?v=ON_JubFRw8M)

Ground truth segmentation for training done using https://app.cvat.ai/
IDEs used - VScode & Pycharm
Setting up a virtual environment allows to import required libraries of relevant versions.

Workflow

1. Set up libraries, variables and preselected parameters
2. Open an image, convert to RGB, downsize it
3. Normalize if needed (min-max, manual double sigma or pytorch.transformation)
4. Create image tensor and supply it to pretrained model
5. Take the mask tensor and select the required layer that corresponds to the paper page
6. Erode-Dilate or Open-Close the image to get rid of artefacts
7. Find biggest contour and smooth it
8. Approximate the contour with a quadrilineal polygon
9. Unwarp the polygon to a rectangle
10. TO DO: expand conversion matrix and apply transformation to the larger source image
11. TO DO: save the processed image recording the key processing data for manual adjustment

Results

In the examples below the thin red line is a product of AI processing:

![DSC0386](https://github.com/user-attachments/assets/17dc3688-1a01-4211-aa98-8ae650587e78)
![DSC04569](https://github.com/user-attachments/assets/d10e8cee-09ea-4dfc-8459-e7bea0cc9a5c)
![DSC0390](https://github.com/user-attachments/assets/a0db30cf-778f-46b7-a5f0-bbf3a455fea5)
