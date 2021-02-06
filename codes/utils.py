# -*- coding: utf-8 -*-
'''
This script contains all the common functions and global variables and training
parameters used by all the other scripts.

@author: abhanjac
'''

import cv2, numpy as np, os, time, datetime, copy, sys, json, random, shutil
import matplotlib.pyplot as plt

################################################################################
################################################################################

# Global variables and parameters.

# Variables that will mark the points in the image by mouse click using the 
# callback function.
cBix, cBiy = -1, -1

################################################################################

classNameToIdx = {'nuts': 0, 'washers': 1, 'gears': 2, 'emptyBin': 3, \
                  'crankArmW': 4, 'crankShaft': 5}

classIdxToName = {0: 'nuts', 1: 'washers', 2: 'gears', 3: 'emptyBin', \
                  4: 'crankArmW', 5: 'crankShaft'}

savedImgFileExtn = '.png'
nClasses = len(classIdxToName)
inImgH, inImgW = 480, 640
leak = 0.1      # Parameter for the leaky relu.
learningRate = 0.0001
batchSize = 100
nEpochs = 100
threshProb = 0.5    # Beyond this threshold, a label element is 1 else 0.
nSavedCkpt = 5      # Number of checkpoints to keep at a time.
modelSaveInterval = 1   # Number of epochs after which the model will be saved.
ckptDirPath = 'saved_models'    # Location where the model checkpoints are saved.
#recordedMean = None  # Mean of train dataset.
#recordedStd = None  # Std of train dataset.
recordedMean = np.array([0.0, 0.0, 0.0])  # Mean of train dataset.
recordedStd = np.array([1.0, 1.0, 1.0])  # Std of train dataset.

modelName = 'cnn_model'
savedCkptName = modelName + '_ckpt'

################################################################################

# Colors meant for segmentation label images (if created).

classNameToColor = {'nuts':   [255,255,0], 'washers':   [190,190,250], \
                    'gears': [0,128,128], 'emptyBin':    [128,0,0], \
                    'crankArmW': [255,0,0], 'crankShaft': [255,0,255]}  # BGR format.

partialOBJcolor = [128,128,128]     # 'grey' for partial object color.

classIdxToColor = {0: [255,255,0], 1: [190,190,250], 2: [0,128,128], \
                   3: [128,0,0], 4: [255,0,0], 5: [255,0,255], 8: [0,0,255], \
                   9: [0,255,0]}  # BGR format.

colorDict = {'red':   [0,0,255],     'green':   [0,255,0],     'yellow':   [0,225,255],   \
             'blue':  [255,0,0],     'orange':  [48,130,245],  'purple':   [180,30,145],  \
             'cyan':  [255,255,0],   'magenta': [255,0,255],   'lime':     [60,245,210],  \
             'pink':  [190,190,250], 'teal':    [128,128,0],   'lavender': [255,190,230], \
             'brown': [40,110,170],  'beige':   [200,250,255], 'maroon':   [0,0,128],     \
             'mint':  [195,255,170], 'olive':   [0,128,128],   'coral':    [180,215,255], \
             'navy':  [128,0,0],     'white':   [255,255,255], 'black': [0,0,0]}  # BGR format.

# This dictionary gives the weights to be assigned to the pixels of different 
# colors, since the number of pixels of different colors is not the same which 
# may make the segmentation network biased toward any particular pixel color.
# The weight of each pixel is inversely proportional to the percentage of those 
# kind of pixels in the overall set of training images.
classIdxToSegColorWeight = {0: 57.496, 1: 109.076, 2: 58.761, 3: 104.468, \
                             4: 42.830, 5: 163.219, 6: 1177.220, 7: 467.332, \
                             8: 7.553,  9: 7.948,   10: 1.524}

# Average number of pixels per object. This can be used for counting the number
# of objects from a predicted segmentation map.
classIdxToAvgPixelsPerClassObj = {0: 4986.603, 1: 4994.226, 2: 4879.284, 3: 2744.476, \
                                   4: 6694.133, 5: 1982.704, 6: 243.549, 7: 613.506, \
                                   8: 1092.446, 9: 1040.280}

################################################################################

# Variables for the detector.

# List of anchor boxes (width, height). These sizes are relative to a 
# finalLayerH x finalLayerW sized image.
#anchorList = [[3.1, 3.1], [6.125, 3.1], [3.1, 6.125], \
               #[6.125, 6.125], [5.32, 5.32], [2.7, 2.7]]
#anchorList = [[4.5, 4.5], [3.5, 5.5], [5.5, 3.5], \
               #[5.5, 10], [10, 5.5], [4.5, 6.5], [6.5, 4.5]]
anchorList = [[2.0, 2.0], [2.0, 4.0], [4.0, 2.0], [4.0, 4.0], [3.0, 3.0], \
              [3.0, 10.0], [10.0, 3.0]]     # Obtained from height vs width bounding boxes plot.
nAnchors = len(anchorList)

ckptDirPathDetector = ckptDirPath + '_detector'
savedCkptNameDetector = modelName + '_detector' + '_ckpt'
finalLayerH, finalLayerW = 15, 20   # Dimension of the final conv layer activation map.

iouThresh = 0.7
iouThreshForMAPcalculation = 0.3
lambdaCoord = 5.0
lambdaNoObj = 0.5
lambdaClass = 2.5
threshProbDetection = 0.2    # Beyond this threshold, a label element is 1 else 0.

################################################################################
################################################################################

def robustInput(inType='str', inLwRange=-np.inf, inUpRange=np.inf, specificVal=[], 
                inputMsg='Enter input', errorMsg='Invalid input! Try Again.'):
    '''
    This function is used to take input from the user. When the parameters are 
    defined properly, then this function makes sure that the user cannot enter 
    invalid inputs. It is compatible with both python2 and python3. 
    The input type is defined by 'inType', which can be 'int', 'float', 'str' or 
    'bool'. However, it should be noted that 'int' and 'np.int' are different. 
    And similarly 'float' and 'np.float' are different.
    When input is int or float, the user can define ranges with inLwRange and 
    inUpRange. This way the function will only accept an input if it is within 
    the range inLwRange <= input < inUpRange. For example, if you have to ensure 
    the input is a +ve non-zero integer < 100, then put inLwRange = 1 and 
    inUpRange = 100. These parameters are not used if the input is an 'str. 
    If However your input should be a specific value, of a specific set of values, 
    then these are stored in the 'specificVal' list. E.g. if the input should only
    be 2, then define specificVal = [2]. In that case the inLwRange and inUpRange
    parameters will be ignored. The specificVal list can also be used for float 
    or str inputs or for more that one specific value. E.g. if the input has to 
    be one of the following: 2.2, 3.7, -4.5, 0.0, then define the 
    specificVal = [2.2, 3.7, -4.5, 0.0]. Same thing can be used for different 
    value of str and int inputs. Also, if you have a case where the input can be
    of different data types like the following: 2, 3, -4.5, 'ijkl', then define 
    specificVal = ['2', '3', '-4.5', 'ijkl']. I.e. put the valid values as strings
    as that is the default format in which the function takes in the raw input 
    from the user. Also, if your valid input is a string like 'hello', then set 
    specificVal = ['hello']. And input the value hello from the terminal, not 
    'hello'. I.e. dont use the quotes while inputing string inputs.
    The input message and error message can be modified to provide the user a 
    better hint of the type of input to be given.
    Usage example:
    a = robustInput(inType='int', inLwRange=0, inUpRange=250)
    print(a)
    '''
    quitMsg = '(or press \'Ctrl + \\\' to quit)'
    
    while True:     # Taking user input for number of rows of M.
        try:
            if sys.version[0] == '2':     # Python 2 uses 'raw_input' function.
                inp = raw_input('{} {}: '.format(inputMsg, quitMsg))
            elif sys.version[0] == '3':     # Python 3 uses 'input' function.
                inp = input('{} {}: '.format(inputMsg, quitMsg))
            
            # Analyzing the input.
            if inp == 'q' or inp == 'Q':    # If you want to exit on pressing q or Q.
                #exit(0)
                pass
            
            if inType == 'int':         inp = int(inp)
            elif inType == 'float':     inp = float(inp)
            elif inType == 'bool':      inp = bool(inp)
            elif inType == 'str':       inp = str(inp)

            if inType != 'str' and len(specificVal) == 0:
                if inp < inLwRange:     print('{} {}.'.format(errorMsg, quitMsg))
                elif inp >= inUpRange:  print('{} {}.'.format(errorMsg, quitMsg))
                else:                   break
            else:   # When input is 'str' or there are values inside the specificVal.
                if inp in specificVal:  break
                else:       print('{} {}.'.format(errorMsg, quitMsg))
                #else:       print(inp, specificVal[0], inp==specificVal[0])
                
        except ValueError:
            print('{} {} 444.'.format(errorMsg, quitMsg))

    return inp
    
################################################################################
################################################################################

def horiFlipSampleAndMask(sample=None, mask=None):
    '''
    Performs horizontal flips on input sample and mask.
    '''
    if sample is None or mask is None:
        print('\nERROR: one or more input arguments missing ' \
               'in horiFlipSampleAndMask. Aborting.\n')
        sys.exit()    

    newSample = cv2.flip(sample, 1)      # Flip around y axis.
    newMask = cv2.flip(mask, 1)

    return newSample, newMask

################################################################################
################################################################################

def vertFlipSampleAndMask(sample=None, mask=None):
    '''
    Performs vertical flips on input sample and mask.
    '''
    if sample is None or mask is None:
        print('\nERROR: one or more input arguments missing ' \
               'in vertFlipSampleAndMask. Aborting.\n')
        sys.exit()    

    newSample = cv2.flip(sample, 0)      # Flip around x axis.
    newMask = cv2.flip(mask, 0)

    return newSample, newMask

################################################################################
################################################################################

def random90degFlipSampleAndMask(sample=None, mask=None):
    '''
    Performs 90 deg flips on input sample and mask randomly.
    '''
    if sample is None or mask is None:
        print('\nERROR: one or more input arguments missing ' \
               'in random90degFlipSampleAndMask. Aborting.\n')
        sys.exit()    
        
    # Now the selection of whether the flip should be by 90, 180 or 270
    # deg, is done randomly (with equal probablity).
    number1 = np.random.randint(100)

    if number1 < 33:
        # Flip by 90 deg (same as horizontal flip + transpose).
        newSample = cv2.transpose(cv2.flip(sample, 1))
        newMask = cv2.transpose(cv2.flip(mask, 1))
        
    elif number1 >= 33 and number1 < 66:
        # Flip by 180 deg (same as horizontal flip + vertical flip).
        newSample = cv2.flip(sample, -1)
        newMask = cv2.flip(mask, -1)
        
    else:   # Flip by 270 deg (same as vertical flip + transpose).
        newSample = cv2.transpose(cv2.flip(sample, 0))
        newMask = cv2.transpose(cv2.flip(mask, 0))
        
    # Also, finding the bbox for the rotated sample from the mask.
    h, w, _ = newSample.shape

    return newSample, newMask, w, h

################################################################################
################################################################################

def randomRotationSampleAndMask(sample=None, mask=None):
    '''
    Performs rotation of the sample by arbitrary angles.
    '''
    if sample is None or mask is None:
        print('\nERROR: one or more input arguments missing ' \
               'in randomRotationSampleAndMask. Aborting.\n')
        sys.exit()    
        
    # During rotation by arbitrary angles, the sample first needs to be
    # pasted on a bigger blank array, otherwise it will get cropped 
    # due to rotation.
    sampleH, sampleW, _ = sample.shape
    
    # The length of the side of the new blank array should be equal to
    # the diagonal length of the sample, so that any rotation can be 
    # accomodated.
    sideLen = int(np.sqrt(sampleH **2 + sampleW **2) + 1)
    blankArr = np.zeros((sideLen, sideLen, 3), dtype=np.uint8)
    
    # Top left corner x and y coordinates of the region where the 
    # sample will be affixed on the blank array.
    sampleTlY = int((sideLen - sampleH) / 2)
    sampleTlX = int((sideLen - sampleW) / 2)
    
    # Affixing the sample on the blank array.
    blankArr[sampleTlY : sampleTlY + sampleH, \
              sampleTlX : sampleTlX + sampleW, :] = sample
             
    newSample = copy.deepcopy(blankArr)
    
    # Rotation angle is determined at random between 0 to 360 deg.
    angle = np.random.randint(360)
    
    # Create the rotation matrix and rotate the sample.
    M = cv2.getRotationMatrix2D((sideLen/2, sideLen/2), angle, 1)
    newSample = cv2.warpAffine(newSample, M, (sideLen, sideLen))
    
    # Modifying the mask in the same manner.
    blankArr1 = np.zeros((sideLen, sideLen, 3), dtype=np.uint8)
    blankArr1[sampleTlY : sampleTlY + sampleH, \
               sampleTlX : sampleTlX + sampleW, :] = mask
    newMask = copy.deepcopy(blankArr1)
    newMask = cv2.warpAffine(newMask, M, (sideLen, sideLen))
    
    # Also, finding the bbox for the rotated sample from the mask contour.
    newMask1 = cv2.cvtColor(newMask, cv2.COLOR_BGR2GRAY)
    returnedTuple = cv2.findContours(newMask1, method=cv2.CHAIN_APPROX_SIMPLE, \
                                                mode=cv2.RETR_LIST)
    contours = returnedTuple[-2]
    
    x, y, w, h = cv2.boundingRect(contours[0])

    return newSample, newMask, w, h

#################################################################################
#################################################################################

def fixSampleToBg(sample=None, mask=None, bg=None, tlX=None, tlY=None):
    '''
    This function takes in a sample, its mask, a background and the top left 
    corner x and y coordinates of the region where the sample will be affixed 
    on the background. The background, sample and mask has to be in proper 
    shape and should have already undergone whatever data augmentation was 
    necessary. This function does not handle those processings.
    It also returns the center and bbox of the object after pasting.
    '''
    if sample is None or mask is None or bg is None or tlX is None or tlY is None:
        print('\nERROR: one or more input arguments missing ' \
               'in fixSampleToBg. Aborting.\n')
        sys.exit()    

    sampleH, sampleW, _ = sample.shape
    bgH, bgW, _ = bg.shape
    
    invMask = (255 - mask) / 255
    
    # There are some bounding regions surrounding the actual object in the 
    # sample image. When this sample is affixed, we do not want these 
    # surrounding regions to replace the corresponding pixels in the bg image.
    # So the inverted mask is used to copy the pixels of the background 
    # corresponding to this bounding region and later paste those back after 
    # the sample has been affixed.
    
    # Now it may happen that the tlX and tlY are such that the sample will get
    # clipped at the image boundary. So determining the x and y coordinates of 
    # the bottom right corner as well.
    brY, brX = min(tlY + sampleH, bgH), min(tlX + sampleW, bgW)
    
    bgRegionToBeReplaced = bg[tlY : brY, tlX : brX, :]
    bgRegTBRh, bgRegTBRw, _ = bgRegionToBeReplaced.shape
    
    # While taking out the bounding region (or the object only region), it is to 
    # be made sure that the size of these regions and the invMask (or mask) are 
    # same, otherwise this may throw errors in the cases where the sample is 
    # getting clipped.
    boundingRegion = bgRegionToBeReplaced * invMask[0 : bgRegTBRh, \
                                                     0 : bgRegTBRw, :]
    boundingRegion = np.asarray(boundingRegion, dtype=np.uint8)
    
    # Taking out only the object part of the sample using the mask.
    onlyObjectRegionOfSample = cv2.bitwise_and(sample, mask)
    onlyObjectRegionOfSample = onlyObjectRegionOfSample[0 : bgRegTBRh, \
                                                         0 : bgRegTBRw, :]
    
    # Now pasting the sample onto the bg along with the pixels of bg 
    # corresponding to the blank region (which is called bounding region in this 
    # case).
    img = copy.deepcopy(bg)
    img[tlY : brY, tlX : brX, :] = onlyObjectRegionOfSample + boundingRegion
    
    # The location where the sample is affixed, this is the center pixel of this
    # region, not the top left corner.
    posY = round((brY + tlY) * 0.5)
    posX = round((brX + tlX) * 0.5)
    bboxH = brY - tlY
    bboxW = brX - tlX
    
    return img, posX, posY, bboxW, bboxH

################################################################################
################################################################################

def createTightMasksAndSamples(imgDir=None, maskDir=None, replaceOriginal=False):
    '''
    Samples are affixed on the background images using masks. But sometimes, 
    the masks have a thick black border around the white region where the object
    is located. Hence in that case if the dimension of the mask image is taken 
    as the dimension of the bounding box, then the bounding box is not very 
    tight around the object. Hence to take care of these cases, the thick border 
    around the white portion of the mask is removed along with the removal of 
    the corresponding region on the sample image. This function does that.
    If the replaceOriginal flag is True, then the function will save the new 
    images and masks replacing the old ones. If this flag is false, then the 
    function will create new folders and save the new images and masks there.
    If original folder names were 'crankArmW' and 'crankArmW_masks' then the new 
    folder name will be 'crankArmW_new' and 'crankArmW_masks_new'. There may be 
    subfolders like 'train', 'test', 'valid' inside the original 'crankArmW' and 
    'crankArmW_masks' folders. Hence identical folders 'train', 'test', 'valid' 
    will be created inside the new folders as well with the same names for the 
    subfolders.
    '''
    if imgDir is None or maskDir is None:
           print('\nERROR: one or more input arguments missing ' \
                  'in createTightMasksAndSamples. Aborting.\n')
           sys.exit()
           
    imgList = os.listdir(imgDir)
    nImgs = len(imgList)
    
    for idx, imgName in enumerate(imgList):
        img = cv2.imread(os.path.join(imgDir, imgName))
        imgH, imgW, _ = img.shape

        # If name of sample is Eosinophil_1.png, the name of the 
        # corresponding mask is Eosinophil_1_mask.png
        maskName = imgName.split('.')[0] + '_mask.' + imgName.split('.')[1]
        mask = cv2.imread(os.path.join(maskDir, maskName))
        
        # It may happen that the mask itself already tightly bounds the object.
        # So, to make the method more generalized, a black border is appended to all 
        # sides of the mask and then contours are extracted anyway to find the bounding 
        # boxes. This way the cases of both tight masks and not-tight masks are dealt 
        # with in the same manner.
        
        # Find the bounding box around the biggest contour in the mask image, 
        # and return that as the bounding box of the object so that this box 
        # tightly bounds the object and not include any black boundary around 
        # the white portion of the object.
        # But to include the cases where the mask is already tightly bounding the 
        # object, black borders are appended to all sides of the mask before 
        # finding the contours.
        if len(mask.shape) == 3:    # If the mask has 3 channels.
            appendedMask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)   # Convert to 1 channel.
        else:
            appendedMask = copy.deepcopy(appendedMask)
        
        borderThickness = 10
        leftRightBorder = np.zeros((imgH, borderThickness), dtype=np.uint8)
        appendedMask = np.hstack((leftRightBorder, appendedMask, leftRightBorder))
        topBotBorder = np.zeros((borderThickness, imgW+borderThickness*2), dtype=np.uint8)
        appendedMask = np.vstack((topBotBorder, appendedMask, topBotBorder))    
        
        returnedTuple = cv2.findContours(appendedMask, mode=cv2.RETR_TREE, \
                                         method=cv2.CHAIN_APPROX_SIMPLE)
        contours = returnedTuple[-2]
        
        biggestContour = contours[0]
        for c in contours:      # Finding the biggest contour.
            if cv2.contourArea(c) > cv2.contourArea(biggestContour):
                biggestContour = c
        
        # Finding the bounding rectangle around the biggestContour.
        bboxX, bboxY, bboxW, bboxH = cv2.boundingRect(biggestContour)
    
        # Now mapping the bboxX and bboxY to the dimension of the original mask.
        # This is done by subtracting the borderThickness from the bboxX 
        # and bboxY values.
        bboxX, bboxY = bboxX - borderThickness, bboxY - borderThickness
        
        # Now crop out only the region inside the bounding box from the sample 
        # and the mask save it as the new tight sample and tight mask.
        newImg = img[bboxY : bboxY + bboxH, bboxX : bboxX + bboxW, :]
        newMask = mask[bboxY : bboxY + bboxH, bboxX : bboxX + bboxW, :]
        
        if replaceOriginal:
            imgSaveDir, maskSaveDir = imgDir, maskDir
        else:
            mainFolderName = '/'.join(imgDir.split('/')[:-1]) + '_new'
            subFolderName = imgDir.split('/')[-1]
            imgSaveDir = os.path.join(mainFolderName, subFolderName)
        
            mainFolderName = '/'.join(maskDir.split('/')[:-1]) + '_new'
            subFolderName = maskDir.split('/')[-1]
            maskSaveDir = os.path.join(mainFolderName, subFolderName)
            
            # Create the directories if not present.
            if not os.path.exists(imgSaveDir):    os.makedirs(imgSaveDir)
            if not os.path.exists(maskSaveDir):    os.makedirs(maskSaveDir)
        
        cv2.imwrite(os.path.join(imgSaveDir, imgName), newImg)
        cv2.imwrite(os.path.join(maskSaveDir, maskName), newMask)
        
        print('[{}\{}] Saved image {}.'.format(idx+1, nImgs, imgName))

################################################################################
################################################################################

def blankBackground(bgLoc=None, imgSaveLoc=None, labelSaveLoc=None, \
                     nImgs=None, imgH=None, imgW=None,
                     saveNameSuffix=None, createSegmentLabelImg=False, \
                     segmentSaveLoc=None):
    '''
    This function creates images where there are no wbc cells. Only a background
    image of rbc cells.
    These images are created by taking backgrounds from bgLoc.
    The backgrounds are selected randomly from the available collection. 
    Total number of images created is nImgs. Images are saved in the imgSaveLoc.
    The labels of the corresponding images are also created as json files in 
    the labelSaveLoc.
    imgH and imgW defines the size of the image to be created.
    The saveNameSuffix is a string, that will be appended to the name of the 
    image file while saving. This is important to identify the dataset from 
    where the image has been synthesized.
    The createSegmentLabelImg indicates if a semantic segmentation label image 
    has to be created as well. The colors of the segments for different objects 
    are mentioned in the global variables. Segment save location is also provided.
    '''
    
    if bgLoc is None or imgSaveLoc is None or labelSaveLoc is None \
       or nImgs is None or imgH is None or imgW is None or saveNameSuffix is None:
           print('\nERROR: one or more input arguments missing ' \
                  'in blankBackground. Aborting.\n')
           sys.exit()
    
    if createSegmentLabelImg:
        if segmentSaveLoc is None:
            print('\nERROR: one or more input arguments missing ' \
                   'in blankBackground for segments. Aborting.\n')
            sys.exit()
    
################################################################################
        
    # Checking if there is any label file for the bg images present or not. 
    # These files will include rbc annotations, if present.
    imgFolderParentDir = '\\'.join(bgLoc.split('\\')[:-1])
    imgFolderName = bgLoc.split('\\')[-1]
    labelFolderName = imgFolderName + '_labels'
    labelFolderLoc = os.path.join(imgFolderParentDir, labelFolderName)
    
    if createSegmentLabelImg:
        bgSegmentFolderName = imgFolderName + '_segments'
        bgSegmentFolderLoc = os.path.join(imgFolderParentDir, bgSegmentFolderName)

################################################################################
    
    # Number of files already existing in the imgSaveLoc is calculated. This 
    # will be used to assign the index to the file while saving.
    nAlreadyExistingFiles = len(os.listdir(imgSaveLoc))
    
    bgList = []
    
    # Creating the images.    
    for i in range(nImgs):
        # Fill the lists if they are empty.
        # As a sample and a bg is used for creating an image, they are deleted
        # from this list. So if this list gets empty, then it is reinitialized.
        if len(bgList) == 0:      bgList = os.listdir(bgLoc)
        
        # Select a background at random.
        bgIdx = np.random.randint(len(bgList))
        bgName = bgList[bgIdx]
        bg = cv2.imread(os.path.join(bgLoc, bgName))
        
        # Remove the entry of this bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing bg images in the lists are used up and the lists become empty.
        bgList.pop(bgIdx)

################################################################################
        
        # Setting the background.
        
        # It may happen that the bg image is larger than size imgH x imgW.
        # In that case, a imgH x imgW region is cropped out from the bg image.
        bgH, bgW, _ = bg.shape
        
        # Determining the x and y of the top left corner of the region to be
        # cropped out from the bg image.
        bgTlY = np.random.randint(bgH - imgH) if bgH > imgH else 0
        bgTlX = np.random.randint(bgW - imgW) if bgW > imgW else 0
        
        # IMPORTANT: The bg image must be larger or equal in size to imgH x imgW.
        newBg = bg[bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW]
        
        newBgH, newBgW, _ = newBg.shape

################################################################################

        # Also doing the same processing for the segmented image label.
        if createSegmentLabelImg:
            bgSegName = 'seg_' + '_'.join(bgName.split('_')[1:])
            bgSegImg = cv2.imread(os.path.join(bgSegmentFolderLoc, bgSegName))
            
            newBgSegImg = bgSegImg[bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW]
            
################################################################################

        # Saving the image.
        idx = nAlreadyExistingFiles + i    # This is the image index.
        
        imgSaveName = 'back' + '_' + \
                      saveNameSuffix + '_' + str(idx) + savedImgFileExtn
        # The file extension is explicitly specified here because, many of the 
        # background images have .jpg or .png file extension as well. So to 
        # make the saved file consistent, this is specified.
                
        cv2.imwrite(os.path.join(imgSaveLoc, imgSaveName), newBg)
        
        # Saving the segmented image label as well if createSegmentLabelImg is True.
        if createSegmentLabelImg:
            segImgSaveName = 'back' + '_seg' + '_' + \
                             saveNameSuffix + '_' + str(idx) + savedImgFileExtn
            cv2.imwrite(os.path.join(segmentSaveLoc, segImgSaveName), newBgSegImg)
                
        # Creating the label json file.
        labelSaveName = 'back' + '_' + \
                        saveNameSuffix + '_' + str(idx) + '.json'
                
        infoDict = {}
        
        with open(os.path.join(labelSaveLoc, labelSaveName), 'w') as infoFile:
            json.dump(infoDict, infoFile, indent=4, separators=(',', ': '))

################################################################################

        #cv2.imshow('image', newBg)
        # Show the segment label as well if the createSegmentLabelImg is True.
        if createSegmentLabelImg:   cv2.imshow('segment label', newBgSegImg)
        cv2.waitKey(30)

################################################################################
################################################################################

def singleInstance(sampleLoc=None, maskLoc=None, \
                    bgLoc=None, imgSaveLoc=None, labelSaveLoc=None, \
                    nImgs=None, imgH=None, imgW=None,
                    saveNameSuffix=None, do90degFlips=False, \
                    doHoriFlip=False, doVertFlip=False, doRandomRot=False, \
                    createSegmentLabelImg=False, segmentSaveLoc=None):
    '''
    This function creates images where an object of a certain class appears 
    just one time.
    These images are created by taking object samples from sampleLoc and 
    backgrounds from bgLoc by affixing the object samples onto the backgrounds.
    The maskLoc holds the masks for the sample, but this is optional. If there
    is no maskLoc provided, then the samples are just pasted as they are, 
    otherwise the corresponding mask is used while pasting the sample.
    The samples and backgrounds are selected randomly from the available 
    collection in their respective locations. Total number of images created is 
    nImgs. These images are saved in the imgSaveLoc.
    The labels of the corresponding images are also created as json files in 
    the labelSaveLoc.
    imgH and imgW defines the size of the image to be created.
    The saveNameSuffix is a string, that will be appended to the name of the 
    image file while saving. This is important to identify the dataset from 
    where the image has been synthesized.
    The flag do90degFlips indicates whether the sample should undergo rotations
    by multiples of 90 deg (randomly), while getting affixed on the bg image.
    The flag doRandomRot indicates whether the sample should undergo rotations
    by random angles, while getting affixed on the bg image.
    Flags doHoriFlip and doVertFlip indicates if the sample should be flipped 
    horizontally or vertically (randomly) before getting affixed on bg image.
    The createSegmentLabelImg indicates if a semantic segmentation label image 
    has to be created as well. The colors of the segments for different objects 
    are mentioned in the global variables. Segment save location is also provided.
    '''
    
    if sampleLoc is None or bgLoc is None \
       or imgSaveLoc is None or labelSaveLoc is None or nImgs is None \
       or imgH is None or imgW is None or saveNameSuffix is None:
           print('\nERROR: one or more input arguments missing ' \
                  'in singleInstance. Aborting.\n')
           sys.exit()
           
    if createSegmentLabelImg:
        if segmentSaveLoc is None:
            print('\nERROR: one or more input arguments missing ' \
                   'in singleInstance for segments. Aborting.\n')
            sys.exit()
    
    # Flag indicating mask present.
    maskPresent = False if maskLoc is None else True
    
################################################################################
        
    # Checking if there is any label file for the bg images present or not. 
    # These files will include rbc annotations, if present.
    imgFolderParentDir = '\\'.join(bgLoc.split('\\')[:-1])
    imgFolderName = bgLoc.split('\\')[-1]
    labelFolderName = imgFolderName + '_labels'
    labelFolderLoc = os.path.join(imgFolderParentDir, labelFolderName)
    
    if createSegmentLabelImg:
        bgSegmentFolderName = imgFolderName + '_segments'
        bgSegmentFolderLoc = os.path.join(imgFolderParentDir, bgSegmentFolderName)

################################################################################
    
    # Number of files already existing in the imgSaveLoc is calculated. This 
    # will be used to assign the index to the file while saving.
    nAlreadyExistingFiles = len(os.listdir(imgSaveLoc))
    
    bgList, sampleList = [], []
    
    # Creating the images.    
    for i in range(nImgs):
        # Fill the lists if they are empty.
        # As a sample and a bg is used for creating an image, they are deleted
        # from this list. So if this list gets empty, then it is reinitialized.
        if len(bgList) == 0:      bgList = os.listdir(bgLoc)
        if len(sampleList) == 0:      sampleList = os.listdir(sampleLoc)
        
        # Select a sample at random.
        sampleIdx = np.random.randint(len(sampleList))
        
        sampleName = sampleList[sampleIdx]
        sample = cv2.imread(os.path.join(sampleLoc, sampleName))
        
        className = sampleName.split('_')[0]
        
        if maskPresent:     
            # If name of sample is Eosinophil_1.png, the name of the 
            # corresponding mask is Eosinophil_1_mask.png
            maskName = sampleName.split('.')[0] + '_mask.' + sampleName.split('.')[1]        
            mask = cv2.imread(os.path.join(maskLoc, maskName))
        else:
            # If mask is not present then a dummy mask is created which is just
            # a blank array of 255s, of the same type and shape as sample.
            # This makes all future processing easier and also prevents the 
            # check for maskPresent flag every time.
            mask = np.ones(sample.shape) * 255
            mask = np.asarray(mask, dtype=np.uint8)
        
        # The bg and sample idxs are determined separately because the number of
        # available samples and bg may be different.
        bgIdx = np.random.randint(len(bgList))
        bgName = bgList[bgIdx]
        bg = cv2.imread(os.path.join(bgLoc, bgName))
        
        # Remove the entry of this sample and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        sampleList.pop(sampleIdx)
        bgList.pop(bgIdx)

################################################################################
        
        # Setting the background.
        
        # It may happen that the bg image is larger than size imgH x imgW.
        # In that case, a imgH x imgW region is cropped out from the bg image.
        bgH, bgW, _ = bg.shape
        
        # Determining the x and y of the top left corner of the region to be
        # cropped out from the bg image.
        bgTlY = np.random.randint(bgH - imgH) if bgH > imgH else 0
        bgTlX = np.random.randint(bgW - imgW) if bgW > imgW else 0
        
        # IMPORTANT: The bg image must be larger or equal in size to imgH x imgW.
        newBg = bg[bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW]
        
        newBgH, newBgW, _ = newBg.shape
        
################################################################################

        # Also doing the same processing for the segmented image label.
        if createSegmentLabelImg:
            bgSegName = 'seg_' + '_'.join(bgName.split('_')[1:])
            bgSegImg = cv2.imread(os.path.join(bgSegmentFolderLoc, bgSegName))
            
            newBgSegImg = bgSegImg[bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW]

################################################################################
                
        if doHoriFlip or doVertFlip or do90degFlips or doRandomRot:

            # Augmenting the samples before affixing onto the background.
            
            # There are altogether 4 kinds of augmentation that this function can 
            # do, doRandomRot, doHoriFlip, doVertFlip, do90degFlips and no 
            # augmentation.
            # What kind of augmentation is to be done for this sample is chosen 
            # at random with a equal probability (20% for each type).
            # However, if the type of augmentation chosen doen not have it's 
            # corresponding flag True, then no augmentation is done.
    
            number = np.random.randint(100)
            
            # Horizontal flip sample.
            
            if number < 20 and doHoriFlip:
                newSample, newMask = horiFlipSampleAndMask(sample, mask)
                bboxH, bboxW, _ = newSample.shape

################################################################################
    
            # Vertical flip sample.
    
            elif number >= 20 and number < 40 and doVertFlip:
                newSample, newMask = vertFlipSampleAndMask(sample, mask)
                bboxH, bboxW, _ = newSample.shape

################################################################################
    
            # 90 deg flip sample.
    
            elif number >= 40 and number < 60 and do90degFlips:
                # Now the selection of whether the flip should be by 90, 180 or 270
                # deg, is done randomly (with equal probablity).                
                newSample, newMask, bboxW, bboxH = random90degFlipSampleAndMask(sample, mask)
            
################################################################################
    
            # Rotation by random angles sample.
    
            elif number >= 60 and number < 80 and doRandomRot:
                # During rotation by arbitrary angles, the sample first needs to be
                # pasted on a bigger blank array, otherwise it will get cropped 
                # due to rotation.
                newSample, newMask, bboxW, bboxH = randomRotationSampleAndMask(sample, mask)
                
################################################################################
                
            # No augmentation sample.
            
            else:
                newSample, newMask = sample, mask
                bboxH, bboxW, _ = newSample.shape
        
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint(newBgH - newSampleH)
            tlX = np.random.randint(newBgW - newSampleW)
            
            # Fixing the sample onto the background.
            image, posX, posY, _, _ = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

            # Create the segmented label image as well if createSegmentLabelImg is True:
            if createSegmentLabelImg:
                sampleColor = classNameToColor[className]
                sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
                    
################################################################################
    
        # If both the clipSample and the other augmentation flags are False, 
        # then no augmentation is performed.

        else:
            newSample, newMask = sample, mask
        
            # x, y of top left corner of the region where sample will be pasted.
            newSampleH, newSampleW, _ = newSample.shape
            tlY = np.random.randint(newBgH - newSampleH)
            tlX = np.random.randint(newBgW - newSampleW) 
            # Fixing the sample onto the background.
            image, posX, posY, bboxW, bboxH = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

            # Create the segmented label image as well if createSegmentLabelImg is True:
            if createSegmentLabelImg:
                sampleColor = classNameToColor[className]
                sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
            
#################################################################################
#
#        cv2.imshow('sample', sample)
#        cv2.imshow('newSample', newSample)
#        cv2.imshow('mask', mask)
#        cv2.imshow('newMask', newMask)
#        cv2.imshow('image', image)
#        cv2.waitKey(0)
#                     
################################################################################

        # Saving the image.
        idx = nAlreadyExistingFiles + i    # This is the image index.
        
        imgSaveName = className + '_' + \
                      saveNameSuffix + '_' + str(idx) + savedImgFileExtn
        cv2.imwrite(os.path.join(imgSaveLoc, imgSaveName), image)
        
        # Saving the segmented image label as well if createSegmentLabelImg is True.
        if createSegmentLabelImg:
            segImgSaveName = className + '_seg' + '_' + \
                             saveNameSuffix + '_' + str(idx) + savedImgFileExtn
            cv2.imwrite(os.path.join(segmentSaveLoc, segImgSaveName), segImg)
        
        # Creating the label json file.
        labelSaveName = className + '_' + \
                        saveNameSuffix + '_' + str(idx) + '.json'
        
        if className in classNameToIdx:
            classIdx = classNameToIdx[className]
        # If the partialOBJ className is not included in the classNameToIdx 
        # dictionary, then it is assigned a new classIdx value.
        elif className == 'partialOBJ':
            classIdx = nClasses
        
        infoDict = {}
        
        with open(os.path.join(labelSaveLoc, labelSaveName), 'w') as infoFile:
            
################################################################################

            # Make sure the coordinates are inside the boundaries of the image.
            if posX >= imgW:      posX = imgW - 1
            if posX < 0:            posX = 0
            if posY >= imgH:      posY = imgH - 1
            if posY < 0:            posY = 0
            tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
            brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
            if tlX < 0:            tlX = 0
            if tlY < 0:            tlY = 0
            if brX >= imgW:      brX = imgW - 1
            if brY >= imgH:      brY = imgH - 1
            bboxW, bboxH = int(brX - tlX), int(brY - tlY)   # Update box size.
            
################################################################################

            infoDict[0] = {
                            'className': className, 'classIdx': classIdx, \
                            'posX': int(posX), 'posY': int(posY), \
                            'bboxW': bboxW, 'bboxH': bboxH, \
                            'tlX': int(tlX), 'tlY': int(tlY), \
                            'samplePath': os.path.join(sampleLoc, sampleName), \
                            'bgPath': os.path.join(bgLoc, bgName) \
                         }
            
            json.dump(infoDict, infoFile, indent=4, separators=(',', ': '))

################################################################################

        for k, v in infoDict.items():
            cv2.circle(image, (v['posX'], v['posY']), 2, (0,255,0), 2)
            if v['className'] != '_':
                cv2.rectangle(image, (v['tlX'], v['tlY']), (v['tlX']+v['bboxW'], \
                                       v['tlY']+v['bboxH']), (0,255,0), 2)
        #cv2.imshow('image', image)
        # Show the segment label as well if the createSegmentLabelImg is True.
        if createSegmentLabelImg:   cv2.imshow('segment label', segImg)
        cv2.waitKey(0)

################################################################################
################################################################################
        
def doubleInstance(sampleLoc1=None, sampleLoc2=None, maskLoc1=None, \
                    maskLoc2=None, bgLoc=None, imgSaveLoc=None, labelSaveLoc=None, \
                    nImgs=None, imgH=None, imgW=None,
                    saveNameSuffix=None, do90degFlips=False, \
                    doHoriFlip=False, doVertFlip=False, doRandomRot=False, \
                    createSegmentLabelImg=False, segmentSaveLoc=None):
    '''
    This function creates images where an object from each of the folders 
    sampleLoc1 and sampleLoc2 are randomly selected and pasted on a background
    seleced from the bgLoc folder. So there are two instances of wbc in the same 
    image.
    The maskLoc1 and maskLoc2 holds the masks for the sample of sampleLoc1 and 
    sampleLoc2 respectively, but these are optional. If there
    are no maskLoc1 or maskLoc2 provided, then the corresponding samples are 
    just pasted as they are, otherwise the corresponding mask is used while 
    pasting the sample.
    The samples and backgrounds are selected randomly from the available 
    collection in their respective locations. Total number of images created is 
    nImgs. These images are saved in the imgSaveLoc.
    The labels of the corresponding images are also created as json files in 
    the labelSaveLoc.
    imgH and imgW defines the size of the image to be created.
    The saveNameSuffix is a string, that will be appended to the name of the 
    image file while saving. This is important to identify the dataset from 
    where the image has been synthesized.
    The flag do90degFlips indicates whether the sample should undergo rotations
    by multiples of 90 deg (randomly), while getting affixed on the bg image.
    The flag doRandomRot indicates whether the sample should undergo rotations
    by random angles, while getting affixed on the bg image.
    Flags doHoriFlip and doVertFlip indicates if the sample should be flipped 
    horizontally or vertically (randomly) before getting affixed on bg image.
    The createSegmentLabelImg indicates if a semantic segmentation label image 
    has to be created as well. The colors of the segments for different objects 
    are mentioned in the global variables. Segment save location is also provided.
    '''
    
    if sampleLoc1 is None or sampleLoc2 is None or bgLoc is None \
       or imgSaveLoc is None or labelSaveLoc is None or nImgs is None \
       or imgH is None or imgW is None or saveNameSuffix is None:
           print('\nERROR: one or more input arguments missing ' \
                  'in doubleInstance. Aborting.\n')
           sys.exit()
    
    if createSegmentLabelImg:
        if segmentSaveLoc is None:
            print('\nERROR: one or more input arguments missing ' \
                   'in doubleInstance for segments. Aborting.\n')
            sys.exit()
    
    # Flag indicating mask present.
    maskPresent1 = False if maskLoc1 is None else True
    maskPresent2 = False if maskLoc2 is None else True

################################################################################
        
    # Checking if there is any label file for the bg images present or not. 
    # These files will include rbc annotations, if present.
    imgFolderParentDir = '\\'.join(bgLoc.split('\\')[:-1])
    imgFolderName = bgLoc.split('\\')[-1]
    labelFolderName = imgFolderName + '_labels'
    labelFolderLoc = os.path.join(imgFolderParentDir, labelFolderName)
    
    if createSegmentLabelImg:
        bgSegmentFolderName = imgFolderName + '_segments'
        bgSegmentFolderLoc = os.path.join(imgFolderParentDir, bgSegmentFolderName)

################################################################################

    # Number of files already existing in the imgSaveLoc is calculated. This 
    # will be used to assign the index to the file while saving.
    nAlreadyExistingFiles = len(os.listdir(imgSaveLoc))
    
    bgList, sampleList1, sampleList2 = [], [], []
    
    # Creating the images.    
    for i in range(nImgs):
        # Fill the lists if they are empty.
        # As a sample and a bg is used for creating an image, they are deleted
        # from this list. So if this list gets empty, then it is reinitialized.
        if len(bgList) == 0:      bgList = os.listdir(bgLoc)
        if len(sampleList1) == 0:      sampleList1 = os.listdir(sampleLoc1)
        if len(sampleList2) == 0:      sampleList2 = os.listdir(sampleLoc2)
        
        # Select a sample1 at random.
        sampleIdx1 = np.random.randint(len(sampleList1))
        sampleIdx2 = np.random.randint(len(sampleList2))
        
        sampleName1 = sampleList1[sampleIdx1]
        sampleName2 = sampleList2[sampleIdx2]
        sample1 = cv2.imread(os.path.join(sampleLoc1, sampleName1))
        sample2 = cv2.imread(os.path.join(sampleLoc2, sampleName2))
                
        className1 = sampleName1.split('_')[0]
        className2 = sampleName2.split('_')[0]
        
        if maskPresent1:
            # If name of sample is Eosinophil_1.png, the name of the 
            # corresponding mask is Eosinophil_1_mask.png
            maskName1 = sampleName1.split('.')[0] + '_mask.' + sampleName1.split('.')[1]
            mask1 = cv2.imread(os.path.join(maskLoc1, maskName1))
        else:
            # If mask is not present then a dummy mask is created which is just
            # a blank array of 255s, of the same type and shape as sample.
            # This makes all future processing easier and also prevents the 
            # check for maskPresent flag every time.
            mask1 = np.ones(sample1.shape) * 255
            mask1 = np.asarray(mask1, dtype=np.uint8)
        
        if maskPresent2:
            # If name of sample is Eosinophil_1.png, the name of the 
            # corresponding mask is Eosinophil_1_mask.png
            maskName2 = sampleName2.split('.')[0] + '_mask.' + sampleName2.split('.')[1]
            mask2 = cv2.imread(os.path.join(maskLoc2, maskName2))
        else:
            # If mask is not present then a dummy mask is created which is just
            # a blank array of 255s, of the same type and shape as sample.
            # This makes all future processing easier and also prevents the 
            # check for maskPresent flag every time.
            mask2 = np.ones(sample2.shape) * 255
            mask2 = np.asarray(mask2, dtype=np.uint8)

        # The bg and sample idxs are determined separately because the number of
        # available samples and bg may be different.
        bgIdx = np.random.randint(len(bgList))
        bgName = bgList[bgIdx]
        bg = cv2.imread(os.path.join(bgLoc, bgName))
        
        # Remove the entry of these samples and bg from the respective lists so 
        # that they are not used again. It will only be used again if all the 
        # existing samples in the lists are used up and the lists become empty.
        sampleList1.pop(sampleIdx1)
        sampleList2.pop(sampleIdx2)
        bgList.pop(bgIdx)

################################################################################
        
        # Setting the background.
        
        # It may happen that the bg image is larger than size imgH x imgW.
        # In that case, a imgH x imgW region is cropped out from the bg image.
        bgH, bgW, _ = bg.shape
        
        # Determining the x and y of the top left corner of the region to be
        # cropped out from the bg image.
        bgTlY = np.random.randint(bgH - imgH) if bgH > imgH else 0
        bgTlX = np.random.randint(bgW - imgW) if bgW > imgW else 0
        
        # IMPORTANT: The bg image must be larger or equal in size to imgH x imgW.
        newBg = bg[bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW]
        
        newBgH, newBgW, _ = newBg.shape
        
################################################################################

        # Also doing the same processing for the segmented image label.
        if createSegmentLabelImg:
            bgSegName = 'seg_' + '_'.join(bgName.split('_')[1:])
            bgSegImg = cv2.imread(os.path.join(bgSegmentFolderLoc, bgSegName))
            
            newBgSegImg = bgSegImg[bgTlY : bgTlY + imgH, bgTlX : bgTlX + imgW]
            
################################################################################

        # Since two samples will be pasted here, so there are two ways that can 
        # be done. The background image can be divided into a top and bottom 
        # half and one sample can be pasted in each half. Or the background 
        # image can be divided into a left and right half and one sample can be 
        # pasted in each half. The method to be chosen is done randomly.
        number1 = np.random.randint(100)
        if number1 < 50:   # Choose top and bottom half.
            topBotHalf, leftRightHalf = True, False
        else:
            topBotHalf, leftRightHalf = False, True

################################################################################

        if topBotHalf:
            
            # Pasting sample1 on the top half.

            if doHoriFlip or doVertFlip or do90degFlips or doRandomRot:

                # Augmenting the samples before affixing onto the background.
                
                # There are altogether 4 kinds of augmentation that this function can 
                # do, doRandomRot, doHoriFlip, doVertFlip, do90degFlips and no 
                # augmentation.
                # What kind of augmentation is to be done for this sample is chosen 
                # at random with a equal probability (20% for each type).
                # However, if the type of augmentation chosen doen not have it's 
                # corresponding flag True, then no augmentation is done.
        
                number = np.random.randint(100)
                
                # Horizontal flip sample1.
                
                if number < 20 and doHoriFlip:
                    newSample, newMask = horiFlipSampleAndMask(sample1, mask1)
                    bboxH1, bboxW1, _ = newSample.shape

################################################################################
    
                # Vertical flip sample1.
        
                elif number >= 20 and number < 40 and doVertFlip:
                    newSample, newMask = vertFlipSampleAndMask(sample1, mask1)
                    bboxH1, bboxW1, _ = newSample.shape

################################################################################
    
                # 90 deg flip sample1.
        
                elif number >= 40 and number < 60 and do90degFlips:
                    # Now the selection of whether the flip should be by 90, 180 or 270
                    # deg, is done randomly (with equal probablity).                
                    newSample, newMask, bboxW1, bboxH1 = random90degFlipSampleAndMask(sample1, mask1)
                
################################################################################
    
                # Rotation by random angles sample1.
        
                elif number >= 60 and number < 80 and doRandomRot:
                    # During rotation by arbitrary angles, the sample first needs to be
                    # pasted on a bigger blank array, otherwise it will get cropped 
                    # due to rotation.
                    newSample, newMask, bboxW1, bboxH1 = randomRotationSampleAndMask(sample1, mask1)
                
################################################################################
                
                # No augmentation sample1.
                
                else:
                    newSample, newMask = sample1, mask1
                    bboxH1, bboxW1, _ = newSample.shape
            
                # x, y of top left corner of the region where sample will be pasted.
                newSampleH, newSampleW, _ = newSample.shape
                tlY = np.random.randint(max(newBgH * 0.5 - newSampleH, 1))
                tlX = np.random.randint(newBgW - newSampleW)
                
                # Fixing the sample onto the background.
                image, posX1, posY1, _, _ = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[className1]
                    sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                    segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
                        
################################################################################
        
            # If both the clipSample1 and the other augmentation flags are False, 
            # then no augmentation is performed.

            else:
                newSample, newMask = sample1, mask1
            
                # x, y of top left corner of the region where sample will be pasted.
                newSampleH, newSampleW, _ = newSample.shape
                tlY = np.random.randint(max(newBgH * 0.5 - newSampleH, 1))
                tlX = np.random.randint(newBgW - newSampleW) 
                # Fixing the sample onto the background.
                image, posX1, posY1, bboxW1, bboxH1 = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[className1]
                    sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                    segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
                
################################################################################

            # The new background for sample2 will be the image formed earlier where
            # the sample1 was affixed onto the background.
            newBg = image
            newBgH, newBgW, _ = newBg.shape
            
            if createSegmentLabelImg:   newBgSegImg = segImg

################################################################################
            
            # Pasting sample2 on the bottom half.
            
            if doHoriFlip or doVertFlip or do90degFlips or doRandomRot:

                # Augmenting the samples before affixing onto the background.
                
                # There are altogether 4 kinds of augmentation that this function can 
                # do, doRandomRot, doHoriFlip, doVertFlip, do90degFlips and no 
                # augmentation.
                # What kind of augmentation is to be done for this sample is chosen 
                # at random with a equal probability (20% for each type).
                # However, if the type of augmentation chosen doen not have it's 
                # corresponding flag True, then no augmentation is done.
        
                number = np.random.randint(100)
                
                # Horizontal flip sample2.
                
                if number < 20 and doHoriFlip:
                    newSample, newMask = horiFlipSampleAndMask(sample2, mask2)
                    bboxH2, bboxW2, _ = newSample.shape

################################################################################
    
                # Vertical flip sample2.
        
                elif number >= 20 and number < 40 and doVertFlip:
                    newSample, newMask = vertFlipSampleAndMask(sample2, mask2)
                    bboxH2, bboxW2, _ = newSample.shape

################################################################################
    
                # 90 deg flip sample2.
        
                elif number >= 40 and number < 60 and do90degFlips:
                    # Now the selection of whether the flip should be by 90, 180 or 270
                    # deg, is done randomly (with equal probablity).                
                    newSample, newMask, bboxW2, bboxH2 = random90degFlipSampleAndMask(sample2, mask2)
                
################################################################################
    
                # Rotation by random angles sample2.
        
                elif number >= 60 and number < 80 and doRandomRot:
                    # During rotation by arbitrary angles, the sample first needs to be
                    # pasted on a bigger blank array, otherwise it will get cropped 
                    # due to rotation.
                    newSample, newMask, bboxW2, bboxH2 = randomRotationSampleAndMask(sample2, mask2)
                
################################################################################
                
                # No augmentation sample2.
                
                else:
                    newSample, newMask = sample2, mask2
                    bboxH2, bboxW2, _ = newSample.shape
            
                # x, y of top left corner of the region where sample will be pasted.
                newSampleH, newSampleW, _ = newSample.shape
                tlY = np.random.randint(max(newBgH * 0.5 - newSampleH, 1)) + int(newBgH * 0.5)
                tlX = np.random.randint(newBgW - newSampleW)
                
                # Fixing the sample onto the background.
                image, posX2, posY2, _, _ = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[className2]
                    sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                    segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
                        
################################################################################
        
            # If both the clipSample1 and the other augmentation flags are False, 
            # then no augmentation is performed.

            else:
                newSample, newMask = sample2, mask2
            
                # x, y of top left corner of the region where sample will be pasted.
                newSampleH, newSampleW, _ = newSample.shape
                tlY = np.random.randint(max(newBgH * 0.5 - newSampleH, 1)) + int(newBgH * 0.5)
                tlX = np.random.randint(newBgW - newSampleW)
                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[className2]
                    sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                    segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
                
################################################################################

        if leftRightHalf:
            
            # Pasting sample1 on the left half.

            if doHoriFlip or doVertFlip or do90degFlips or doRandomRot:

                # Augmenting the samples before affixing onto the background.
                
                # There are altogether 4 kinds of augmentation that this function can 
                # do, doRandomRot, doHoriFlip, doVertFlip, do90degFlips and no 
                # augmentation.
                # What kind of augmentation is to be done for this sample is chosen 
                # at random with a equal probability (20% for each type).
                # However, if the type of augmentation chosen doen not have it's 
                # corresponding flag True, then no augmentation is done.
        
                number = np.random.randint(100)
                
                # Horizontal flip sample1.
                
                if number < 20 and doHoriFlip:
                    newSample, newMask = horiFlipSampleAndMask(sample1, mask1)
                    bboxH1, bboxW1, _ = newSample.shape

################################################################################
    
                # Vertical flip sample1.
        
                elif number >= 20 and number < 40 and doVertFlip:
                    newSample, newMask = vertFlipSampleAndMask(sample1, mask1)
                    bboxH1, bboxW1, _ = newSample.shape

################################################################################
    
                # 90 deg flip sample1.
        
                elif number >= 40 and number < 60 and do90degFlips:
                    # Now the selection of whether the flip should be by 90, 180 or 270
                    # deg, is done randomly (with equal probablity).                
                    newSample, newMask, bboxW1, bboxH1 = random90degFlipSampleAndMask(sample1, mask1)
                
################################################################################
    
                # Rotation by random angles sample1.
        
                elif number >= 60 and number < 80 and doRandomRot:
                    # During rotation by arbitrary angles, the sample first needs to be
                    # pasted on a bigger blank array, otherwise it will get cropped 
                    # due to rotation.
                    newSample, newMask, bboxW1, bboxH1 = randomRotationSampleAndMask(sample1, mask1)
                
################################################################################
                
                # No augmentation sample1.
                
                else:
                    newSample, newMask = sample1, mask1
                    bboxH1, bboxW1, _ = newSample.shape
            
                # x, y of top left corner of the region where sample will be pasted.
                newSampleH, newSampleW, _ = newSample.shape
                tlY = np.random.randint(newBgH - newSampleH)
                tlX = np.random.randint(max(newBgW * 0.5 - newSampleW, 1))
                
                # Fixing the sample onto the background.
                image, posX1, posY1, _, _ = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[className1]
                    sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                    segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
                        
################################################################################
        
            # If both the clipSample1 and the other augmentation flags are False, 
            # then no augmentation is performed.

            else:
                newSample, newMask = sample1, mask1
            
                # x, y of top left corner of the region where sample will be pasted.
                newSampleH, newSampleW, _ = newSample.shape
                tlY = np.random.randint(newBgH - newSampleH)
                tlX = np.random.randint(max(newBgW * 0.5 - newSampleW, 1))

                # Fixing the sample onto the background.
                image, posX1, posY1, bboxW1, bboxH1 = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[className1]
                    sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                    segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
                
################################################################################

            # The new background for sample2 will be the image formed earlier where
            # the sample1 was affixed onto the background.
            newBg = image
            newBgH, newBgW, _ = newBg.shape
            
            if createSegmentLabelImg:   newBgSegImg = segImg

################################################################################
            
            # Pasting sample2 on the right half.
            
            if doHoriFlip or doVertFlip or do90degFlips or doRandomRot:

                # Augmenting the samples before affixing onto the background.
                
                # There are altogether 4 kinds of augmentation that this function can 
                # do, doRandomRot, doHoriFlip, doVertFlip, do90degFlips and no 
                # augmentation.
                # What kind of augmentation is to be done for this sample is chosen 
                # at random with a equal probability (20% for each type).
                # However, if the type of augmentation chosen doen not have it's 
                # corresponding flag True, then no augmentation is done.
        
                number = np.random.randint(100)
                
                # Horizontal flip sample2.
                
                if number < 20 and doHoriFlip:
                    newSample, newMask = horiFlipSampleAndMask(sample2, mask2)
                    bboxH2, bboxW2, _ = newSample.shape

################################################################################
    
                # Vertical flip sample2.
        
                elif number >= 20 and number < 40 and doVertFlip:
                    newSample, newMask = vertFlipSampleAndMask(sample2, mask2)
                    bboxH2, bboxW2, _ = newSample.shape

################################################################################
    
                # 90 deg flip sample2.
        
                elif number >= 40 and number < 60 and do90degFlips:
                    # Now the selection of whether the flip should be by 90, 180 or 270
                    # deg, is done randomly (with equal probablity).                
                    newSample, newMask, bboxW2, bboxH2 = random90degFlipSampleAndMask(sample2, mask2)
                
################################################################################
    
                # Rotation by random angles sample2.
        
                elif number >= 60 and number < 80 and doRandomRot:
                    # During rotation by arbitrary angles, the sample first needs to be
                    # pasted on a bigger blank array, otherwise it will get cropped 
                    # due to rotation.
                    newSample, newMask, bboxW2, bboxH2 = randomRotationSampleAndMask(sample2, mask2)
                
################################################################################
                
                # No augmentation sample2.
                
                else:
                    newSample, newMask = sample2, mask2
                    bboxH2, bboxW2, _ = newSample.shape
            
                # x, y of top left corner of the region where sample will be pasted.
                newSampleH, newSampleW, _ = newSample.shape
                tlY = np.random.randint(newBgH - newSampleH)
                tlX = np.random.randint(max(newBgW * 0.5 - newSampleW, 1)) + int(newBgW * 0.5)
                
                # Fixing the sample onto the background.
                image, posX2, posY2, _, _ = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[className2]
                    sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                    segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
                        
################################################################################
        
            # If both the clipSample1 and the other augmentation flags are False, 
            # then no augmentation is performed.

            else:
                newSample, newMask = sample2, mask2
            
                # x, y of top left corner of the region where sample will be pasted.
                newSampleH, newSampleW, _ = newSample.shape
                tlY = np.random.randint(newBgH - newSampleH)
                tlX = np.random.randint(max(newBgW * 0.5 - newSampleW, 1)) + int(newBgW * 0.5)

                # Fixing the sample onto the background.
                image, posX2, posY2, bboxW2, bboxH2 = fixSampleToBg(newSample, newMask, newBg, tlX, tlY)

                # Create the segmented label image as well if createSegmentLabelImg is True:
                if createSegmentLabelImg:
                    sampleColor = classNameToColor[className2]
                    sampleSegImg = cv2.bitwise_and(np.array(sampleColor), newMask)
                    segImg, _, _, _, _ = fixSampleToBg(sampleSegImg, newMask, newBgSegImg, tlX, tlY)
                
#################################################################################
#
#        cv2.imshow('sample', sample)
#        cv2.imshow('newSample', newSample)
#        cv2.imshow('mask', mask)
#        cv2.imshow('newMask', newMask)
#        cv2.imshow('image', image)
#        cv2.waitKey(0)
#                     
################################################################################

        # Saving the image.
        idx = nAlreadyExistingFiles + i    # This is the image index.
    
        imgSaveName = className1 + '_' + \
                      className2 + '_' + \
                      saveNameSuffix + '_' + str(idx) + savedImgFileExtn
                      
        cv2.imwrite(os.path.join(imgSaveLoc, imgSaveName), image)

        # Saving the segmented image label as well if createSegmentLabelImg is True.
        if createSegmentLabelImg:
            segImgSaveName = className1 + '_' + \
                          className2 + '_seg' + '_' + \
                          saveNameSuffix + '_' + str(idx) + savedImgFileExtn
            cv2.imwrite(os.path.join(segmentSaveLoc, segImgSaveName), segImg)
        
        # Creating the label json file.
        labelSaveName = className1 + '_' + \
                        className2 + '_' + \
                        saveNameSuffix + '_' + str(idx) + '.json'
        
        if className1 in classNameToIdx:
            classIdx1 = classNameToIdx[className1]
        # If the partialOBJ className is not included in the classNameToIdx 
        # dictionary, then it is assigned a new classIdx value.
        elif className1 == 'partialOBJ':
            classIdx1 = nClasses
        
        if className2 in classNameToIdx:
            classIdx2 = classNameToIdx[className2]
        # If the partialOBJ className is not included in the classNameToIdx 
        # dictionary, then it is assigned a new classIdx value.
        elif className2 == 'partialOBJ':
            classIdx2 = nClasses
        
        infoDict = {}
        
        with open(os.path.join(labelSaveLoc, labelSaveName), 'w') as infoFile:
            
################################################################################

            posX, posY, bboxW, bboxH = posX1, posY1, bboxW1, bboxH1
            className, classIdx = className1, classIdx1
            sampleLoc, sampleName = sampleLoc1, sampleName1

################################################################################

            # Make sure the coordinates are inside the boundaries of the image.
            if posX >= imgW:      posX = imgW - 1
            if posX < 0:            posX = 0
            if posY >= imgH:      posY = imgH - 1
            if posY < 0:            posY = 0
            tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
            brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
            if tlX < 0:            tlX = 0
            if tlY < 0:            tlY = 0
            if brX >= imgW:      brX = imgW - 1
            if brY >= imgH:      brY = imgH - 1
            bboxW, bboxH = int(brX - tlX), int(brY - tlY)   # Update box size.
            
################################################################################

            infoDict[0] = {
                            'className': className, 'classIdx': classIdx, \
                            'posX': int(posX), 'posY': int(posY), \
                            'bboxW': bboxW, 'bboxH': bboxH, \
                            'tlX': int(tlX), 'tlY': int(tlY), \
                            'samplePath': os.path.join(sampleLoc, sampleName), \
                            'bgPath': os.path.join(bgLoc, bgName) \
                         }
            
################################################################################

            posX, posY, bboxW, bboxH = posX2, posY2, bboxW2, bboxH2
            className, classIdx = className2, classIdx2
            sampleLoc, sampleName = sampleLoc2, sampleName2

################################################################################

            # Make sure the coordinates are inside the boundaries of the image.
            if posX >= imgW:      posX = imgW - 1
            if posX < 0:            posX = 0
            if posY >= imgH:      posY = imgH - 1
            if posY < 0:            posY = 0
            tlX, tlY = posX-bboxW*0.5, posY-bboxH*0.5   # Top left corner.
            brX, brY = posX+bboxW*0.5, posY+bboxH*0.5   # Bottom right corner.
            if tlX < 0:            tlX = 0
            if tlY < 0:            tlY = 0
            if brX >= imgW:      brX = imgW - 1
            if brY >= imgH:      brY = imgH - 1
            bboxW, bboxH = int(brX - tlX), int(brY - tlY)   # Update box size.
            
################################################################################

            infoDict[1] = {
                            'className': className, 'classIdx': classIdx, \
                            'posX': int(posX), 'posY': int(posY), \
                            'bboxW': bboxW, 'bboxH': bboxH, \
                            'tlX': int(tlX), 'tlY': int(tlY), \
                            'samplePath': os.path.join(sampleLoc, sampleName), \
                            'bgPath': os.path.join(bgLoc, bgName) \
                         }
                            
            json.dump(infoDict, infoFile, indent=4, separators=(',', ': '))

################################################################################

        for k, v in infoDict.items():
            cv2.circle(image, (v['posX'], v['posY']), 2, (0,255,0), 2)
            if v['className'] != '_':
                cv2.rectangle(image, (v['tlX'], v['tlY']), (v['tlX']+v['bboxW'], \
                                       v['tlY']+v['bboxH']), (0,255,0), 2)
        #cv2.imshow('image', image)
        # Show the segment label as well if the createSegmentLabelImg is True.
        if createSegmentLabelImg:   cv2.imshow('segment label', segImg)
        cv2.waitKey(0)

################################################################################
################################################################################

def markPoints(event, x, y, flags, params):
    '''
    This is a function that is called on mouse callback.
    '''
    global cBix, cBiy
    if event == cv2.EVENT_LBUTTONDOWN:
        cBix, cBiy = x, y

################################################################################
################################################################################

def selectPts(filePath=None):
    '''
    This function opens the image and lets user select the points in it.
    These points are returned as a list.
    If the image is bigger than 800 x 600, it is displayed as 800 x 600. But
    the points are mapped and stored as per the original dimension of the image.
    The points are clicked by mouse on the image itself and they are stored in
    the listOfPts.
    '''
    
    global cBix, cBiy
    
    img = cv2.imread(filePath)
    h, w = img.shape[0], img.shape[1]
    
    w1, h1, wRatio, hRatio, resized = w, h, 1, 1, False
#    print('Image size: {}x{}'.format(w, h))
    
    if w > 800:
        w1, resized = 800, True
        wRatio = w / w1
    if h > 600:
        h1, resized = 600, True
        hRatio = h / h1

    if resized:     img = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', markPoints)  # Function to detect mouseclick
    key = ord('`')

################################################################################
    
    listOfPts = []      # List to collect the selected points.
    
    while key & 0xFF != 27:         # Press esc to break.

        imgTemp = np.array(img)      # Temporary image.

        # Displaying all the points in listOfPts on the image.
        for i in range(len(listOfPts)):
            cv2.circle(imgTemp, tuple(listOfPts[i]), 3, (0, 255, 0), -1)
            
        # After clicking on the image, press any key (other than esc) to display
        # the point on the image.
        
        if cBix > 0 and cBiy > 0:
            print('{}'.format(' '*80))   # Erase the last line.
            print('New point: ({}, {}). Press \'s\' to save.'.format(cBix, cBiy))
        
            # Since this point is not saved yet, so it is displayed on the 
            # temporary image.
            cv2.circle(imgTemp, (cBix, cBiy), 3, (0, 0, 255), -1)
            
        cv2.imshow('Image', imgTemp)
        key = cv2.waitKey(125)
        
        # If 's' is pressed then the point is saved to the listOfPts.
        if key == ord('s'):
            listOfPts.append([cBix, cBiy])
            cv2.circle(imgTemp, (cBix, cBiy), 3, (0, 255, 0), -1)
            cBix, cBiy = -1, -1
            print('\nPoint Saved.')
            
        # Delete point by pressing 'd'.
        elif key == ord('d'):   cBix, cBiy = -1, -1
        
################################################################################

    # Map the selected points back to the size of original image using the 
    # wRatio and hRatio (if they we resized earlier).
    if resized:   listOfPts = [[int(p[0] * wRatio), int(p[1] * hRatio)] \
                                                        for p in listOfPts]
    return listOfPts

################################################################################
################################################################################
        
def timeStamp():
    '''
    Returns the current time stamp including the date and time with as a string 
    of the following format as shown.
    '''
    return datetime.datetime.now().strftime('_%m_%d_%Y_%H_%M_%S')

################################################################################
################################################################################

prettyTime = lambda x: str(datetime.timedelta(seconds=x))

################################################################################
################################################################################

def nCx(n, x):
    '''
    Calculates combination: i.e. n!/(m! * (n-m)!).
    '''
    if x > n or x < 0 or n < 0 or type(n) != int or type(x) != int:
        print('\nERROR: n should be > x and n and x should be non-negative ' \
              'integers. Aborting.\n')
        sys.exit()
        
    return np.math.factorial(n) / (np.math.factorial(m) * np.math.factorial(n-m))

################################################################################
################################################################################

def nPx(n, x):
    '''
    Calculates permutation: i.e. n!/(n-m)!.
    '''
    if x > n or x < 0 or n < 0 or type(n) != int or type(x) != int:
        print('\nERROR: n should be > x and n and x should be non-negative ' \
              'integers. Aborting.\n')
        sys.exit()
        
    return np.math.factorial(n) / np.math.factorial(n-m)

################################################################################
################################################################################

def findOneContourIntPt(contourPtArr=None):
    '''
    This function takes in a contour and returns a point that is garunteed to 
    lie inside the contour (mostly it returns a point near the edge of the contour).
    The input is the array of points which the contours are composed of in opencv.
    '''
    nPts = contourPtArr.shape[0]
    
    for i in range(nPts):      # Scanning all the points in the contour.
        # Contours are arrays with dimensions (nPts, 1, 2).
        # But the format of points in this contour array is [x, y], 
        # not [y, x] like other numpy arrays, because it is created by opencv.
        x, y = contourPtArr[i, 0, 0], contourPtArr[i, 0, 1]
        
        # Now x, y is a boundary point of this contour. So this means that 
        # some of the neighboring points of this x, y should lie inside the contour.
        # So all 9 points in the neighborhood of x, y (including x, y) is checked.
        for x1 in range(x-5, x+5):
            for y1 in range(y-5, y+5):
                test = cv2.pointPolygonTest(contourPtArr, (x1, y1), measureDist=False)
                #print('test', test)
                if int(test) > 0:   # Positive output means point is inside contour.
                    return [x1, y1]
                
                #If all these are outside the contour (which is theoritically 
                #impossible), only then the next point in the contour is checked.
                
        return [x, y]

################################################################################
################################################################################

def findLatestCkpt(checkpointDirPath=None, training=True):
    '''
    Finds out the latest checkpoint file in the checkpoint directory and
    deletes the incompletely created checkpoint.
    It returns the metaFilePath and ckptPath if found, else returns None.
    It also returns the epoch number of the latest completed epoch.
    The usual tensorflow funtion used to find the latest checkpoint does not
    take into account the fact that the learning rate or the batch size for the 
    training may have changed since the last checkpoint. So for that this 
    function and the json file created along with the checkpoint are used to 
    find the true latest checkpoint.
    It returns the checkpoint and json filepath.
    '''
    if checkpointDirPath is None:
        print('\nERROR: one or more input arguments missing ' \
               'in findLatestCkpt. Aborting.\n')
        sys.exit()
    
    # If this is a testing mode, and no checkpoint directory is there, then abort.
    if not os.path.exists(checkpointDirPath) and not training:
        print('\nERROR: checkpoint directory \'{}\' not found. ' \
               'in findLatestCkpt. Aborting.\n'.format(checkpointDirPath))
        sys.exit()

################################################################################

    # Create a folder to store the model checkpoints.
    if not os.path.exists(checkpointDirPath):  # If no previous model is saved.
        os.makedirs(checkpointDirPath)
        return None, None, 0
    
    # If there is previously saved model, then import the graph 
    # along with all the variables, operations etc. (.meta file).
    # Import the variable values (.data binary file with the use of
    # the .index file as well).
    # There is also a file called 'checkpoint' which keeps a record
    # of the latest checkpoint files saved.
    # There may be other files like '.thumbs' etc. which pops up often in 
    # windows.
    # So all files which do not have the proper extensions are deleted.
    listOfFiles = os.listdir(checkpointDirPath)

    for i in listOfFiles:
        extension = i.split('.')[-1]
        if extension != 'json' and extension != 'meta' and extension != 'index' and \
           extension != 'data-00000-of-00001' and extension != 'checkpoint':
               os.remove(os.path.join(checkpointDirPath, i))
    
################################################################################

    # Name of checkpoint to be loaded (in general the latest one).
    # Sometimes due to keyboard interrupts or due to error in saving
    # checkpoints, not all the .meta, .index or .data files are saved.
    # So before loading the checkpoint we need to check if all the 
    # required files are there or not, else the latest complete 
    # checkpoint files should be loaded. And all the incomplete latest 
    # but incomplete ones should be deleted.

    # List to hold the names of checkpoints which have all files.
    listOfValidCkptPaths = []
    
    listOfFiles = os.listdir(checkpointDirPath)
    
    # If there are files inside the checkpoint directory.
    while len(listOfFiles) > 0:
        # Continue till all the files are scanned.
        
        fileName = listOfFiles[-1]
        if fileName == 'checkpoint':    listOfFiles.remove(fileName)

        ckptName = '.'.join(fileName.split('.')[:-1])
        metaFileName = ckptName + '.meta'
        indexFileName = ckptName + '.index'
        dataFileName = ckptName + '.data-00000-of-00001'
        jsonFileName = ckptName + '.json'
        
        ckptPath = os.path.join(checkpointDirPath, ckptName)
        metaFilePath = os.path.join(checkpointDirPath, metaFileName)
        indexFilePath = os.path.join(checkpointDirPath, indexFileName)
        dataFilePath = os.path.join(checkpointDirPath, dataFileName)
        jsonFilePath = os.path.join(checkpointDirPath, jsonFileName)
        
        if metaFileName in listOfFiles and dataFileName in listOfFiles and \
           indexFileName in listOfFiles and jsonFileName in listOfFiles:
                # All the files exists, then this is a valid checkpoint. So 
                # adding that into the listOfValidCkptPaths.
                listOfValidCkptPaths.append(ckptPath)

                # Now removing these files from the listOfFiles as all processing 
                # related to them are done.
                listOfFiles.remove(metaFileName)
                listOfFiles.remove(indexFileName)
                listOfFiles.remove(dataFileName)
                listOfFiles.remove(jsonFileName)

        else:
            # If one or more of the .meta, .index or .data files are 
            # missing, then the remaining are deleted and also removed 
            # from the listOfFiles and then we loop back again.
            if os.path.exists(metaFilePath):
                os.remove(metaFilePath)
                listOfFiles.remove(metaFileName)
            if os.path.exists(indexFilePath):
                os.remove(indexFilePath)
                listOfFiles.remove(indexFileName)
            if os.path.exists(dataFilePath):
                os.remove(dataFilePath)
                listOfFiles.remove(dataFileName)
            if os.path.exists(jsonFilePath):
                os.remove(jsonFilePath)
                listOfFiles.remove(jsonFileName)

        #print(len(listOfFiles))

################################################################################

    # At this stage we do not have any incomplete checkpoints in the
    # checkpointDirPath. So now we find the latest checkpoint.
    latestCkptIdx, latestCkptPath = 0, None
    for ckptPath in listOfValidCkptPaths:
        currentCkptIdx = ckptPath.split('-')[-1]   # Extract checkpoint index.
        
        # If the current checkpoint index is '', (which can happen if the
        # checkpoints are simple names like 'cnn_model' and do not have 
        # index like cnn_model.ckpt-2 etc.) then break.
        if currentCkptIdx == '':    break
        
        currentCkptIdx = int(currentCkptIdx)
        
        if currentCkptIdx > latestCkptIdx:     # Compare.
            latestCkptIdx, latestCkptPath = currentCkptIdx, ckptPath
            
    # This will give the latest epoch that has completed successfully.
    # When the checkpoints are saved the epoch is added with +1 in the 
    # filename. So for extracting the epoch the -1 is done.
    latestEpoch = latestCkptIdx if latestCkptIdx > 0 else 0
    
#################################################################################

    ##latestCkptPath = tf.train.latest_checkpoint(checkpointDirPath)
    # We do not use the tf.train.latest_checkpoint(checkpointDirPath) 
    # function here as it is only dependent on the 'checkpoint' file 
    # inside checkpointDirPath. 
    # So this does not work properly if the latest checkpoint mentioned
    # inside this file is deleted because of incompleteness (missing 
    # some files).

    #ckptPath = os.path.join(checkpointDirPath, 'tiny_yolo.ckpt-0')

################################################################################

    if latestCkptPath != None:
        # This will happen when only the 'checkpoint' file remains.
        #print(latestCkptPath)
        latestJsonFilePath = latestCkptPath + '.json'
        return latestJsonFilePath, latestCkptPath, latestEpoch
    
    else:   
        # If no latest checkpoint is found or all are deleted 
        # because of incompleteness and only the 'checkpoint' file 
        # remains, then None is returned.
        return None, None, 0

################################################################################
################################################################################

def plotBboxDimensions(dataDir=None, showFig=True, saveFig=True, figSaveLoc='.'):
    '''
    This function goes through all the labels of the images in the given 
    dataset and notes down the bounding box dimensions of all the objects in the 
    image and plots them in a 2D graph so that the user can see the clusters 
    formed in the graph. That way the user can then define the number and 
    dimension of the anchor boxes to be used in the network. It also saves and 
    shows the final plot if the saveFig and showFig flags are True.
    '''
    labelDir = os.path.join(dataDir, 'labels')
    listOfLabel = os.listdir(labelDir)
    nLabel = len(listOfLabel)
    
    # Plotting the bboxH and bboxW of all the objects in all the label files of 
    # the given dataset.
    for i, fileName in enumerate(listOfLabel):
        jsonFileLoc = os.path.join(labelDir, fileName)      # Get the filename.

        with open(jsonFileLoc, 'r') as infoFile:    # Open file and read dict.
            infoDict = json.load(infoFile)

        # Plot the bboxH and bboxW of each of the objects in this dict.
        for k, v in infoDict.items():
            bboxW, bboxH = v['bboxW'], v['bboxH']
            plt.plot(bboxW, bboxH, '.b')
            
        print('[{}/{}] Scanned file: {}'.format(i+1, nLabel, fileName))
    
    plt.title('Height and Widths of bounding boxes')
    plt.xlabel('Width of bounding boxes')
    plt.ylabel('Height of bounding boxes')
    plt.grid()
    
    if saveFig:
        savedFigName = 'bounding_box_dimensions_{}.png'.format(dataDir)
        plt.savefig(os.path.join(figSaveLoc, savedFigName))
    
    if showFig:    plt.show()
    
################################################################################
################################################################################

def datasetMeanStd(dataDir=None):
    '''
    Takes in the location of the images as input.
    Calculates the mean and std of the images of a dataset that is needed 
    to normalize the images before training.
    Returns the mean and std in the form of float arrays 
    (e.g. mean = [0.52, 0.45, 0.583], std = [0.026, 0.03, 0.0434])
    '''
    imgDir = os.path.join(dataDir, 'images')
    listOfImg = os.listdir(imgDir)
    meanOfImg = np.zeros((inImgH, inImgW, 3), dtype=np.float32)
    meanOfImgSquare = np.zeros((inImgH, inImgW, 3), dtype=np.float32)
    nImg = len(listOfImg)
    
    for idx, i in enumerate(listOfImg):
        img = cv2.imread(os.path.join(imgDir, i))
        
        print('Adding the images to create mean and std {} of {}'.format(idx+1, \
                                                len(listOfImg)))
        meanOfImg += img / nImg
        meanOfImgSquare += img * (img / nImg)
    
    # Now taking mean of all pixels in the mean image created in the loop.
    # Now meanOfImg is 224 x 224 x 3.
    meanOfImg = np.mean(meanOfImg, axis=0)
    meanOfImgSquare = np.mean(meanOfImgSquare, axis=0)
    # Now meanOfImg is 224 x 3.
    meanOfImg = np.mean(meanOfImg, axis=0)
    meanOfImgSquare = np.mean(meanOfImgSquare, axis=0)
    # Now meanOfImg is 3.
    variance = meanOfImgSquare - meanOfImg * meanOfImg
    std = np.sqrt(variance)
    
    return meanOfImg, std

################################################################################
################################################################################
        
def getImgLabelClassification(curLoc, imgName):
    '''
    This function takes in the location of an current folder (train or test or
    valid, which contains the images folder) and the image file name, and then 
    accesses the json file for this image and returns the details from that file
    as a list.
    If the image has one sample, this list will have one subdictionary and if the 
    image has two samples, then this list will have two subdictionaries.
    It also creates the multi hot label for the image (which is a list of 1's and 0's).
    If there is 1 or more Eosinophils: label = [1,0,0,0,0,0,0,0,0,0]
    If there is 1 or more Eosinophils and 1 or more Neutrophil: label = [1,0,1,0,0,0,0,0,0,0]
    If there is 1 or more Neutrophils and 1 or more partialWBCs: label = [0,0,1,0,0,1,0,0,0,0]
    '''
    labelName = imgName.split('.')[0] + '.json'
    labelLoc = os.path.join(curLoc, 'labels', labelName)
    
    # Reading the json file.
    with open(labelLoc, 'r') as infoFile:
        infoDict = json.load(infoFile)
        
    nObj = len(infoDict)  # Number of objects in the image.

    multiHotLabel = np.zeros(nClasses, dtype=np.int32)
    
    # List to hold the label dictionaries extracted from the json file.
    labelDictList = []
    
    for i in range(nObj):
        labelDict = infoDict[str(i)]
        classIdx = labelDict['classIdx']
        labelDictList.append(labelDict)
        multiHotLabel[classIdx] = 1
    
    return labelDictList, multiHotLabel

################################################################################
################################################################################
        
def getImgLabelDetection(curLoc, imgName):
    '''
    This function takes in the location of an current folder (train or test or
    valid, which contains the images folder) and the image file name, and then 
    accesses the json file for this image and returns the details from that file
    as a list.
    If the image has one sample, this list will have one subdictionary and if the 
    image has two samples, then this list will have two subdictionaries.
    It also creates the label required for the detection. The label for each 
    image is 14 x 14 x nAnchors x (nClasses + 5) array.
    The function also returns the multihot label as well, similar to the case of
    classification (this comes in handy if we want to compare the classification
    accuracy during the classification phase and the detection phase).
    The function also returns a list of bounding boxes along with what object 
    category is there within that bounding box, as a 5 element list 
    (classIdx, x, y, w, h). This is needed to find the mAP during testing phase.
    This list will not be of a fixed lenght, as this will vary with the number of
    objects present in the image.
    '''
    labelName = imgName.split('.')[0] + '.json'
    labelLoc = os.path.join(curLoc, 'labels', labelName)
    
    # Reading the json file.
    with open(labelLoc, 'r') as infoFile:
        infoDict = json.load(infoFile)
        
    nObj = len(infoDict)  # Number of objects in the image.

    regionLabel = np.zeros((finalLayerH, finalLayerW, nAnchors, nClasses + 5), \
                                                               dtype=np.float32)
    
    # Creating the multihot label as well.
    multiHotLabel = np.zeros(nClasses, dtype=np.int32)
    
    # This is the list of class index and their corresponding bounding boxes.
    # This is blank if no object is present in the image.
    listOfClassIdxAndBbox = []
    
    # List to hold the label dictionaries extracted from the json file.
    labelDictList = []
    
    for i in range(nObj):
        labelDict = infoDict[str(i)]
        classIdx = labelDict['classIdx']

        posX, posY = labelDict['posX'], labelDict['posY']
        bboxW, bboxH = labelDict['bboxW'], labelDict['bboxH']
        tlX, tlY = labelDict['tlX'], labelDict['tlY']
        
        multiHotLabel[classIdx] = 1
        
        listOfClassIdxAndBbox.append([classIdx, tlX, tlY, bboxW, bboxH])
        
        # Finding the one hot class label.
        oneHotClassLabel = np.zeros(nClasses).tolist()
        oneHotClassLabel[classIdx] = 1
        
        # Finding the pixel of the final activation layer where the center of 
        # the bbox of the current object will lie. Also finding the offset from 
        # the location (as the location coordinate will be integer values).
        gridX, gridY = (posX / inImgW) * finalLayerW, (posY / inImgH) * finalLayerH
        gridXoffset, gridYoffset = gridX - int(gridX), gridY - int(gridY)
        gridX, gridY = int(gridX), int(gridY)
        
################################################################################

        # Finding the best anchor boxes which will have good iou score with the
        # bbox of the current object and also findin the scales by which these 
        # anchor boxes have to be scaled to match the ground truth bbox.
        
        # The ground truth box in this case is not the bbox obtained from the 
        # annotation in the image. Because this annotated bbox size is relative
        # to a inImgH x inImgW sized image. But to compare it with the anchor 
        # boxes (whose sizes are relative to the finalLayerH x finalLayerW sized
        # image), it has to be scaled down to the size of the finalLayerH x 
        # finalLayerW sized image.
        resizedBboxW = (bboxW / inImgW) * finalLayerW
        resizedBboxH = (bboxH / inImgH) * finalLayerH
        
        anchorFound = False    # Indicates if suitable anchor boxes are found.
        maxIou, maxIouIdx = 0, 0
        
        for adx, a in enumerate(anchorList):
            
            # Finding the iou with each of the anchor boxes in the list.
            # This iou is different from the one calculated by the findIOU function.
            # Here the iou is calculated assuming that the center of the ground 
            # truth bbox and the anchor box center coincides.
            minW, minH = min(a[0], resizedBboxW), min(a[1], resizedBboxH)
            iou = (minW * minH) / (a[0] * a[1] + resizedBboxW * resizedBboxH - minW * minH)
            
            if iou > iouThresh:
                anchorFound = True
                anchorWoffset, anchorHoffset = resizedBboxW / a[0], resizedBboxH / a[1]

                # Store this into the regionLabel in the suitable location.
                regionLabel[gridY, gridX, adx] = oneHotClassLabel + \
                                                   [gridXoffset, gridYoffset, \
                                                     np.log(anchorWoffset), \
                                                     np.log(anchorHoffset), 1.0]
                # The 1.0 is the confidence score that an object is present.
            
            # Also keep a record of the best anchor box found.
            if iou > maxIou:    maxIou, maxIouIdx = iou, adx
            
################################################################################

        # If it happens that none of the anchor boxes have a good enough iou 
        # score (all the scores are less than iouThresh), then just store the 
        # anchor box that has the max iou among all (even though it can be lower
        # than iouThresh). (This may happen when the ground truth bbox is of such
        # a dimension, that none of the anchors is having a proper match with a
        # good iou with it).
        if not anchorFound:
            anchorWoffset = resizedBboxW / anchorList[maxIouIdx][0]
            anchorHoffset = resizedBboxH / anchorList[maxIouIdx][1]

            # Store this into the regionLabel in the suitable location.
            regionLabel[gridY, gridX, maxIouIdx] = oneHotClassLabel + \
                                                     [gridXoffset, gridYoffset, \
                                                       np.log(anchorWoffset), \
                                                       np.log(anchorHoffset), 1.0]
            # The 1.0 is the confidence score that an object is present.
        
################################################################################
        
        labelDictList.append(labelDict)
    
    return labelDictList, regionLabel, multiHotLabel, listOfClassIdxAndBbox

################################################################################
################################################################################
        
def getImgLabelSegmentation(curLoc, imgName):
    '''
    This function takes in the location of an current folder (train or test or
    valid, which contains the images folder) and the image file name, and then 
    accesses the segment label to create the segment label batch.
    '''

    # Creating the label name from the image name.
    labelName = imgName.split('_')
    labelName.insert(-4, 'seg')
    labelName = '_'.join(labelName)
    labelLoc = os.path.join(curLoc, 'segments', labelName)
    
    # Reading the segment image file.
    segLabelImg = cv2.imread(labelLoc)
    
    # Creating a weight map. This array should not have a channel axis, otherwise
    # tf.losses.softmax_cross_entropy can handle it.
    segWeightMap = np.zeros((inImgH, inImgW))
    
    for c in range(nClasses):
        segColor = np.array(classIdxToColor[c])
        
        # Creating the mask for every class by using color filter.
        segMask = cv2.inRange(segLabelImg, segColor, segColor) # Values from 0-255.
        
        # Stack the segMasks along the depths to create the segmentLabel.
        # For 10 classes, the segmentLabel will be an image with 10 channels.
        segmentLabel = np.dstack((segmentLabel, segMask)) if c > 0 else segMask
        
        # Updating the weight map.
        segWeightMap += (segMask / 255.0) * classIdxToSegColorWeight[c]
        
    # Since this is basically creating a one hot vector for every pixel, so for 
    # the pixels that are just background, and not part of any object, there 
    # should be a channel as well. Hence a channel is added for the all black 
    # pixels as well. So for 10 classes the number of channels for this segmentLabel
    # array will be 11.
    segColor = np.array([0,0,0])
    segMask = cv2.inRange(segLabelImg, segColor, segColor)
    segmentLabel = np.dstack((segmentLabel, segMask))
        
    # Updating the weight map.
    segWeightMap += (segMask / 255.0) * classIdxToSegColorWeight[nClasses]

    # Reading the json file.
    jsonFileName = imgName.split('.')[0] + '.json'
    jsonFileLoc = os.path.join(curLoc, 'labels', jsonFileName)
    
    with open(jsonFileLoc, 'r') as infoFile:
        infoDict = json.load(infoFile)
        
    nObj = len(infoDict)  # Number of objects in the image.

    # List to hold the label dictionaries extracted from the json file.
    labelDictList = []
    
    for i in range(nObj):
        labelDict = infoDict[str(i)]
        classIdx = labelDict['classIdx']
        labelDictList.append(labelDict)
    
    return labelDictList, segmentLabel, segWeightMap

################################################################################
################################################################################

def calculateSegMapWeights(dataDir=None):
    '''
    This function calculates the weights of the different segment maps of the 
    different objects in the images of the given directory. This is useful during
    training the segmentation network where because of the disparity in the number
    of pixels of different colors in the segmented map, different weights have 
    to be assigned to them. Otherwise the network will become biased to some  
    particular segment map.
    It also counts the number of objects present in all the images in the given
    dataDir and calculates the average number of pixels used to represent every
    class object.    
    '''
    jsonLoc = os.path.join(dataDir, 'labels')
    listOfJson = os.listdir(jsonLoc)

    nClassObjs = np.zeros(nClasses)    # Blank array.

    labelLoc = os.path.join(dataDir, 'segments')
    listOfSegLabel = os.listdir(labelLoc)
    nSegLabels = len(listOfSegLabel)
    
    segLabelPixels = np.zeros(nClasses+1)    # Blank array.
    pppp = 0
    
    for i in range(nSegLabels):
        # Reading the dictionary from the json file.
        jsonFileName = listOfJson[i]
        with open(os.path.join(jsonLoc, jsonFileName), 'r') as infoFile:
            infoDict = json.load(infoFile)
        nObjInCurrentImg = np.zeros(nClasses)     # Blank array.
        
        # Counting the number of different objects in the image.
        for k, v in infoDict.items():
            classIdx = v['classIdx']
            nObjInCurrentImg[classIdx] += 1
            
        # Adding to the total object count.
        nClassObjs += nObjInCurrentImg
        
################################################################################

        segLabelName = listOfSegLabel[i]
        segLabelLoc = os.path.join(labelLoc, segLabelName)
        segLabel = cv2.imread(segLabelLoc)    # Reading the segment map image.
        currentSegLabelPixels = np.zeros(nClasses+1)      # Blank array.
        
        # Now count the number of pixels of each type of segment maps (indicated
        # by a separate color) in the current image.
        for c in range(nClasses):
            segColor = np.array(classIdxToColor[c])
            mask = cv2.inRange(segLabel, segColor, segColor)
            mask = mask / 255.0     # Converting from 0-255 to 0-1 range.
            nPixels = np.sum(mask)
            currentSegLabelPixels[c] += nPixels    # Add to pixel count for current class.
            
        # Now count the number of background pixels which are black.
        segColor = np.array([0,0,0])
        mask = cv2.inRange(segLabel, segColor, segColor)
        mask = mask / 255.0
        nPixels = np.sum(mask)
        currentSegLabelPixels[nClasses] += nPixels
        
        # Adding to the total pixel count.
        segLabelPixels += currentSegLabelPixels
        
        print('[{}/{}] Total pixels in {}: {}'.format(i+1, nSegLabels, segLabelName, \
                                            np.sum(currentSegLabelPixels)))

################################################################################

    #print(segLabelPixels)
    nTotalPixels = np.sum(segLabelPixels)
    segWeights = nTotalPixels / segLabelPixels      # Calculating the weights.
    
    # Calculating the average number of pixels used to represent every class object.
    # The background class pixels are however ignored.
    avgPixelsPerClassObj = segLabelPixels[: nClasses] / nClassObjs
    
    print('Segment weights:\n{}\n'.format(segWeights))
    print('Average number of pixels per object:\n{}\n'.format(avgPixelsPerClassObj))
    #print(nClassObjs, np.sum(nClassObjs))

################################################################################

    #for i in range(nSegLabels):
        #segLabelName = listOfSegLabel[i]
        #segLabelLoc = os.path.join(labelLoc, segLabelName)
        #segLabel = cv2.imread(segLabelLoc)    # Reading the segment map image.
        #segLabel1 = copy.deepcopy(segLabel)
        #listOfColors = [v for k, v in classIdxToColor.items()] + [[0,0,0]]
        
        #print(segLabelName)
        #exception = []
        
        #for x in range(224):
            #for y in range(224):
                #color = segLabel[y, x, :].tolist()
                #if color in listOfColors:
                    #pass
                #else:
                    #exception.append(color)
                    #segLabel[y,x,:] = [255,255,255]
                    ##segLabel[y,x,:] = [0,0,0]
                    
        #print(exception)
        #print(len(exception))
        #cv2.imshow('segLabel', segLabel)
        #cv2.imshow('original', segLabel1)
        #cv2.waitKey(0)

################################################################################
################################################################################

def createBatchForClassification(dataDir=None, listOfImg=None, batchSize=None, \
                                  shuffle=False, mean=0.0, std=1.0):
    '''
    This function takes in a list of images and their location directory of the
    dataset. It also takes in the batch size and returns an image batch, a label 
    batch and the updated listOfImg.
    The bounding box information is there in some json files having the same 
    name as the image files.
    The final batch of images and labels are sent as numpy arrays.
    The list of remaining images and list of selected images are also returned.
    '''
    if dataDir is None or listOfImg is None or batchSize is None:
        print('\nERROR: one or more input arguments missing ' \
               'in createBatchForClassification. Aborting.\n')
        sys.exit()

    # Shuffling in place if shuffle flag is True (training phase). 
    # This will be false for test and validation phases.
    if shuffle:     random.shuffle(listOfImg)
    
    listOfBatchImg = listOfImg[0 : batchSize]
    
    imgBatch, labelBatch = [], []
    for i in listOfBatchImg:
        img = cv2.imread(os.path.join(dataDir, 'images', i))
        
        labelDictList, multiHotLabel = getImgLabelClassification(dataDir, i)

        labelBatch.append(multiHotLabel)

#        print(multiHotLabel, os.path.join(dataDir, 'images', i))
#        cv2.imshow('Image', img)
#        cv2.waitKey(0)
        
#        # Normalizing the image by mean and std.
#        img = (img - mean) / std
        
        # Converting image to range -1 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 127.5, then it would result in np.float64.
        img = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        
        imgBatch.append(img)  
        
    # Removing these from the original listOfImg by first converting them to a 
    # set and then removing the set of element in the imgBatch and converting
    # back the resulting set to list.
    listOfImg = list(set(listOfImg) - set(listOfBatchImg))

    return np.array(imgBatch), np.array(labelBatch), listOfImg, listOfBatchImg

################################################################################
################################################################################

def createBatchForDetection(dataDir=None, listOfImg=None, batchSize=None, \
                                  shuffle=False, mean=0.0, std=1.0):
    '''
    This function takes in a list of images and their location directory of the
    dataset. It also takes in the batch size and returns an image batch, a label 
    batch and the updated listOfImg.
    The bounding box information is there in some json files having the same 
    name as the image files.
    The final batch of images and labels are sent as numpy arrays.
    The list of remaining images and list of selected images are also returned.
    A batch of multihot classification labels are also returned. This is for 
    the convenience of comparing the classification accuracy during detection
    and classification phases.
    A batch of class idx and their corresponding bounding boxes are also returned
    which are used to calculate the mAP during the testing phase.
    '''
    if dataDir is None or listOfImg is None or batchSize is None:
        print('\nERROR: one or more input arguments missing ' \
               'in createBatchForDetection. Aborting.\n')
        sys.exit()

    # Shuffling in place if shuffle flag is True (training phase). 
    # This will be false for test and validation phases.
    if shuffle:     random.shuffle(listOfImg)
    
    listOfBatchImg = listOfImg[0 : batchSize]
    
    imgBatch, labelBatch, labelBatchMultiHot, labelBatchClassIdxAndBbox = [], [], [], []
    for i in listOfBatchImg:
        img = cv2.imread(os.path.join(dataDir, 'images', i))
        
        labelDictList, regionLabel, multiHotLabel, listOfClassIdxAndBbox = \
                                            getImgLabelDetection(dataDir, i)

        labelBatch.append(regionLabel)
        
        labelBatchMultiHot.append(multiHotLabel)
        
        labelBatchClassIdxAndBbox.append(listOfClassIdxAndBbox)

#        print(multiHotLabel, os.path.join(dataDir, 'images', i))
#        cv2.imshow('Image', img)
#        cv2.waitKey(0)

#        # Normalizing the image by mean and std.
#        img = (img - mean) / std
        
        # Converting image to range -1 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 127.5, then it would result in np.float64.
        img = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        
        imgBatch.append(img)        
        
    # Removing these from the original listOfImg by first converting them to a 
    # set and then removing the set of element in the imgBatch and converting
    # back the resulting set to list.
    listOfImg = list(set(listOfImg) - set(listOfBatchImg))

    return np.array(imgBatch), np.array(labelBatch), np.array(labelBatchMultiHot), \
                            labelBatchClassIdxAndBbox, listOfImg, listOfBatchImg

################################################################################
################################################################################

def createBatchForSegmentation(dataDir=None, listOfImg=None, batchSize=None, \
                                  shuffle=False, mean=0.0, std=1.0):
    '''
    This function takes in a list of images and their location directory of the
    dataset. It also takes in the batch size and returns an image batch, a label 
    batch and the updated listOfImg.
    The label for segmentation is a set of images which has as many channels as 
    the number of classes.
    '''
    if dataDir is None or listOfImg is None or batchSize is None:
        print('\nERROR: one or more input arguments missing ' \
               'in createBatchForSegmentation. Aborting.\n')
        sys.exit()

    # Shuffling in place if shuffle flag is True (training phase). 
    # This will be false for test and validation phases.
    if shuffle:     random.shuffle(listOfImg)
    
    listOfBatchImg = listOfImg[0 : batchSize]
    
    imgBatch, labelBatch, weightBatch = [], [], []
    for i in listOfBatchImg:
        img = cv2.imread(os.path.join(dataDir, 'images', i))
        
        labelDictList, segmentLabel, segWeightMap = getImgLabelSegmentation(dataDir, i)

        # Converting image to range 0 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 255.0, then it would result in np.float64.
        segmentLabel = np.asarray(segmentLabel, dtype=np.float32) / 255.0

        labelBatch.append(segmentLabel)
        weightBatch.append(segWeightMap)
        
#        print(multiHotLabel, os.path.join(dataDir, 'images', i))
#        cv2.imshow('Image', img)
#        cv2.waitKey(0)

#        # Normalizing the image by mean and std.
#        img = (img - mean) / std
        
        # Converting image to range -1 to 1.
        # The image is explicitly converted to float32 to match the type 
        # specified in the placeholder. If img would have been directly divided
        # by 127.5, then it would result in np.float64.
        img = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        
        imgBatch.append(img)
    
    # Removing these from the original listOfImg by first converting them to a 
    # set and then removing the set of element in the imgBatch and converting
    # back the resulting set to list.
    listOfImg = list(set(listOfImg) - set(listOfBatchImg))
    
    return np.array(imgBatch), np.array(labelBatch), np.array(weightBatch), \
                            listOfImg, listOfBatchImg

################################################################################
################################################################################

def findIOU(boxA, boxB):
    '''
    Finds the IOU value between two rectangles. The rectangles are described in
    the format of [x, y, w, h], where x and y are the top left corner vertex.
    '''
    xA = max(boxA[0], boxB[0])      # Max top left x.
    yA = max(boxA[1], boxB[1])      # Max top left y.
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])      # Min bottom right x.
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])      # Min bottom right y.

    # Compute the area of intersection rectangle.
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # Compute the intersection over union by taking the intersection area and 
    # dividing it by the sum of areas of the rectangles - intersection area.
    union = boxAArea + boxBArea - intersection
    iou = intersection / float(union + 0.000001)

    # return the intersection over union value, intersection value and union value.
    return iou, intersection, union

################################################################################
################################################################################
    
def scanWholeImage(img):
    '''
    This may be an image that is bigger than what the network is trained with
    (which is inImgH x inImgW).
    Hence, it will be divided into several inImgH x inImgW segments to analyze 
    the presence of wbc in the image.
    '''
    imgH, imgW, _ = img.shape
    
    stepH, stepW = int(inImgH / 2), int(inImgH / 2)
    
    # Dividing the images into a grid of inImgH x inImgW cells, and then 
    # converting all those cells into a batch of images.
    # The top left corner coordinates of these grid cells are also stored in lists.
    imgBatch, locList = [], []
    for r in range(0, imgH, stepH):
        for c in range(0, imgW, stepW):
            cell = img[r : r + inImgH, c : c + inImgW]
            if cell.shape[0] == inImgH and cell.shape[1] == inImgW:
                imgBatch.append(cell)
                locList.append([c, r])
            
    # Now if the original image size is not a multiple of inImgH x inImgW, then
    # the remaining portion left out on the right and the bottom margins are 
    # included separately into the imgBatch.

    # Including bottom row if height is not a multiple.
    if imgH % inImgH > 0:
        for c in range(0, imgW, stepW):
            cell = img[imgH - inImgH : imgH, c : c + inImgW]
            if cell.shape[0] == inImgH and cell.shape[1] == inImgW:
                imgBatch.append(cell)
                locList.append([c, imgH - inImgH])
            
    # Including right column if width is not a multiple.
    if imgW % inImgW > 0:
        for r in range(0, imgH, stepH):
            cell = img[r : r + inImgH, imgW - inImgW : imgW]
            if cell.shape[0] == inImgH and cell.shape[1] == inImgW:
                imgBatch.append(cell)
                locList.append([imgW - inImgW, r])

    # Including bottom right corner if both height and width are not multiples.
    if imgH % inImgH > 0 and imgW % inImgW > 0:
        cell = img[imgH - inImgH : imgH, imgW - inImgW : imgW]
        if cell.shape[0] == inImgH and cell.shape[1] == inImgW:
            imgBatch.append(cell)
            locList.append([imgW - inImgW, imgH - inImgH])
            
#################################################################################
#    
#    # Displaying the corners of the grid cells.        
#    for i in range(len(locList)):
#        cv2.circle(img, (locList[i][0], locList[i][1]), 2, (0,255,0), 2)
#        
#    cv2.imshow('Img', img)
#    cv2.waitKey(0)
#
#    for i in imgBatch:
#        print(i.shape)
#
################################################################################

    return np.array(imgBatch), locList

################################################################################
################################################################################
    
def filterAndAddRect(rectList=None, rectangle=None):
    '''
    This function takes in a list of rectangles and also a new rectangle.
    It then checks if this new rectangle has a high IOU with any other rectangle
    or not. If so, then it averages the two and then stores that in the list
    replacing the old one. But if this new rectangle does not have much overlap
    with any other, then it just stores the rectangle as it is. Any new or 
    updated rectangle is always appended to the end of the rectList.
    '''
    if rectList is None or rectangle is None:
        print('\nERROR: one or more input arguments missing ' \
               'in filterAndAddRect. Aborting.\n')
        sys.exit()
    
    if len(rectList) == 0:    # List is empty.
        rectList.append(rectangle)
        return rectList
    
    # Check for the IOU values.
    overlapFound = False
    for rdx, r in enumerate(rectList):
        iou, intersection, union = findIOU(r, rectangle)
        if iou > 0.5:
            overlapFound = True
            # Take average of the two rectangles.
            x = int((r[0] + rectangle[0]) * 0.5)
            y = int((r[1] + rectangle[1]) * 0.5)
            w = int((r[2] + rectangle[2]) * 0.5)
            h = int((r[3] + rectangle[3]) * 0.5)
            rectList.pop(rdx)     # Remove the old rectangle.
            rectList.append([x,y,w,h])    # Append the new one.
            # The new or updated rectangle is always appended to the end of the 
            # list.
            
            # Since only one rectangle is added at a time, and since as soon as
            # an overlap is found, the rectangle in the original list os replaced,
            # hence no rectangle already stored in the rectList will have any 
            # overlap with each other. And hence the new rectangle can also have
            # an overlap with only one of these rectangles in the rectList.
            
    if not overlapFound:
        rectList.append(rectangle)
        
    return rectList

################################################################################
################################################################################

def localizeWeakly(gapLayer, inferPredLabelList, img=None):
    '''
    This function weakly localizes the objects based on the output of the conv
    layer just before the global average pooling (gap) layer. It also takes 
    help of the inferPredLabels. Only the output corresponding to a SINGLE image
    can be processed by this function. Not a batch of images.
    But the number of elements in the one or multi hot vectors in the final 
    labels, should be the same as the number of channels
    in this conv layer (which is given as input argument to this function).
    The inferPredLabelList is also the predicted label for one image in the batch
    not the overall inferPredLabel for the entire batch.
    '''

#    layer = inferLayerOut['conv19']
    h, w, nChan = gapLayer.shape
    if type(inferPredLabelList) != list:
        inferPredLabelList = inferPredLabelList.tolist()
    
    if len(inferPredLabelList) != nChan:
        print('\nERROR: the number of elements in the one/multi hot label is ' \
               'not the same as the number of channels in the conv layer input ' \
               'input to this function localizeWeakly. Aborting.\n')
        return
            
    # Stacking the channels of this layer together for displaying.
    # Also creating a List that will hold information about the labels, centers 
    # and bboxs.
    classAndLocList = []

    for c in range(nChan):
        channel = gapLayer[:,:,c]
        resizedChan = cv2.resize(channel, (inImgW, inImgW), \
                                     interpolation=cv2.INTER_LINEAR)
        minVal, maxVal = np.amin(resizedChan), np.amax(resizedChan)

        # Normalizing the output and scaling to 0 to 255 range.
        normalizedChan = (resizedChan - minVal) / (maxVal - minVal \
                                                      + 0.000001) * 255
        normalizedChan = np.asarray(normalizedChan, dtype=np.uint8)
        
        # Stacking the normalized channels.
        layerImg = normalizedChan if c == 0 else \
                                np.hstack((layerImg, normalizedChan))
                
################################################################################

        # WEAK LOCALIZATION
        
        # If there is a 1 in the predicted label, the corresponding 
        # channel of the gap layer is used to weakly localize the wbc cell.
        if inferPredLabelList[c]:
            # The normalized image is subjected to otsu's thresholding.
            _, binaryImg = cv2.threshold(normalizedChan, 0, 255, \
                                          cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    
            # Contours are then found out from the thresholded binary image,
            # after appending borders to the image.
            binaryImg = cv2.copyMakeBorder(binaryImg, 5,5,5,5, \
                                            cv2.BORDER_CONSTANT, value=0)
            returnedTuple = cv2.findContours(binaryImg, mode=cv2.RETR_TREE, \
                                               method=cv2.CHAIN_APPROX_SIMPLE)
            contours = returnedTuple[-2]
            
            # Locate the center of these contours with a mark.
            rectList = []
            for cdx, cont in enumerate(contours):
                cont = cont - 5    # Offsetting the border thickness.

                # Finding the bounding rectangle.
                x, y, w, h = cv2.boundingRect(cont)
                cx, cy = int(x+w/2), int(y+h/2)    # Center of the contour.
                
                # Store these bounding rectangles in a list as well, 
                # after checking that there is no duplicate or similar
                # rectangle. In other words, do a non-maximum supression
                # and then save the rectangles.
                rectList = filterAndAddRect(rectList, [cx-50,cy-50,100,100])
                
                # Storing the contours after offsetting border thickness.
                contours[cdx] = cont

            # Storing the information about the labels, centers and bboxs in the
            # classAndLocList.
            for rectangle in rectList:
                x, y, w, h = rectangle
                classAndLocList.append([c, int(x+w/2), int(y+h/2), x, y, w, h])

#################################################################################
#
#            if img is None:     continue
#        
#            # Draw the contours.
#            cv2.drawContours(img, contours, -1, (0,255,255), 2)
#        
#            # Now draw the rectangles.
#            predName = classIdxToName[c]
#            
#            for rectangle in rectList:
#                x, y, w, h = rectangle
#                cx, cy = int(x+w/2), int(y+h/2)    # Center of the contour.
#                cv2.circle(img, (cx, cy), 1, (255,255,0), 1)
#                cv2.rectangle(img, (cx-50, cy-50), (cx+50, cy+50), (255,255,0), 2)
#                cv2.putText(img, predName[0], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, \
#                                                0.6, (0,255,0), 2, cv2.LINE_AA)
#
#    if img is not None:
#        cv2.imshow('Weak localization', img)
#        cv2.imshow('Gap layer', layerImg)
#        cv2.waitKey(0)                        
#                                    
################################################################################

    return classAndLocList
        
################################################################################
################################################################################

def nonMaxSuppression(predResult):
    '''
    This function takes in the raw output of the network and then performs a 
    non-maximum suppression on the results to filter out the redundant and 
    unwanted bounding boxes.
    The filtered boxes are in the format of (top-left-X, top-left-Y, width, height).
    '''
    # nResults is the number of image results present in this data.
    nResults, _, _, _, _ = predResult.shape
    
    detectedBatchClassScores, detectedBatchClassIdxes, detectedBatchClassNames, \
                                            detectedBatchBboxes = [], [], [], []
    
    for b in range(nResults):
        classes, bboxes = [], []

        for i in range(finalLayerH):
            for j in range(finalLayerW):
                for a in range(nAnchors):
                    oneHotVec = predResult[b, i, j, a, : nClasses]
                    xOffset = predResult[b, i, j, a, -5]
                    yOffset = predResult[b, i, j, a, -4]
                    wOffset = predResult[b, i, j, a, -3]
                    hOffset = predResult[b, i, j, a, -2]
                    confOffset = predResult[b, i, j, a, -1]
                    
                    classProb = oneHotVec * confOffset
                    
                    # Now rescaling the w and h to the actual input image size.
                    w = (wOffset / finalLayerW) * inImgW
                    h = (hOffset / finalLayerH) * inImgH
                    
                    # Calculating the top left corner coordinates of the boxes.
                    x = ((j + xOffset) / finalLayerW) * inImgW
                    y = ((i + yOffset) / finalLayerH) * inImgH
                    tlX, tlY = x - w * 0.5, y - h * 0.5
                    
                    # Recording the bboxes and classes into the lists.
                    bboxes.append([int(tlX), int(tlY), int(w), int(h)])
                    classes.append(classProb)
                    
################################################################################
                    
        # Converting the classes list into array.
        classes = np.array(classes)
        
        # classes and bboxes arrays have one row for each of 
        # finalLayerH x finalLayerW x nAnchors (14 x 14 x 5 = 980) anchor boxes. 
        # And the columns represent probabilities of the nClasses (6) classes. 
        
        # We will transpose the classes array before non max suppression. So 
        # now there is one column for each anchor box and each row represents 
        # the probability of the classes.
        classes = np.transpose(classes)
        
        for c in range(nClasses):
            classProb = classes[c]
            
            # Making the class probability 0 if it is less than threshProbDetection.
            classProb = classProb * (classProb > threshProbDetection)

            # Sorting both the arrays in descending order (arrays can also be 
            # sorted in this manner like lists).
            # In the end after the redundant boxes are removed, these sorted 
            # lists has to be reverted back to their original (unsorted) form.
            # To do that a list of indexes are also maintained as a record.
            indexes = list(range(len(bboxes)))
            classProbSorted, bboxesSorted, indexes = zip(*sorted(zip(classProb, \
                                                          bboxes, indexes), \
                                                          key=lambda x: x[0], \
                                                          reverse=True))
            
            # The classProbSorted and bboxesSorted are returned as tuples.
            # Converting them to lists.
            classProbSorted, bboxesSorted, indexes = list(classProbSorted), \
                                                     list(bboxesSorted), list(indexes)
            
################################################################################
            
            # Now we are comparing all the boxes for the current class c for 
            # removing redundant ones.
            for bboxMaxIdx in range(len(bboxesSorted)):
                # Skipping the boxes if the corresponding class probability is 0.
                # Since the numbers are in floats, so we do not use == 0 here.
                # Instead we compare whether the number is very close to 0 or not.
                if classProbSorted[bboxMaxIdx] < 0.000001:      continue
                
                bboxMax = bboxesSorted[bboxMaxIdx]    # Box with max class probability.
                
                for bboxCurIdx in range(bboxMaxIdx+1, len(bboxesSorted)):
                    # Skipping the boxes if the corresponding class probability is 0.
                    # Since the numbers are in floats, so we do not use == 0 here.
                    # Instead we compare whether the number is very close to 0 or not.
                    if classProbSorted[bboxCurIdx] < 0.000001:      continue
                    
                    # Box other than the max class probability box.
                    bboxCur = bboxesSorted[bboxCurIdx]
                    
                    # If the iou between the boxes with max probability and the
                    # current box is greater than the iouTh, then that means 
                    # the current box is redundant. So set corresponding class
                    # probability to 0.
                    iou, _, _ = findIOU(bboxMax, bboxCur)
                    if iou > iouThresh:     classProbSorted[bboxCurIdx] = 0
            
            # Now that all the redundancy is removed, we restore the probability 
            # values to the original classes array after converting them into 
            # their original (unsorted) format.
            classProbUnsorted, indexes = zip(*sorted(zip(classProbSorted, \
                                                       indexes), key=lambda x: x[1]))

################################################################################

            # The classProbSorted and bboxesSorted are returned as tuples.
            # Converting them to lists.
            classProbUnsorted, indexes = list(classProbUnsorted), list(indexes)
            classes[c] = classProbUnsorted

################################################################################

        # Scanning the cols (each col has class probabilities of an anchor box).
        detectedIdxes, detectedScores, detectedObjs, detectedBboxes = [], [], [], []
        for a in range(len(bboxes)):
            maxScore, maxScoreIdx = np.max(classes[:, a]), np.argmax(classes[:, a])
            if maxScore > 0:
                detectedScores.append(maxScore)
                detectedIdxes.append(maxScoreIdx)
                detectedObjs.append(classIdxToName[maxScoreIdx])
                detectedBboxes.append(bboxes[a])

        # Recording the bboxes and classes into the list for the entire batch.
        detectedBatchClassScores.append(detectedScores)
        detectedBatchClassIdxes.append(detectedIdxes)
        detectedBatchClassNames.append(detectedObjs)
        detectedBatchBboxes.append(detectedBboxes)

################################################################################
    
    detectedBatchClassScores = np.array(detectedBatchClassScores)
    detectedBatchClassIdxes = np.array(detectedBatchClassIdxes)
    detectedBatchClassNames = np.array(detectedBatchClassNames)
    detectedBatchBboxes = np.array(detectedBatchBboxes)
        
    return detectedBatchClassScores, detectedBatchClassIdxes, \
                                detectedBatchClassNames, detectedBatchBboxes
    
################################################################################
################################################################################

def calculateMAP(allTestMultiHot=None, allTestClassIdxAndBbox=None, \
                  allDetectedClassIdxes=None, allDetectedClassScores=None, \
                  allDetectedBboxes=None):
    '''
    This function takes in lists of all the details of prediction over an
    entire dataset along with the details of the ground truth indexes and 
    bounding boxes and then calculates the mean average precision over this
    dataset. It also takes in the list of all multihot labels to know which of 
    the images has which kind of objects. This is important to know the average
    precision of the individual classes. All these average precisions are 
    combined together to calculate the mAP.
    '''
    if allTestMultiHot is None or allTestClassIdxAndBbox is None or \
       allDetectedClassIdxes is None or allDetectedClassScores is None or \
       allDetectedBboxes is None:
            print('\nERROR: one or more input arguments missing ' \
                   'in calculateMAP. Aborting.\n')
            sys.exit()

    nImgs = len(allTestClassIdxAndBbox)
    mAP = 0
    APlist = []
    
################################################################################
    
    # Calculating the average precision of each class and then adding them to 
    # find the mAP
    for c in range(nClasses):
        AP = 0
        nInstance = 0   # Number of instances of this class in the entire dataset.
        
        # Now scanning all the records of the images inside the input lists and
        # creating  a list of true and false positives and scores.
        fullTPlist, fullFPlist, fullScoreList = [], [], []
        
        for i in range(nImgs):
            multiHotLabel = allTestMultiHot[i]
            
            # Now checking if the multihot label has the object of class c or not.
            # If not then this image is skipped. This is determined by checking
            # if the c-th element of the multiHotLabel is 1 or not.
            if multiHotLabel[c] == 0:   continue
        
            trueClassIdxAndBbox = allTestClassIdxAndBbox[i]
            detectedClassIdxes = allDetectedClassIdxes[i]
            detectedClassScores = allDetectedClassScores[i]
            detectedBboxes = allDetectedBboxes[i]
            
################################################################################

            # Scanning all the predictions for this image i.
            
            # First count how many of the predicted boxes also predicts class c.
            # Also, store the indexes of these predicted boxes in a list.
            # Create a blank list equal to this count. These will store
            # the true positive status of the predicted boxes.
            indexes = [kdx for kdx, k in enumerate(detectedClassIdxes) if k == c]
            TPlist = np.zeros(len(indexes), dtype=int).tolist()
            scoreList = [detectedClassScores[kdx] for kdx, k in \
                                     enumerate(detectedClassIdxes) if k == c]

################################################################################

            # Now scan through all the records of this image.
            for j in range(len(trueClassIdxAndBbox)):
                classIdx, tlX, tlY, bboxW, bboxH = trueClassIdxAndBbox[j]
                
                # Check if the jth record has the object c or not.
                if classIdx != c:       continue

                nInstance += 1    # Counting number of instances of class c.

                bestIOU = iouThreshForMAPcalculation
                bestPdx = -1    # This will become the index of the box with best iou.
                
################################################################################

                # Now taking only the boxes which are recorded in the indexes list
                # (as only those boxes are detecting the object of class c).
                for pdx, p in enumerate(indexes):
                    predTlX, predTlY, predBboxW, predBboxH = detectedBboxes[p]
                    
                    # Find the iou now.
                    iou, _, _ = findIOU([tlX, tlY, bboxW, bboxH], \
                                         [predTlX, predTlY, predBboxW, predBboxH])
                    
                    # It may happen that there are multiple bounding boxes which
                    # overlap with the same object. In that case select the one
                    # which has the highest iou score as the best one.
                    if iou > bestIOU:   bestIOU, bestPdx = iou, pdx
                    
################################################################################
                    
                # Now make this box corresponding to the bestIOU as a true positive.
                # If however the bestPdx is still -1, then it implies that there
                # are no good boxes here. Hence skip the update then.
                if bestPdx > -1:    TPlist[bestPdx] = 1
                
            # Now make all the other remaining boxes as false positive.
            FPlist = [0 if m == 1 else 1 for m in TPlist]
            
################################################################################
             
            # Combining the true and false positive and the score lists into the
            # bigger list.
            fullTPlist += TPlist
            fullFPlist += FPlist
            fullScoreList += scoreList
        
################################################################################
        
        # Now sort the lists as per the score values.
        sortedScoreList, sortedTPlist, sortedFPlist = zip(*sorted(zip(fullScoreList, \
                                                        fullTPlist, fullFPlist), \
                                                        key=lambda x: x[0], reverse=True))
        
        # The sortedScoreList, sortedTPlist and sortedFPlist are returned as tuples.
        # Converting them to arrays.
        sortedScoreList, sortedTPlist, sortedFPlist = np.array(sortedScoreList), \
                                                      np.array(sortedTPlist), \
                                                      np.array(sortedFPlist)

        # Creating the accumulated true and false positive lists.
        accumulatedTP, accumulatedFP = np.cumsum(sortedTPlist), np.cumsum(sortedFPlist)
        precision = accumulatedTP / (accumulatedTP + accumulatedFP)
        recall = accumulatedTP / nInstance
        
        # Converting the precision and recall from arrays to list.
        precision, recall = precision.tolist(), recall.tolist()

################################################################################

        # Calculating the average precision of this class c.
        
#        plt.plot(recall, precision)
#        plt.show()

        # A lot of the recall values evaluated in this manner will be repeated.
        # So taking a set of the distinct recall values, (then sorting them, as 
        # the values may not be in sorted form while creating the set) and taking 
        # the corresponding precision values.
        recallSet = set(recall)
        recallSet = sorted(list(recallSet))
        precisionSet = [precision[recall.index(r)] for r in recallSet]
        
        # The precisionSet now has the precision values which are the vertices of the
        # sawtooth shaped precision recall curve.
        # Sorting the precisionSet in descending order to find the tallest vertices.
        precisionSet, recallSet = zip(*sorted(zip(precisionSet, recallSet), \
                                               key=lambda x: x[0], reverse=True))
    
        # The precisionSet and recallSet are returned as tuples. Converting them to lists.
        precisionSet, recallSet = list(precisionSet), list(recallSet)
    
        # Appending a 0 to the recallSet.
        recallSet = [0.0] + recallSet
    
################################################################################

        totalArea, previousStep = 0.0, 0
        for r in range(1, len(recallSet)):
            # Calculating the base of the rectangular section.
            base = recallSet[r] - recallSet[previousStep]
            
            if base > 0:
                # Calculating the height of the rectangular section.
                height = precisionSet[r-1]
                totalArea += height * base            
                previousStep = r    # Updating the previousStep.
    
        AP = totalArea * 100
        APlist.append(AP)
        mAP += (AP / nClasses)
    
################################################################################

    return mAP, APlist






