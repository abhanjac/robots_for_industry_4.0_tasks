# -*- coding: utf-8 -*-
'''
This script takes in the available images and their masks from a given folder 
and creates a dataset from those by affixing those onto different backgrounds, 
along with their corresponding labels. These images will be used for training, 
validation and testing the cnn.

In order to prevent the same samples to be used for creating training images
and validation images, the samples in these datasets are seggregated into a 
train, valid and test folder in the very beginning itself, before starting to 
create the new images.

Only the samples in the train folder will be used to synthesize the training 
images. And so are the cases for valid and test images.

If there are masks available for the samples, that kind of shows where exactly
the object is inside the sample image, then those have to be in a separate 
directory in the same folder as the sample images. They also need to be 
seggregated into a train, valid and a test folder just like their corresponding
sample image. If the sample image name is 'nuts_1.png' then its mask should have 
the name 'nuts_1_mask.png'. 

However, this code can work even if there are no masks. In that case the object 
image will be affixed as it is on the background image and the bounding box of 
the object will be assumed to be the same as the size of the object image.

A separate function is also used to make the masks tightly bound the object part 
in the sample image and in the mask image as well.
It removes the unwanted black border regions surrounding the white portion
of the masks, along with the removal of the corresponding region of the 
sample image. If the mask and the sample already tightly bounds the object then 
this function keep those unchanged.

@author: abhanjac
'''

from utils import *

# Create a train, validation and a test folder. Each of these folders should 
# have two subfolders: images and labels.
# These are created manually.

if __name__ == '__main__':
    pass
    
    curLoc = 'train'
    nImgs = 4000

    imgH, imgW = 480, 640

    imgSaveLoc = os.path.join(curLoc, 'images')
    labelSaveLoc = os.path.join(curLoc, 'labels')

    # Create the directories if not present.
    if not os.path.exists(imgSaveLoc):    os.makedirs(imgSaveLoc)
    if not os.path.exists(labelSaveLoc):    os.makedirs(labelSaveLoc)

################################################################################
################################################################################
    '''
    -------------------------------------------------------------------------
     background             = 4000  
     nuts                   = 4000  
     washers                = 4000  
     gears                  = 4000   
     empty_bin              = 4000
     crankArmW              = 4000
     crankShaft             = 4000
     Two same objects       = 2000 * 6 (6 objects, excluding background)
     Two different objects  = 2000 * 15 (each combination. Total 15 combinations)
    -------------------------------------------------------------------------
     TOTAL                  = 70000 
    -------------------------------------------------------------------------
    '''
################################################################################
################################################################################

    bgLocNormal = os.path.join('background', curLoc)
    sampleName0 = 'nuts'
    sampleLoc0 = os.path.join('color_images', sampleName0, curLoc)
    maskLoc0 = os.path.join('color_images', sampleName0 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc0, maskDir=maskLoc0, replaceOriginal=True) # Making tight masks.
    sampleName1 = 'washers'
    sampleLoc1 = os.path.join('color_images', sampleName1, curLoc)
    maskLoc1 = os.path.join('color_images', sampleName1 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc1, maskDir=maskLoc1, replaceOriginal=True) # Making tight masks.
    sampleName2 = 'gears'
    sampleLoc2 = os.path.join('color_images', sampleName2, curLoc)
    maskLoc2 = os.path.join('color_images', sampleName2 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc2, maskDir=maskLoc2, replaceOriginal=True) # Making tight masks.
    sampleName3 = 'emptyBin'
    sampleLoc3 = os.path.join('color_images', sampleName3, curLoc)
    maskLoc3 = os.path.join('color_images', sampleName3 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc3, maskDir=maskLoc3, replaceOriginal=True) # Making tight masks.
    sampleName4 = 'crankArmW'
    sampleLoc4 = os.path.join('color_images', sampleName4, curLoc)
    maskLoc4 = os.path.join('color_images', sampleName4 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc4, maskDir=maskLoc4, replaceOriginal=True) # Making tight masks.
    sampleName5 = 'crankShaft'
    sampleLoc5 = os.path.join('color_images', sampleName5, curLoc)
    maskLoc5 = os.path.join('color_images', sampleName5 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc5, maskDir=maskLoc5, replaceOriginal=True) # Making tight masks.

################################################################################
################################################################################

    print('Creating {} images: background, {}.'.format(nImgs, curLoc))

    blankBackground(bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, labelSaveLoc=labelSaveLoc, \
                    nImgs=nImgs, imgH=imgH, imgW=imgW, saveNameSuffix='' + curLoc)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    # For the objects present in bins like: nuts, washers, gears, emptyBin.

    nImgsOfEachSample = [nImgs]
    listOfSampleNames = [sampleName0, sampleName1, sampleName2, sampleName3]
    listOfSampleLocs = [sampleLoc0, sampleLoc1, sampleLoc2, sampleLoc3]
    listOfMaskLocs = [maskLoc0, maskLoc1, maskLoc2, maskLoc3]
    listOfNimgs = nImgsOfEachSample*len(listOfSampleNames)
    
    for i in range(len(listOfSampleLocs)):
        sampleNameI = listOfSampleNames[i]
        sampleLocI = listOfSampleLocs[i]
        maskLocI = listOfMaskLocs[i]
        nImgsI = listOfNimgs[i]

        print('Creating {} images: {}, {}.'.format(nImgsI, sampleNameI, curLoc))

        # Images having a ONE instance of an object.
        singleInstance(sampleLoc=sampleLocI, \
                       maskLoc=maskLocI, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                       labelSaveLoc=labelSaveLoc, nImgs=nImgsI, \
                       saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                       do90degFlips=False, doHoriFlip=True, doVertFlip=False, \
                       doRandomRot=False)
        cv2.destroyAllWindows()

################################################################################
################################################################################

    # For crankArmW.
    
    print('Creating {} images: {}, {}.'.format(nImgs, sampleName4, curLoc))

    # Images having a ONE instance of an object.
    singleInstance(sampleLoc=sampleLoc4, \
                   maskLoc=maskLoc4, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                   labelSaveLoc=labelSaveLoc, nImgs=nImgs, \
                   saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                   do90degFlips=True, doHoriFlip=True, doVertFlip=True, \
                   doRandomRot=False)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    # For crankShaft.
    
    print('Creating {} images: {}, {}.'.format(nImgs, sampleName5, curLoc))

    # Images having a ONE instance of an object.
    singleInstance(sampleLoc=sampleLoc5, \
                   maskLoc=maskLoc5, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                   labelSaveLoc=labelSaveLoc, nImgs=nImgs, \
                   saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                   do90degFlips=True, doHoriFlip=True, doVertFlip=True, \
                   doRandomRot=False)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    nImgs = 2000
    
    nImgsOfEachSample = [nImgs]
    listOfSampleNames = [sampleName0, sampleName1, sampleName2, sampleName3, sampleName4, sampleName5]
    listOfSampleLocs = [sampleLoc0, sampleLoc1, sampleLoc2, sampleLoc3, sampleLoc4, sampleLoc5]
    listOfMaskLocs = [maskLoc0, maskLoc1, maskLoc2, maskLoc3, maskLoc4, maskLoc5]
    listOfNimgs = nImgsOfEachSample*len(listOfSampleNames)
    
    for i in range(len(listOfSampleLocs)):
        sampleNameI = listOfSampleNames[i]
        sampleLocI = listOfSampleLocs[i]
        maskLocI = listOfMaskLocs[i]
        nImgsI = listOfNimgs[i]
        
        for j in range(i, len(listOfSampleLocs)):
            sampleNameJ = listOfSampleNames[j]
            sampleLocJ = listOfSampleLocs[j]
            maskLocJ = listOfMaskLocs[j]

            print('Creating {} images: {} + {}, {}.'.format(nImgsI, sampleNameI, sampleNameJ, curLoc))

            doubleInstance(sampleLoc1=sampleLocI, sampleLoc2=sampleLocJ, \
                           maskLoc1=maskLocI, maskLoc2=maskLocJ, bgLoc=bgLocNormal, \
                           imgSaveLoc=imgSaveLoc, labelSaveLoc=labelSaveLoc, \
                           nImgs=nImgsI, imgH=imgH, imgW=imgW, saveNameSuffix='' + curLoc, \
                           do90degFlips=False, doHoriFlip=True, doVertFlip=False, \
                           doRandomRot=False)

################################################################################
################################################################################

    # Check the clusters in the bounding box dimensions and then create the 
    # anchor boxes based on those dimensions.
    plotBboxDimensions(dataDir=curLoc, showFig=True, saveFig=True, figSaveLoc='.')
    
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

    curLoc = 'valid'
    nImgs = 400
    
    imgH, imgW = 480, 640

    imgSaveLoc = os.path.join(curLoc, 'images')
    labelSaveLoc = os.path.join(curLoc, 'labels')

    # Create the directories if not present.
    if not os.path.exists(imgSaveLoc):    os.makedirs(imgSaveLoc)
    if not os.path.exists(labelSaveLoc):    os.makedirs(labelSaveLoc)

#################################################################################
#################################################################################
    '''
    -------------------------------------------------------------------------
     background             = 400
     nuts                   = 400  
     washers                = 400  
     gears                  = 400   
     empty_bin              = 400
     crankArmW              = 400
     crankShaft             = 400
     Two same objects       = 200 * 6 (6 objects, excluding background)
     Two different objects  = 200 * 15 (each combination. Total 15 combinations)
    -------------------------------------------------------------------------
     TOTAL                  = 7000 
    -------------------------------------------------------------------------
    '''
################################################################################
################################################################################

    bgLocNormal = os.path.join('background', curLoc)
    sampleName0 = 'nuts'
    sampleLoc0 = os.path.join('color_images', sampleName0, curLoc)
    maskLoc0 = os.path.join('color_images', sampleName0 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc0, maskDir=maskLoc0, replaceOriginal=True) # Making tight masks.
    sampleName1 = 'washers'
    sampleLoc1 = os.path.join('color_images', sampleName1, curLoc)
    maskLoc1 = os.path.join('color_images', sampleName1 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc1, maskDir=maskLoc1, replaceOriginal=True) # Making tight masks.
    sampleName2 = 'gears'
    sampleLoc2 = os.path.join('color_images', sampleName2, curLoc)
    maskLoc2 = os.path.join('color_images', sampleName2 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc2, maskDir=maskLoc2, replaceOriginal=True) # Making tight masks.
    sampleName3 = 'emptyBin'
    sampleLoc3 = os.path.join('color_images', sampleName3, curLoc)
    maskLoc3 = os.path.join('color_images', sampleName3 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc3, maskDir=maskLoc3, replaceOriginal=True) # Making tight masks.
    sampleName4 = 'crankArmW'
    sampleLoc4 = os.path.join('color_images', sampleName4, curLoc)
    maskLoc4 = os.path.join('color_images', sampleName4 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc4, maskDir=maskLoc4, replaceOriginal=True) # Making tight masks.
    sampleName5 = 'crankShaft'
    sampleLoc5 = os.path.join('color_images', sampleName5, curLoc)
    maskLoc5 = os.path.join('color_images', sampleName5 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc5, maskDir=maskLoc5, replaceOriginal=True) # Making tight masks.

################################################################################
################################################################################

    print('Creating {} images: blank background, {}.'.format(nImgs, curLoc))

    blankBackground(bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, labelSaveLoc=labelSaveLoc, \
                    nImgs=nImgs, imgH=imgH, imgW=imgW, saveNameSuffix='' + curLoc)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    # For the objects present in bins like: nuts, washers, gears, emptyBin.

    nImgsOfEachSample = [nImgs]
    listOfSampleNames = [sampleName0, sampleName1, sampleName2, sampleName3]
    listOfSampleLocs = [sampleLoc0, sampleLoc1, sampleLoc2, sampleLoc3]
    listOfMaskLocs = [maskLoc0, maskLoc1, maskLoc2, maskLoc3]
    listOfNimgs = nImgsOfEachSample*len(listOfSampleNames)
    
    for i in range(len(listOfSampleLocs)):
        sampleNameI = listOfSampleNames[i]
        sampleLocI = listOfSampleLocs[i]
        maskLocI = listOfMaskLocs[i]
        nImgsI = listOfNimgs[i]

        print('Creating {} images: {}, {}.'.format(nImgsI, sampleNameI, curLoc))

        # Images having a ONE instance of an object.
        singleInstance(sampleLoc=sampleLocI, \
                       maskLoc=maskLocI, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                       labelSaveLoc=labelSaveLoc, nImgs=nImgsI, \
                       saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                       do90degFlips=False, doHoriFlip=True, doVertFlip=False, \
                       doRandomRot=False)
        cv2.destroyAllWindows()

################################################################################
################################################################################

    # For crankArmW.
    
    print('Creating {} images: {}, {}.'.format(nImgs, sampleName4, curLoc))

    # Images having a ONE instance of an object.
    singleInstance(sampleLoc=sampleLoc4, \
                   maskLoc=maskLoc4, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                   labelSaveLoc=labelSaveLoc, nImgs=nImgs, \
                   saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                   do90degFlips=True, doHoriFlip=True, doVertFlip=True, \
                   doRandomRot=False)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    # For crankShaft.
    
    print('Creating {} images: {}, {}.'.format(nImgs, sampleName5, curLoc))

    # Images having a ONE instance of an object.
    singleInstance(sampleLoc=sampleLoc5, \
                   maskLoc=maskLoc5, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                   labelSaveLoc=labelSaveLoc, nImgs=nImgs, \
                   saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                   do90degFlips=True, doHoriFlip=True, doVertFlip=True, \
                   doRandomRot=False)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    nImgs = 200
    
    nImgsOfEachSample = [nImgs]
    listOfSampleNames = [sampleName0, sampleName1, sampleName2, sampleName3, sampleName4, sampleName5]
    listOfSampleLocs = [sampleLoc0, sampleLoc1, sampleLoc2, sampleLoc3, sampleLoc4, sampleLoc5]
    listOfMaskLocs = [maskLoc0, maskLoc1, maskLoc2, maskLoc3, maskLoc4, maskLoc5]
    listOfNimgs = nImgsOfEachSample*len(listOfSampleNames)
    
    for i in range(len(listOfSampleLocs)):
        sampleNameI = listOfSampleNames[i]
        sampleLocI = listOfSampleLocs[i]
        maskLocI = listOfMaskLocs[i]
        nImgsI = listOfNimgs[i]
        
        for j in range(i, len(listOfSampleLocs)):
            sampleNameJ = listOfSampleNames[j]
            sampleLocJ = listOfSampleLocs[j]
            maskLocJ = listOfMaskLocs[j]

            print('Creating {} images: {} + {}, {}.'.format(nImgsI, sampleNameI, sampleNameJ, curLoc))

            doubleInstance(sampleLoc1=sampleLocI, sampleLoc2=sampleLocJ, \
                           maskLoc1=maskLocI, maskLoc2=maskLocJ, bgLoc=bgLocNormal, \
                           imgSaveLoc=imgSaveLoc, labelSaveLoc=labelSaveLoc, \
                           nImgs=nImgsI, imgH=imgH, imgW=imgW, saveNameSuffix='' + curLoc, \
                           do90degFlips=False, doHoriFlip=True, doVertFlip=False, \
                           doRandomRot=False)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

    curLoc = 'test'
    nImgs = 400
    
    imgH, imgW = 480, 640

    imgSaveLoc = os.path.join(curLoc, 'images')
    labelSaveLoc = os.path.join(curLoc, 'labels')

    # Create the directories if not present.
    if not os.path.exists(imgSaveLoc):    os.makedirs(imgSaveLoc)
    if not os.path.exists(labelSaveLoc):    os.makedirs(labelSaveLoc)

################################################################################
################################################################################
    '''
    -------------------------------------------------------------------------
     background             = 400
     nuts                   = 400  
     washers                = 400  
     gears                  = 400   
     empty_bin              = 400
     crankArmW              = 400
     crankShaft             = 400
     Two same objects       = 200 * 6 (6 objects, excluding background)
     Two different objects  = 200 * 15 (each combination. Total 15 combinations)
    -------------------------------------------------------------------------
     TOTAL                  = 7000 
    -------------------------------------------------------------------------
    '''
################################################################################
################################################################################

    bgLocNormal = os.path.join('background', curLoc)
    sampleName0 = 'nuts'
    sampleLoc0 = os.path.join('color_images', sampleName0, curLoc)
    maskLoc0 = os.path.join('color_images', sampleName0 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc0, maskDir=maskLoc0, replaceOriginal=True) # Making tight masks.
    sampleName1 = 'washers'
    sampleLoc1 = os.path.join('color_images', sampleName1, curLoc)
    maskLoc1 = os.path.join('color_images', sampleName1 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc1, maskDir=maskLoc1, replaceOriginal=True) # Making tight masks.
    sampleName2 = 'gears'
    sampleLoc2 = os.path.join('color_images', sampleName2, curLoc)
    maskLoc2 = os.path.join('color_images', sampleName2 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc2, maskDir=maskLoc2, replaceOriginal=True) # Making tight masks.
    sampleName3 = 'emptyBin'
    sampleLoc3 = os.path.join('color_images', sampleName3, curLoc)
    maskLoc3 = os.path.join('color_images', sampleName3 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc3, maskDir=maskLoc3, replaceOriginal=True) # Making tight masks.
    sampleName4 = 'crankArmW'
    sampleLoc4 = os.path.join('color_images', sampleName4, curLoc)
    maskLoc4 = os.path.join('color_images', sampleName4 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc4, maskDir=maskLoc4, replaceOriginal=True) # Making tight masks.
    sampleName5 = 'crankShaft'
    sampleLoc5 = os.path.join('color_images', sampleName5, curLoc)
    maskLoc5 = os.path.join('color_images', sampleName5 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc5, maskDir=maskLoc5, replaceOriginal=True) # Making tight masks.

################################################################################
################################################################################

    print('Creating {} images: blank background, {}.'.format(nImgs, curLoc))

    blankBackground(bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, labelSaveLoc=labelSaveLoc, \
                    nImgs=nImgs, imgH=imgH, imgW=imgW, saveNameSuffix='' + curLoc)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    # For the objects present in bins like: nuts, washers, gears, emptyBin.

    nImgsOfEachSample = [nImgs]
    listOfSampleNames = [sampleName0, sampleName1, sampleName2, sampleName3]
    listOfSampleLocs = [sampleLoc0, sampleLoc1, sampleLoc2, sampleLoc3]
    listOfMaskLocs = [maskLoc0, maskLoc1, maskLoc2, maskLoc3]
    listOfNimgs = nImgsOfEachSample*len(listOfSampleNames)
    
    for i in range(len(listOfSampleLocs)):
        sampleNameI = listOfSampleNames[i]
        sampleLocI = listOfSampleLocs[i]
        maskLocI = listOfMaskLocs[i]
        nImgsI = listOfNimgs[i]

        print('Creating {} images: {}, {}.'.format(nImgsI, sampleNameI, curLoc))

        # Images having a ONE instance of an object.
        singleInstance(sampleLoc=sampleLocI, \
                       maskLoc=maskLocI, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                       labelSaveLoc=labelSaveLoc, nImgs=nImgsI, \
                       saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                       do90degFlips=False, doHoriFlip=True, doVertFlip=False, \
                       doRandomRot=False)
        cv2.destroyAllWindows()

################################################################################
################################################################################

    # For crankArmW.
    
    print('Creating {} images: {}, {}.'.format(nImgs, sampleName4, curLoc))

    # Images having a ONE instance of an object.
    singleInstance(sampleLoc=sampleLoc4, \
                   maskLoc=maskLoc4, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                   labelSaveLoc=labelSaveLoc, nImgs=nImgs, \
                   saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                   do90degFlips=True, doHoriFlip=True, doVertFlip=True, \
                   doRandomRot=False)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    # For crankShaft.
    
    print('Creating {} images: {}, {}.'.format(nImgs, sampleName5, curLoc))

    # Images having a ONE instance of an object.
    singleInstance(sampleLoc=sampleLoc5, \
                   maskLoc=maskLoc5, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                   labelSaveLoc=labelSaveLoc, nImgs=nImgs, \
                   saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                   do90degFlips=True, doHoriFlip=True, doVertFlip=True, \
                   doRandomRot=False)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    nImgs = 200
    
    nImgsOfEachSample = [nImgs]
    listOfSampleNames = [sampleName0, sampleName1, sampleName2, sampleName3, sampleName4, sampleName5]
    listOfSampleLocs = [sampleLoc0, sampleLoc1, sampleLoc2, sampleLoc3, sampleLoc4, sampleLoc5]
    listOfMaskLocs = [maskLoc0, maskLoc1, maskLoc2, maskLoc3, maskLoc4, maskLoc5]
    listOfNimgs = nImgsOfEachSample*len(listOfSampleNames)
    
    for i in range(len(listOfSampleLocs)):
        sampleNameI = listOfSampleNames[i]
        sampleLocI = listOfSampleLocs[i]
        maskLocI = listOfMaskLocs[i]
        nImgsI = listOfNimgs[i]
        
        for j in range(i, len(listOfSampleLocs)):
            sampleNameJ = listOfSampleNames[j]
            sampleLocJ = listOfSampleLocs[j]
            maskLocJ = listOfMaskLocs[j]

            print('Creating {} images: {} + {}, {}.'.format(nImgsI, sampleNameI, sampleNameJ, curLoc))

            doubleInstance(sampleLoc1=sampleLocI, sampleLoc2=sampleLocJ, \
                           maskLoc1=maskLocI, maskLoc2=maskLocJ, bgLoc=bgLocNormal, \
                           imgSaveLoc=imgSaveLoc, labelSaveLoc=labelSaveLoc, \
                           nImgs=nImgsI, imgH=imgH, imgW=imgW, saveNameSuffix='' + curLoc, \
                           do90degFlips=False, doHoriFlip=True, doVertFlip=False, \
                           doRandomRot=False)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

    curLoc = 'train'
    nImgs = 40

    imgH, imgW = 480, 640

    imgSaveLoc = os.path.join('trial', 'images')
    labelSaveLoc = os.path.join('trial', 'labels')

    # Create the directories if not present.
    if not os.path.exists(imgSaveLoc):    os.makedirs(imgSaveLoc)
    if not os.path.exists(labelSaveLoc):    os.makedirs(labelSaveLoc)

################################################################################
################################################################################
    '''
    -------------------------------------------------------------------------
     background             = 40
     nuts                   = 40  
     washers                = 40  
     gears                  = 40   
     empty_bin              = 40
     crankArmW              = 40
     crankShaft             = 40
     Two same objects       = 20 * 6 (6 objects, excluding background)
     Two different objects  = 20 * 15 (each combination. Total 15 combinations)
    -------------------------------------------------------------------------
     TOTAL                  = 700 
    -------------------------------------------------------------------------
    '''
################################################################################
################################################################################

    bgLocNormal = os.path.join('background', curLoc)
    sampleName0 = 'nuts'
    sampleLoc0 = os.path.join('color_images', sampleName0, curLoc)
    maskLoc0 = os.path.join('color_images', sampleName0 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc0, maskDir=maskLoc0, replaceOriginal=True) # Making tight masks.
    sampleName1 = 'washers'
    sampleLoc1 = os.path.join('color_images', sampleName1, curLoc)
    maskLoc1 = os.path.join('color_images', sampleName1 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc1, maskDir=maskLoc1, replaceOriginal=True) # Making tight masks.
    sampleName2 = 'gears'
    sampleLoc2 = os.path.join('color_images', sampleName2, curLoc)
    maskLoc2 = os.path.join('color_images', sampleName2 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc2, maskDir=maskLoc2, replaceOriginal=True) # Making tight masks.
    sampleName3 = 'emptyBin'
    sampleLoc3 = os.path.join('color_images', sampleName3, curLoc)
    maskLoc3 = os.path.join('color_images', sampleName3 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc3, maskDir=maskLoc3, replaceOriginal=True) # Making tight masks.
    sampleName4 = 'crankArmW'
    sampleLoc4 = os.path.join('color_images', sampleName4, curLoc)
    maskLoc4 = os.path.join('color_images', sampleName4 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc4, maskDir=maskLoc4, replaceOriginal=True) # Making tight masks.
    sampleName5 = 'crankShaft'
    sampleLoc5 = os.path.join('color_images', sampleName5, curLoc)
    maskLoc5 = os.path.join('color_images', sampleName5 + '_masks', curLoc)
    createTightMasksAndSamples(imgDir=sampleLoc5, maskDir=maskLoc5, replaceOriginal=True) # Making tight masks.

################################################################################
################################################################################

    print('Creating {} images: background, {}.'.format(nImgs, curLoc))

    blankBackground(bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, labelSaveLoc=labelSaveLoc, \
                    nImgs=nImgs, imgH=imgH, imgW=imgW, saveNameSuffix='' + curLoc)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    # For the objects present in bins like: nuts, washers, gears, emptyBin.

    nImgsOfEachSample = [nImgs]
    listOfSampleNames = [sampleName0, sampleName1, sampleName2, sampleName3]
    listOfSampleLocs = [sampleLoc0, sampleLoc1, sampleLoc2, sampleLoc3]
    listOfMaskLocs = [maskLoc0, maskLoc1, maskLoc2, maskLoc3]
    listOfNimgs = nImgsOfEachSample*len(listOfSampleNames)
    
    for i in range(len(listOfSampleLocs)):
        sampleNameI = listOfSampleNames[i]
        sampleLocI = listOfSampleLocs[i]
        maskLocI = listOfMaskLocs[i]
        nImgsI = listOfNimgs[i]

        print('Creating {} images: {}, {}.'.format(nImgsI, sampleNameI, curLoc))

        # Images having a ONE instance of an object.
        singleInstance(sampleLoc=sampleLocI, \
                       maskLoc=maskLocI, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                       labelSaveLoc=labelSaveLoc, nImgs=nImgsI, \
                       saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                       do90degFlips=False, doHoriFlip=True, doVertFlip=False, \
                       doRandomRot=False)
        cv2.destroyAllWindows()
        
################################################################################
################################################################################

    # For crankArmW.
    
    print('Creating {} images: {}, {}.'.format(nImgs, sampleName4, curLoc))

    # Images having a ONE instance of an object.
    singleInstance(sampleLoc=sampleLoc4, \
                   maskLoc=maskLoc4, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                   labelSaveLoc=labelSaveLoc, nImgs=nImgs, \
                   saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                   do90degFlips=True, doHoriFlip=True, doVertFlip=True, \
                   doRandomRot=False)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    # For crankShaft.
    
    print('Creating {} images: {}, {}.'.format(nImgs, sampleName5, curLoc))

    # Images having a ONE instance of an object.
    singleInstance(sampleLoc=sampleLoc5, \
                   maskLoc=maskLoc5, bgLoc=bgLocNormal, imgSaveLoc=imgSaveLoc, \
                   labelSaveLoc=labelSaveLoc, nImgs=nImgs, \
                   saveNameSuffix='' + curLoc, imgH=imgH, imgW=imgW, \
                   do90degFlips=True, doHoriFlip=True, doVertFlip=True, \
                   doRandomRot=False)
    cv2.destroyAllWindows()

################################################################################
################################################################################

    nImgs = 20
    
    nImgsOfEachSample = [nImgs]
    listOfSampleNames = [sampleName0, sampleName1, sampleName2, sampleName3, sampleName4, sampleName5]
    listOfSampleLocs = [sampleLoc0, sampleLoc1, sampleLoc2, sampleLoc3, sampleLoc4, sampleLoc5]
    listOfMaskLocs = [maskLoc0, maskLoc1, maskLoc2, maskLoc3, maskLoc4, maskLoc5]
    listOfNimgs = nImgsOfEachSample*len(listOfSampleNames)
    
    for i in range(len(listOfSampleLocs)):
        sampleNameI = listOfSampleNames[i]
        sampleLocI = listOfSampleLocs[i]
        maskLocI = listOfMaskLocs[i]
        nImgsI = listOfNimgs[i]
        
        for j in range(i, len(listOfSampleLocs)):
            sampleNameJ = listOfSampleNames[j]
            sampleLocJ = listOfSampleLocs[j]
            maskLocJ = listOfMaskLocs[j]

            print('Creating {} images: {} + {}, {}.'.format(nImgsI, sampleNameI, sampleNameJ, curLoc))

            doubleInstance(sampleLoc1=sampleLocI, sampleLoc2=sampleLocJ, \
                           maskLoc1=maskLocI, maskLoc2=maskLocJ, bgLoc=bgLocNormal, \
                           imgSaveLoc=imgSaveLoc, labelSaveLoc=labelSaveLoc, \
                           nImgs=nImgsI, imgH=imgH, imgW=imgW, saveNameSuffix='' + curLoc, \
                           do90degFlips=False, doHoriFlip=True, doVertFlip=False, \
                           doRandomRot=False)

################################################################################
################################################################################

    ## Check the clusters in the bounding box dimensions and then create the 
    ## anchor boxes based on those dimensions.
    #plotBboxDimensions(dataDir='trial', showFig=True, saveFig=True, figSaveLoc='.')
    
################################################################################
################################################################################
################################################################################



