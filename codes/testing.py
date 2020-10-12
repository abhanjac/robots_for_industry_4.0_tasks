# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:28:52 2019

@author: abhanjac
"""

from utils import *
from network import *

#===============================================================================

if __name__ == '__main__':
    
    trainDir = 'train2'
    validDir = 'valid2'
    testDir = 'test2'
    trialDir = 'trial2'

    #detector = networkDetector()
    #detector.test(testDir=testDir)
    #detector = networkDetector()
    #detector.test(testDir=validDir)

#-------------------------------------------------------------------------------
   
    inferDir = trialDir
    key = '`'
    listOfImg = os.listdir(os.path.join(inferDir, 'images'))
    np.random.shuffle(listOfImg)    # Shuffling the list randomly.
    nImgs = len(listOfImg)

    detector = networkDetector()
    
#-------------------------------------------------------------------------------

    for idx, i in enumerate(listOfImg[:20]):
        startTime = time.time()
        
        labelDictList, multiHotLabel = getImgLabelClassification(inferDir, i)
        multiHotLabel = multiHotLabel.tolist()
        
#        # Skip images if needed.
#        if i.find('_ni_') == -1:   continue
        
        # Prediction from network.
        img = cv2.imread(os.path.join(inferDir, 'images', i))
        img1 = copy.deepcopy(img)
        img2 = copy.deepcopy(img)
        imgBatch = np.array([img])
        
        inferLayerOut, inferPredLogits, inferPredResult, _, _ = detector.batchInference(imgBatch)
        
        detectedBatchClassScores, _, detectedBatchClassNames, detectedBatchBboxes \
                                            = nonMaxSuppression(inferPredResult)
        
        # The output of the nonMaxSuppression is in the form of a batch.
        # So extracting the contents of this batch since there is an output 
        # of only one image in this batch.        
        detectedBatchClassScores = detectedBatchClassScores[0]
        detectedBatchClassNames = detectedBatchClassNames[0]
        detectedBatchBboxes = detectedBatchBboxes[0]
                
#-------------------------------------------------------------------------------

        # Draw the ground truth results now.
        for l in labelDictList:
            tlX, tlY, bboxW, bboxH = l['tlX'], l['tlY'], l['bboxW'], l['bboxH']
            posX, posY = l['posX'], l['posY']
            trueName = l['className']

            # Only draw the bounding boxes for the non-rbc entities.
            cv2.rectangle(img1, (tlX, tlY), (tlX + bboxW, tlY + bboxH), (0,255,0), 2)
            cv2.putText(img1, trueName, (posX, posY), \
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

#-------------------------------------------------------------------------------

        # Draw the detected results now.
        for pdx, p in enumerate(detectedBatchClassNames):
            x, y, w, h = detectedBatchBboxes[pdx].tolist()

            # Only draw the bounding boxes for the non-rbc entities.
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(img1, p, (x+5, y+15), \
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

            score = detectedBatchClassScores[pdx]
            #print(p, score)

#-------------------------------------------------------------------------------
                
        print('\nTime taken: {}'.format(prettyTime(time.time() - startTime)))
        cv2.imwrite(os.path.join(inferDir, 'saved', 'prediction_' + i), img1)

        cv2.imshow('Image', img1)
        cv2.imshow('Original Image', img)
        print('[{}/{}]\tImage name: {}'.format(idx+1, nImgs, i))
        key = cv2.waitKey(2000)
        if key & 0xFF == 27:    break    # break with esc key.
    
    cv2.destroyAllWindows()

#-------------------------------------------------------------------------------
