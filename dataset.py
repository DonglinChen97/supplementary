################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Dataset handling
#
################

from torch.utils.data import Dataset
import numpy as np
from os import listdir
import random
import math


# global switch, use fixed max values for dim-less airfoil data?
fixedAirfoilNormalization = False
# global switch, make data dimensionless?
makeDimLess = False
# global switch, remove constant offsets from pressure channel?
removePOffset = True

## helper - compute absolute of inputs or targets

def process(filename):
    lis = filename.split('_')
    length = len(lis)
    a = lis[length -2]
    a = float(a)/100
    b = lis[length -1].split('.')[0]
    b =float(b)/100
    v = np.sqrt(a*a + b*b)
    angle = 180 * math.atan(b/a)/math.pi 
    # if length >3 :
        # print(filename)
    return v,angle #a,b


def find_absmax(data, use_targets, x):
    maxval = 0
    is_input2 = 0
    for i in range(data.totalLength):
        if use_targets == 0:
            if x == 0:
                temp_tensor = data.inputs[i]
            else:
                temp_tensor = data.inputs2[i]
                is_input2 = 1
        else:
            temp_tensor = data.targets[i]
        if is_input2 == 0:
            temp_max = np.max(np.abs(temp_tensor[x]))
        else:
            temp_max = temp_tensor[x-1]
        if temp_max > maxval:
            maxval = temp_max
    return maxval

######################################## DATA LOADER #########################################
#         also normalizes data with max , and optionally makes it dimensionless              #

def LoaderNormalizer(data, isTest = False, shuffle = 0, dataProp = None):
    """
    # data: pass TurbDataset object with initialized dataDir / dataDirTest paths
    # train: when off, process as test data (first load regular for normalization if needed, then replace by test data)
    # dataProp: proportions for loading & mixing 3 different data directories "reg", "shear", "sup"
    #           should be array with [total-length, fraction-regular, fraction-superimposed, fraction-sheared],
    #           passing None means off, then loads from single directory
    """
    if isTest:
        files = listdir(data.dataDirTest)
        files.sort()
        data.totalLength = len(files)
        data.inputs  = np.empty((len(files), 1, 128, 128))
        # data.inputs2  = np.empty((len(files), 2))
        data.targets = np.empty((len(files), 2, 128, 128))
        
        for i, file in enumerate(files):
            npfile = np.load(data.dataDirTest + file)
            # streamx, streamy = process(file)
            d = npfile['a']
            data.inputs[i] = d[0]
            # data.inputs2[i] = [streamx,streamy]
            data.targets[i] = d[1:3]
    else:
        if dataProp is None:
            # load single directory
            files = listdir(data.dataDir)
            files.sort()
            for i in range(shuffle):
                random.shuffle(files) 
            data.totalLength = len(files)
            data.inputs  = np.empty((len(files), 1, 128, 128))
            # data.inputs2  = np.empty((len(files), 2))
            data.targets = np.empty((len(files), 2, 128, 128))


            for i, file in enumerate(files):
                npfile = np.load(data.dataDir + file)
                # streamx, streamy = process(file)
                d = npfile['a']
                data.inputs[i] = d[0]
                # data.inputs2[i] = [streamx,streamy]
                data.targets[i] = d[1:3]
            print("Number of data loaded:", len(data.inputs) )

   ################################## NORMALIZATION OF TRAINING DATA ##########################################

    data.max_inputs_0 = 100.
    data.max_inputs_1 = 38.12
    data.max_inputs_2 = 1.0

    data.max_targets_0 = 200.
    data.max_targets_1 = 216.




    data.targets[:,0,:,:] *= (1.0/data.max_targets_0)
    data.targets[:,1,:,:] *= (1.0/data.max_targets_1)

    # data.targets = data.targets[:,1:3,:,:]

    print("Data stats, input  mean %f, max  %f;   targets mean %f , max %f " % ( 
      np.mean(np.abs(data.targets), keepdims=False), np.max(np.abs(data.targets), keepdims=False) , 
      np.mean(np.abs(data.inputs), keepdims=False) , np.max(np.abs(data.inputs), keepdims=False) ) ) 

    return data

######################################## DATA SET CLASS #########################################

class TurbDataset(Dataset):

    # mode "enum" , pass to mode param of TurbDataset (note, validation mode is not necessary anymore)
    TRAIN = 0
    TEST  = 2

    def __init__(self, dataProp=None, mode=TRAIN, dataDir="./data/train/", dataDirTest="../data/test/", shuffle=0, normMode=0):
        global makeDimLess, removePOffset
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """
        if not (mode==self.TRAIN or mode==self.TEST):
            print("Error - TurbDataset invalid mode "+format(mode) ); exit(1)

        if normMode==1:	
            print("Warning - poff off!!")
            removePOffset = False
        if normMode==2:	
            print("Warning - poff and dimless off!!!")
            makeDimLess = False
            removePOffset = False

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest # only for mode==self.TEST

        # load & normalize data
        self = LoaderNormalizer(self, isTest=(mode==self.TEST), dataProp=dataProp, shuffle=shuffle)

        if not self.mode==self.TEST:
            # split for train/validation sets (80/20) , max 400
            targetLength = self.totalLength - min( int(self.totalLength*0.2) , 100)

            self.valiInputs = self.inputs[targetLength:]

            # self.valiInputs2 = self.inputs2[targetLength:]

            self.valiTargets = self.targets[targetLength:]
            self.valiLength = self.totalLength - targetLength

            self.inputs = self.inputs[:targetLength]

            # self.inputs2 = self.inputs2[:targetLength]

            self.targets = self.targets[:targetLength]
            self.totalLength = self.inputs.shape[0]

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# simplified validation data set (main one is TurbDataset above)

class ValiDataset(TurbDataset):
    def __init__(self, dataset): 
        self.inputs = dataset.valiInputs
        # self.inputs2 = dataset.valiInputs2
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

