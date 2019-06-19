import os
import sys
import h5py
import re
import csv
import numpy as np
from glob import glob
import scipy
import pickle

def init(mdlParams_):
    mdlParams = {}
    # Save summaries and model here
    mdlParams['saveDir'] = mdlParams_['pathBase']+'/data/isic/'
    # Data is loaded from here
    mdlParams['dataDir'] = mdlParams_['pathBase']+'/data/isic/task3'
    # Number of GPUs on device
    mdlParams['numGPUs'] = [0]

    ### Model Selection ###
    mdlParams['model_type'] = 'CNN_FC'
    mdlParams['input_func_type'] = 'ISIC' 
    mdlParams['dataset_names'] = ['HAM10000']
    mdlParams['same_sized_crops'] = True
    mdlParams['multiCropTrain']= 5
    mdlParams['multiCropEval'] = 5
    mdlParams['orderedCrop'] = True
    mdlParams['orderedCropTrain'] = True
    mdlParams['voting_scheme'] = 'average'    
    mdlParams['classification'] = True
    # A type of class balancing. Available: 
    # 1: Use inverse class freq*numclasses
    # 2: Removed
    # 3: Balanced batch sampling (repeat underrepresented examples)
    # 4: Removed
    # 5: Use inverse class freq
    # 6: Use inverse class freq + loss per example with diagnosis info
    # 7: Use inverse class freq*numclasses + loss per example with diagnosis info
    # 8: loss per example with diagnosis info
    # 9: Use inverse class freq only based on HAM       
    mdlParams['balance_classes'] = 5
    mdlParams['extra_fac'] = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0])
    # Loss is multiplied by this factor for diagnosis types: "expert consensus", "serial imaging showed not change", "confocal microscopy", "histopathology"
    mdlParams['diagnosis_type_penalty'] = np.array([1.0,2.0,3.0,4.0])
    mdlParams['setMean'] = np.array([0,0,0])    
    mdlParams['numClasses'] = 7
    mdlParams['numOut'] = mdlParams['numClasses']
    mdlParams['numCV'] = 3
    mdlParams['new_inds'] = True

    ### CNN_FC Parameters ###
    # Base CNN after convGRU
    mdlParams['model_type_cnn'] = 'InceptionV3'
    # Extra lr for convgru
    mdlParams['learning_rate_congru'] = (0.0000125)*(5.0/3.0)*len(mdlParams['numGPUs'])   
    # Number of CNN features
    mdlParams['CNN_Features'] = 2048
    # Which is the output point?
    mdlParams['CNN_Output_Point'] = 'end'
    # Randomly permute order?
    mdlParams['randPerm'] = False
    # How many times to repeat it for eval?
    mdlParams['numRandValSeq'] = 0
    # Patch dropout?
    mdlParams['patch_dropout'] = 0.1
    # Bidirectional
    mdlParams['bidirectional'] = False    
    mdlParams['with_global_path'] = False    
    mdlParams['aux_classifier'] = False
    mdlParams['combine_features'] = 'add'
    # Attention
    mdlParams['initial_attention'] = True
    # List, first param for inital, second for end
    #mdlParams['attention_size'] = [(2,2,3)]
    mdlParams['end_attention'] = False
    mdlParams['end_pool'] = True

    ### Training Parameters ###
    # Batch size
    mdlParams['batchSize'] = 5*len(mdlParams['numGPUs'])
    # Initial learning rate
    mdlParams['learning_rate'] = (0.0000125)*(5.0/3.0)*len(mdlParams['numGPUs'])
    # Lower learning rate after no improvement over 100 epochs
    mdlParams['lowerLRAfter'] = 10
    # If there is no validation set, start lowering the LR after X steps
    mdlParams['lowerLRat'] = 20
    # Divide learning rate by this value
    mdlParams['LRstep'] = 2
    # Maximum number of training iterations
    mdlParams['training_steps'] = 50 #250
    # Display error every X steps
    mdlParams['display_step'] = 5
    # Scale?
    mdlParams['scale_targets'] = False
    # Peak at test error during training? (generally, dont do this!)
    mdlParams['peak_at_testerr'] = False
    # Print trainerr
    mdlParams['print_trainerr'] = False
    # Decay of moving averages
    mdlParams['moving_avg_var_decay'] = 0.999
    # Compatibility to other models
    mdlParams['keep_prob_1'] = 1.0
    # Compatibility to other models
    mdlParams['keep_prob_2'] = 1.0
    # Subtract trainset mean?
    mdlParams['subtract_set_mean'] = False


    ### Data ###
    mdlParams['preload'] = False
    # Labels first
    # Targets, as dictionary, indexed by im file name
    mdlParams['labels_dict'] = {}
    path1 = mdlParams['dataDir'] + '/labels/'
     # All sets
    allSets = glob(path1 + '*/')   
    # Go through all sets
    for i in range(len(allSets)):
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
        if not foundSet:
            continue                
        # Find csv file
        files = sorted(glob(allSets[i]+'*'))
        for j in range(len(files)):
            if 'csv' in files[j]:
                break
        # Load csv file
        with open(files[j], newline='') as csvfile:
            labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in labels_str:
                if 'ISIC' not in row[0]:
                    continue
                mdlParams['labels_dict'][row[0]] = np.array([int(float(row[1])),int(float(row[2])),int(float(row[3])),int(float(row[4])),int(float(row[5])),int(float(row[6])),int(float(row[7]))])
    # Data size 
    mdlParams['input_size'] = [224,224,3]
    mdlParams['input_size_load'] = [450,600,3]
    # Save all im paths here
    mdlParams['im_paths'] = []
    mdlParams['labels_list'] = []
    # Define the sets
    path1 = mdlParams['dataDir'] + '/images/'
    # All sets
    allSets = sorted(glob(path1 + '*/'))
    # Ids which name the folders
    mdlParams['setName'] = {}
    # Find set lengths
    mdlParams['setSize'] = np.zeros([len(allSets)])
    # Setinds
    mdlParams['setInds'] = {}
    ind = 0
    # Make HAM10000 first dataset
    for i in range(len(allSets)):
        if 'HAM10000' in allSets[i]:
            temp = allSets[i]
            allSets.remove(allSets[i])
            allSets.insert(0, temp)
    print(allSets)        
    # Get diagnosis type
    mdlParams['diagnosis_dict'] = {}
    with open(mdlParams['dataDir']+"/ISIC2018_Task3_Training_LesionGroupings.csv", newline='') as csvfile:
        labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in labels_str:
            if 'ISIC' not in row[0]:
                continue
            # Distinguish diagnosis type
            if 'single' in row[2]:
                # Single image expert consensus
                mdlParams['diagnosis_dict'][row[0]] = mdlParams['diagnosis_type_penalty'][0]
            elif 'serial' in row[2]:
                # Serial imaging showed no change
                mdlParams['diagnosis_dict'][row[0]] = mdlParams['diagnosis_type_penalty'][1]
            elif 'confocal' in row[2]:
                # Confocal microscopy with consesus dermoscopy
                mdlParams['diagnosis_dict'][row[0]] = mdlParams['diagnosis_type_penalty'][2]
            elif 'histopathology' in row[2]:
                # Histopathology
                mdlParams['diagnosis_dict'][row[0]] = mdlParams['diagnosis_type_penalty'][3]                                
    mdlParams['loss_fac_per_example_list'] = []        
    # Extract setids
    for i in range(len(allSets)):
        # All files in that set
        files = sorted(glob(allSets[i]+'*'))
        # Check if there is something in there, if not, discard
        if len(files) == 0:
            continue
        # Check if want to include this dataset
        foundSet = False
        for j in range(len(mdlParams['dataset_names'])):
            if mdlParams['dataset_names'][j] in allSets[i]:
                foundSet = True
        if not foundSet:
            continue                    
        # Set set size
        mdlParams['setSize'][ind] = len(files)
        # Last is set number
        mdlParams['setName'][ind] = allSets[i]
        mdlParams['setInds'][mdlParams['setName'][ind]] = np.zeros([len(files)],dtype=np.int32)
        # Setinds
        for j in range(len(files)):
            inds = [int(s) for s in re.findall(r'\d+',files[j])]
            mdlParams['setInds'][mdlParams['setName'][ind]][j] = int(inds[-1])
            if 'ISIC_' in files[j]:
                mdlParams['im_paths'].append(files[j])
                # Add according label, find it first
                for key in mdlParams['labels_dict']:
                    if key in files[j]:
                        mdlParams['labels_list'].append(mdlParams['labels_dict'][key])
                # Add according weighting factor for diagnosis type, only for nevus
                if np.argmax(mdlParams['labels_dict'][key]) == 1:
                    for key in mdlParams['diagnosis_dict']:
                        if key in files[j]:
                            mdlParams['loss_fac_per_example_list'].append(mdlParams['diagnosis_dict'][key])        
                else:
                    mdlParams['loss_fac_per_example_list'].append(1.0)         
        # Sort inds
        mdlParams['setInds'][mdlParams['setName'][ind]] = np.sort(mdlParams['setInds'][mdlParams['setName'][ind]])
        # Inc
        ind = ind+1
    # Convert loss fac list to array
    mdlParams['loss_fac_per_example'] = np.array(mdlParams['loss_fac_per_example_list'])
    # Convert label list to array
    mdlParams['labels_array'] = np.zeros([len(mdlParams['labels_list']),mdlParams['numClasses']],dtype=np.float32)
    for i in range(len(mdlParams['labels_list'])):
        mdlParams['labels_array'][i,:] = mdlParams['labels_list'][i]
    print(np.mean(mdlParams['labels_array'],axis=0))          
    # Reduce to actual size
    mdlParams['setSize'] = mdlParams['setSize'][:ind]
    # set starts
    # Find set starts
    mdlParams['setStart'] = np.zeros([len(mdlParams['setName'])+1])
    mdlParams['setStart'][0] = 1
    for i in range(len(mdlParams['setName'])):
        mdlParams['setStart'][i+1] = mdlParams['setStart'][i]+mdlParams['setSize'][i]
    print("Set sizes",mdlParams['setSize'])
    print("Set Names",mdlParams['setName'])
    print("Set starts",mdlParams['setStart'])
    print(mdlParams['setInds'][mdlParams['setName'][0]])
    # Actual number of sets
    mdlParams['numSets'] = len(mdlParams['setSize'])

    ### Define Indices ###
    with open(mdlParams['saveDir'] + 'indices_tbe.pkl','rb') as f:
        indices = pickle.load(f)  
    mdlParams['trainIndCV'] = indices['trainIndCV']
    mdlParams['valIndCV'] = indices['valIndCV']
    mdlParams['testInd'] = indices['testInd']
    # Consider case with more than one set
    if len(mdlParams['dataset_names']) > 1:
        restInds = np.array(np.arange(10015,mdlParams['labels_array'].shape[0]))
        for i in range(mdlParams['numCV']):
            mdlParams['trainIndCV'][i] = np.concatenate((mdlParams['trainIndCV'][i],restInds))        
    print("Train")
    for i in range(len(mdlParams['trainIndCV'])):
        print(mdlParams['trainIndCV'][i].shape)
    print("Val")
    for i in range(len(mdlParams['valIndCV'])):
        print(mdlParams['valIndCV'][i].shape)    

    # Use this for ordered multi crops
    if mdlParams['orderedCrop']:
        # Crop positions, always choose multiCropEval to be 4, 9, 16, 25, etc.
        mdlParams['cropPositions'] = np.zeros([mdlParams['multiCropEval'],2],dtype=np.int64)
        if mdlParams['multiCropEval'] == 5:
            numCrops = 4
        elif mdlParams['multiCropEval'] == 7:
            numCrops = 9
            mdlParams['cropPositions'] = np.zeros([9,2],dtype=np.int64)                
        else:
            numCrops = mdlParams['multiCropEval']
        ind = 0
        for i in range(np.int32(np.sqrt(numCrops))):
            for j in range(np.int32(np.sqrt(numCrops))):
                mdlParams['cropPositions'][ind,0] = mdlParams['input_size'][0]/2+i*((mdlParams['input_size_load'][0]-mdlParams['input_size'][0])/(np.sqrt(numCrops)-1))
                mdlParams['cropPositions'][ind,1] = mdlParams['input_size'][1]/2+j*((mdlParams['input_size_load'][1]-mdlParams['input_size'][1])/(np.sqrt(numCrops)-1))
                ind += 1
        # Add center crop
        if mdlParams['multiCropEval'] == 5:
            mdlParams['cropPositions'][4,0] = mdlParams['input_size_load'][0]/2
            mdlParams['cropPositions'][4,1] = mdlParams['input_size_load'][1]/2
        if mdlParams['multiCropEval'] == 7:      
            mdlParams['cropPositions'] = np.delete(mdlParams['cropPositions'],[3,7],0)               
        # Crop positions training, always choose multiCropTrain to be 4, 9, 16, 25, etc.
        mdlParams['cropPositionsTrain'] = np.zeros([mdlParams['multiCropTrain'],2],dtype=np.int64)
        if mdlParams['multiCropTrain'] == 5:
            numCropsTrain = 4
        elif mdlParams['multiCropEval'] == 7:
            numCropsTrain = 9
            mdlParams['cropPositionsTrain'] = np.zeros([9,2],dtype=np.int64)            
        else:
            numCropsTrain = mdlParams['multiCropTrain']        
        ind = 0
        for i in range(np.int32(np.sqrt(numCropsTrain))):
            for j in range(np.int32(np.sqrt(numCropsTrain))):
                mdlParams['cropPositionsTrain'][ind,0] = mdlParams['input_size'][0]/2+i*((mdlParams['input_size_load'][0]-mdlParams['input_size'][0])/(np.sqrt(numCropsTrain)-1))
                mdlParams['cropPositionsTrain'][ind,1] = mdlParams['input_size'][1]/2+j*((mdlParams['input_size_load'][1]-mdlParams['input_size'][1])/(np.sqrt(numCropsTrain)-1))
                ind += 1
        # Add center crop
        if mdlParams['multiCropEval'] == 5:
            mdlParams['cropPositionsTrain'][4,0] = mdlParams['input_size_load'][0]/2
            mdlParams['cropPositionsTrain'][4,1] = mdlParams['input_size_load'][1]/2   
        if mdlParams['multiCropEval'] == 7:      
            mdlParams['cropPositionsTrain'] = np.delete(mdlParams['cropPositionsTrain'],[3,7],0)                                              
        # Sanity checks
        print("Positions val",mdlParams['cropPositions'])
        # Test image sizes
        test_im = np.zeros(mdlParams['input_size_load'])
        height = mdlParams['input_size'][0]
        width = mdlParams['input_size'][1]
        for i in range(mdlParams['multiCropEval']):
            im_crop = test_im[np.int32(mdlParams['cropPositions'][i,0]-height/2):np.int32(mdlParams['cropPositions'][i,0]-height/2)+height,np.int32(mdlParams['cropPositions'][i,1]-width/2):np.int32(mdlParams['cropPositions'][i,1]-width/2)+width,:]
            print("Shape",i+1,im_crop.shape)           
    return mdlParams