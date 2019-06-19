import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models as tv_models
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
import models
import threading
import pickle
from pathlib import Path
import math
import os
import sys
from glob import glob
import re
import gc
import importlib
import time
import csv
import sklearn.preprocessing
import utils
from sklearn.utils import class_weight
import custom_models

# add configuration file
# Dictionary for model configuration
mdlParams = {}

# Import machine config
pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
mdlParams.update(pc_cfg.mdlParams)

# Path name where model is saved is the fourth argument
mdlParams['saveDirBase'] = sys.argv[5]

# If there is another argument, its which checkpoint should be used
if len(sys.argv) > 6:
    if 'last' in sys.argv[6]:
        mdlParams['ckpt_name'] = 'checkpoint-'
    else:
        mdlParams['ckpt_name'] = 'checkpoint_best-'
    if 'first' in sys.argv[6]:
        mdlParams['use_first'] = True
else:
    mdlParams['ckpt_name'] = 'checkpoint-'

# Set visible devices
if 'gpu' in sys.argv[6]:
    mdlParams['numGPUs']= [[int(s) for s in re.findall(r'\d+',sys.argv[6])][-1]]
    cuda_str = ""
    for i in range(len(mdlParams['numGPUs'])):
        cuda_str = cuda_str + str(mdlParams['numGPUs'][i])
        if i is not len(mdlParams['numGPUs'])-1:
            cuda_str = cuda_str + ","
    print("Devices to use:",cuda_str)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str      

# If there is another argument, also use a meta learner
if len(sys.argv) > 7:
    if 'SVC' in sys.argv[7] or 'RF' in sys.argv[7]:
        mdlParams['learn_on_preds'] = True
        mdlParams['meta_learner'] = sys.argv[7]
    else:
        mdlParams['learn_on_preds'] = False
else:
    mdlParams['learn_on_preds'] = False

# Import model config
model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)

# Third is multi crop yes no
if 'multi' in sys.argv[3]:
    if 'rand' in sys.argv[3]:
        mdlParams['numRandValSeq'] = [int(s) for s in re.findall(r'\d+',sys.argv[3])][0]
        print("Random sequence number",mdlParams['numRandValSeq'])
    else:
        mdlParams['numRandValSeq'] = 0
    mdlParams['multiCropEval'] = [int(s) for s in re.findall(r'\d+',sys.argv[3])][-1]
    mdlParams['voting_scheme'] = sys.argv[4]
    if 'scale' in sys.argv[3]:
        print("Multi Crop and Scale Eval with crop number:",mdlParams['multiCropEval']," Voting scheme: ",mdlParams['voting_scheme'])
        mdlParams['orderedCrop'] = False
    elif 'order' in sys.argv[3]:
        mdlParams['orderedCrop'] = True
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
        # Sanity checks
        print("Positions val",mdlParams['cropPositions'])
        # Test image sizes
        test_im = np.zeros(mdlParams['input_size_load'])
        height = mdlParams['input_size'][0]
        width = mdlParams['input_size'][1]
        for i in range(mdlParams['multiCropEval']):
            im_crop = test_im[np.int32(mdlParams['cropPositions'][i,0]-height/2):np.int32(mdlParams['cropPositions'][i,0]-height/2)+height,np.int32(mdlParams['cropPositions'][i,1]-width/2):np.int32(mdlParams['cropPositions'][i,1]-width/2)+width,:]
            print("Shape",i+1,im_crop.shape)         
        print("Multi Crop with order with crop number:",mdlParams['multiCropEval']," Voting scheme: ",mdlParams['voting_scheme'])
    else:
        print("Multi Crop Eval with crop number:",mdlParams['multiCropEval']," Voting scheme: ",mdlParams['voting_scheme'])
        mdlParams['orderedCrop'] = False
else:
    mdlParams['multiCropEval'] = 0
    mdlParams['orderedCrop'] = False

# Set training set to eval mode
mdlParams['trainSetState'] = 'eval'

# Scaler, scales targets to a range of 0-1
if mdlParams['scale_targets']:
    mdlParams['scaler'] = sklearn.preprocessing.MinMaxScaler().fit(mdlParams['targets'][mdlParams['trainInd'],:][:,mdlParams['tar_range'].astype(int)])

if len(sys.argv) > 8 and 'sevenpoint' in sys.argv[8]:
    num_classes = mdlParams['numClasses']-1
else:
    num_classes = mdlParams['numClasses']

# Save results in here
allData = {}
allData['f1Best'] = np.zeros([mdlParams['numCV']])
allData['sensBest'] = np.zeros([mdlParams['numCV'],num_classes])
allData['specBest'] = np.zeros([mdlParams['numCV'],num_classes])
allData['accBest'] = np.zeros([mdlParams['numCV']])
allData['waccBest'] = np.zeros([mdlParams['numCV'],num_classes])
allData['aucBest'] = np.zeros([mdlParams['numCV'],num_classes])
allData['convergeTime'] = {}
allData['bestPred'] = {}
allData['bestPredMC'] = {}
allData['targets'] = {}
allData['extPred'] = {}
allData['f1Best_meta'] = np.zeros([mdlParams['numCV']])
allData['sensBest_meta'] = np.zeros([mdlParams['numCV'],num_classes])
allData['specBest_meta'] = np.zeros([mdlParams['numCV'],num_classes])
allData['accBest_meta'] = np.zeros([mdlParams['numCV']])
allData['waccBest_meta'] = np.zeros([mdlParams['numCV'],num_classes])
allData['aucBest_meta'] = np.zeros([mdlParams['numCV'],num_classes])
#allData['convergeTime'] = {}
allData['bestPred_meta'] = {}
allData['targets_meta'] = {}

if not (len(sys.argv) > 8 and ('sevenpoint' in sys.argv[8] or 'ISIC_Rest' in sys.argv[8] or 'HAM' in sys.argv[8])):
    for cv in range(mdlParams['numCV']):
        # Reset model graph 
        importlib.reload(models)
        #importlib.reload(torchvision)
        # Collect model variables
        modelVars = {}
        modelVars['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(modelVars['device'])
        # Def current CV set
        mdlParams['trainInd'] = mdlParams['trainIndCV'][cv]
        if 'valIndCV' in mdlParams:
            mdlParams['valInd'] = mdlParams['valIndCV'][cv]
        # Def current path for saving stuff
        if 'valIndCV' in mdlParams:
            mdlParams['saveDir'] = mdlParams['saveDirBase'] + '/CVSet' + str(cv)
        else:
            mdlParams['saveDir'] = mdlParams['saveDirBase']

        # Potentially calculate setMean to subtract
        if mdlParams['subtract_set_mean'] == 1:
            mdlParams['setMean'] = np.mean(mdlParams['images_means'][mdlParams['trainInd'],:],(0))
            print("Set Mean",mdlParams['setMean']) 

        # balance classes
        if mdlParams['balance_classes'] < 3 or mdlParams['balance_classes'] == 7 or mdlParams['balance_classes'] == 11:
            class_weights = class_weight.compute_class_weight('balanced',np.unique(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1)),np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1)) 
            print("Current class weights",class_weights)
            class_weights = class_weights*mdlParams['extra_fac']
            print("Current class weights with extra",class_weights)             
        elif mdlParams['balance_classes'] == 3 or mdlParams['balance_classes'] == 4:
            # Split training set by classes
            not_one_hot = np.argmax(mdlParams['labels_array'],1)
            mdlParams['class_indices'] = []
            for i in range(mdlParams['numClasses']):
                mdlParams['class_indices'].append(np.where(not_one_hot==i)[0])
                # Kick out non-trainind indices
                mdlParams['class_indices'][i] = np.setdiff1d(mdlParams['class_indices'][i],mdlParams['valInd'])
                #print("Class",i,mdlParams['class_indices'][i].shape,np.min(mdlParams['class_indices'][i]),np.max(mdlParams['class_indices'][i]),np.sum(mdlParams['labels_array'][np.int64(mdlParams['class_indices'][i]),:],0))        
        elif mdlParams['balance_classes'] == 5 or mdlParams['balance_classes'] == 6 or mdlParams['balance_classes'] == 13:
            # Other class balancing loss
            class_weights = 1.0/np.mean(mdlParams['labels_array'][mdlParams['trainInd'],:],axis=0)
            print("Current class weights",class_weights) 
            class_weights = class_weights*mdlParams['extra_fac']
            print("Current class weights with extra",class_weights) 
        elif mdlParams['balance_classes'] == 9:
            # Only use HAM indicies for calculation
            indices_ham = mdlParams['trainInd'][mdlParams['trainInd'] < 10015]
            class_weights = 1.0/np.mean(mdlParams['labels_array'][indices_ham,:],axis=0)
            print("Current class weights",class_weights) 


        # Set up dataloaders
        if mdlParams.get('model_type_cnn') is None:
            # For train
            dataset_train = utils.ISICDataset(mdlParams, 'trainInd')
            # For val
            dataset_val = utils.ISICDataset(mdlParams, 'valInd')
            if mdlParams['multiCropEval'] > 0:
                modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=8, pin_memory=True)  
            else:
                modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=8, pin_memory=True)         
        else:
            # For train
            dataset_train = utils.ISICDataset_custom(mdlParams, 'trainInd')
            # For val
            dataset_val = utils.ISICDataset_custom(mdlParams, 'valInd')   
            if mdlParams['numRandValSeq'] > 0:
                modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['numRandValSeq'], shuffle=False, num_workers=8, pin_memory=True)  
            else:
                modelVars['dataloader_valInd'] = DataLoader(dataset_val, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=8, pin_memory=True)          
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=8, pin_memory=True)
    
        # For test
        if 'testInd' in mdlParams:
            if mdlParams.get('model_type_cnn') is None:
                dataset_test = utils.ISICDataset(mdlParams, 'testInd')
                if mdlParams['multiCropEval'] > 0:
                    modelVars['dataloader_testInd'] = DataLoader(dataset_test, batch_size=mdlParams['multiCropEval'], shuffle=False, num_workers=8, pin_memory=True)  
                else:
                    modelVars['dataloader_testInd'] = DataLoader(dataset_test, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=8, pin_memory=True)            
            else:
                dataset_test = utils.ISICDataset_custom(mdlParams, 'testInd')
                if mdlParams['numRandValSeq'] > 0:
                    modelVars['dataloader_testInd'] = DataLoader(dataset_test, batch_size=mdlParams['numRandValSeq'], shuffle=False, num_workers=8, pin_memory=True)  
                else:
                    modelVars['dataloader_testInd'] = DataLoader(dataset_test, batch_size=mdlParams['batchSize'], shuffle=False, num_workers=8, pin_memory=True)             
            
        #print("Setdiff",np.setdiff1d(mdlParams['trainInd'],mdlParams['trainInd']))
        # Define model 
        if 'CNN_GRU_TP' in mdlParams['model_type']:
            modelVars['model'] = custom_models.CNN_GRU_TP(mdlParams)              
        elif 'CNN_GRU' in mdlParams['model_type']:
            modelVars['model'] = custom_models.CNN_GRU(mdlParams)  
        elif 'CNN_FC' in mdlParams['model_type']:
            modelVars['model'] = custom_models.CNN_FC(mdlParams)
        else:
            modelVars['model'] = models.getModel(mdlParams['model_type'])()
            # Original input size
            #if 'Dense' not in mdlParams['model_type']:
            #    print("Original input size",modelVars['model'].input_size)
            #print(modelVars['model'])
            if 'Dense' in mdlParams['model_type']:
                if mdlParams['input_size'][0] != 224:
                    modelVars['model'] = utils.modify_densenet_avg_pool(modelVars['model'])
                    #print(modelVars['model'])            
                num_ftrs = modelVars['model'].classifier.in_features
                modelVars['model'].classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])
                #print(modelVars['model'])
            elif 'dpn' in mdlParams['model_type']:
                num_ftrs = modelVars['model'].classifier.in_channels
                modelVars['model'].classifier = nn.Conv2d(num_ftrs,mdlParams['numClasses'],[1,1])
                #modelVars['model'].add_module('real_classifier',nn.Linear(num_ftrs, mdlParams['numClasses']))
                #print(modelVars['model'])
            else:
                num_ftrs = modelVars['model'].last_linear.in_features
                modelVars['model'].last_linear = nn.Linear(num_ftrs, mdlParams['numClasses'])  
        modelVars['model']  = modelVars['model'].to(modelVars['device'])
        #summary(modelVars['model'], (mdlParams['input_size'][2], mdlParams['input_size'][0], mdlParams['input_size'][1]))
        # Loss, with class weighting
        # Loss, with class weighting
        if mdlParams['balance_classes'] == 3 or mdlParams['balance_classes'] == 0 or mdlParams['balance_classes'] == 12:
            modelVars['criterion'] = nn.CrossEntropyLoss()
        elif mdlParams['balance_classes'] == 8:
            modelVars['criterion'] = nn.CrossEntropyLoss(reduce=False)
        elif mdlParams['balance_classes'] == 6 or mdlParams['balance_classes'] == 7:
            modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)),reduce=False)
        elif mdlParams['balance_classes'] == 10:
            modelVars['criterion'] = utils.FocalLoss(mdlParams['numClasses'])
        elif mdlParams['balance_classes'] == 11:
            modelVars['criterion'] = utils.FocalLoss(mdlParams['numClasses'],alpha=torch.cuda.FloatTensor(class_weights.astype(np.float32)))
        else:
            modelVars['criterion'] = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(class_weights.astype(np.float32)))

        # Observe that all parameters are being optimized
        modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])

        # Decay LR by a factor of 0.1 every 7 epochs
        modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/np.float32(mdlParams['LRstep']))

        # Define softmax
        modelVars['softmax'] = nn.Softmax(dim=1)

        # Manually find latest chekcpoint, tf.train.latest_checkpoint is doing weird shit
        files = glob(mdlParams['saveDir']+'/*')
        #print(mdlParams['saveDir'])
        #print("Files",files)
        global_steps = np.zeros([len(files)])
        for i in range(len(files)):
            # Use meta files to find the highest index
            if 'checkpoint' not in files[i]:
                continue
            if mdlParams['ckpt_name'] not in files[i]:
                continue
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',files[i])]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found, if first is not wanted
        global_steps = np.sort(global_steps)
        if mdlParams.get('use_first') is not None:
            chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(global_steps[-2])) + '.pt'
        else:
            chkPath = mdlParams['saveDir'] + '/' + mdlParams['ckpt_name'] + str(int(np.max(global_steps))) + '.pt'
        print("Restoring: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # Initialize model and optimizer
        modelVars['model'].load_state_dict(state['state_dict'])
        #modelVars['optimizer'].load_state_dict(state['optimizer'])   
        # Construct pkl filename: config name, last/best, saved epoch number
        pklFileName = sys.argv[2] + "_" + sys.argv[6] + "_" + str(int(np.max(global_steps))) + ".pkl"
        modelVars['model'].eval()
        if mdlParams['classification']:
            print("CV Set ",cv+1)
            print("------------------------------------")
            # Training err first, deactivated
            if 'trainInd' in mdlParams and False:
                loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets = utils.getErrClassification_mgpu(mdlParams, 'trainInd', modelVars)
                print("Training Results:")
                print("----------------------------------")
                print("Loss",np.mean(loss))
                print("F1 Score",f1)            
                print("Sensitivity",sensitivity)
                print("Specificity",specificity)
                print("Accuracy",accuracy)
                print("Per Class Accuracy",waccuracy)
                print("Weighted Accuracy",waccuracy)
                print("AUC",auc)
                print("Mean AUC", np.mean(auc))            
            if 'valInd' in mdlParams and (len(sys.argv) <= 8 or mdlParams['learn_on_preds']):
                if len(sys.argv) > 8:
                    allFiles = sorted(glob(mdlParams['saveDirBase'] + "/*"))
                    found = False
                    for fileName in allFiles:
                        if ".pkl" in fileName and sys.argv[6] in fileName:
                            with open(fileName, 'rb') as f:
                                allData_new = pickle.load(f)  
                            if 'bestPredMC' in allData_new:
                                allData = allData_new
                                print("Val predictions for learning are there, continue to prediction on unlabeled data")
                                found = True
                                break
                    if found:
                        break
                    else:
                        print("No exisiting file with val predictions, evaluating again")
                loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc = utils.getErrClassification_mgpu(mdlParams, 'valInd', modelVars)
                print("Validation Results:")
                print("----------------------------------")
                print("Loss",np.mean(loss))
                print("F1 Score",f1)            
                print("Sensitivity",sensitivity)
                print("Specificity",specificity)
                print("Accuracy",accuracy)
                print("Per Class Accuracy",waccuracy)
                print("Weighted Accuracy",np.mean(waccuracy))
                print("AUC",auc)
                print("Mean AUC", np.mean(auc))  
                # Save results in dict
                if 'testInd' not in mdlParams:
                    allData['f1Best'][cv] = f1
                    allData['sensBest'][cv,:] = sensitivity
                    allData['specBest'][cv,:] = specificity
                    allData['accBest'][cv] = accuracy
                    allData['waccBest'][cv,:] = waccuracy
                    allData['aucBest'][cv,:] = auc  
                allData['bestPred'][cv] = predictions
                allData['bestPredMC'][cv] = predictions_mc
                allData['targets'][cv] = targets 
                print("Pred shape",predictions.shape,"Tar shape",targets.shape)
                # Learn on ordered multi-crop results validation -> validation/test
                if mdlParams['learn_on_preds']:
                    if 'testInd' in mdlParams:
                        loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc = utils.getErrClassification_mgpu(mdlParams, 'testInd', modelVars)
                        print("Test Results Normal:")
                        print("----------------------------------")
                        print("Loss",np.mean(loss))
                        print("F1 Score",f1)            
                        print("Sensitivity",sensitivity)
                        print("Specificity",specificity)
                        print("Accuracy",accuracy)
                        print("Per Class Accuracy",waccuracy)
                        print("Weighted Accuracy",np.mean(waccuracy))
                        print("AUC",auc)
                        print("Mean AUC", np.mean(auc))  
                        # Save results in dict
                        allData['f1Best'][cv] = f1
                        allData['sensBest'][cv,:] = sensitivity
                        allData['specBest'][cv,:] = specificity
                        allData['accBest'][cv] = accuracy
                        allData['waccBest'][cv,:] = waccuracy
                        allData['aucBest'][cv,:] = auc                
                        accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy = utils.learn_on_predictions(mdlParams, modelVars, allData['bestPredMC'][cv], allData['targets'][cv], pred_test=predictions_mc,tar_test=targets)
                    else:
                        accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy = utils.learn_on_predictions(mdlParams, modelVars, allData['bestPredMC'][cv], allData['targets'][cv], split=400)
                    print("Meta Learning:")
                    print(mdlParams['meta_learner'])
                    print("----------------------------------")
                    print("F1 Score",f1)            
                    print("Sensitivity",sensitivity)
                    print("Specificity",specificity)
                    print("Accuracy",accuracy)
                    print("Per Class Accuracy",waccuracy)
                    print("Weighted Accuracy",np.mean(waccuracy))
                    print("AUC",auc)
                    print("Mean AUC", np.mean(auc))  
                    # Save results in dict
                    allData['f1Best_meta'][cv] = f1
                    allData['sensBest_meta'][cv,:] = sensitivity
                    allData['specBest_meta'][cv,:] = specificity
                    allData['accBest_meta'][cv] = accuracy
                    allData['waccBest_meta'][cv,:] = waccuracy
                    allData['aucBest_meta'][cv,:] = auc  
                    allData['bestPred_meta'][cv] = predictions
                    allData['targets_meta'][cv] = targets     

            if 'testInd' in mdlParams:        
                loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, predictions_mc = utils.getErrClassification_mgpu(mdlParams, 'testInd', modelVars)
                print("Test Results Normal:")
                print("----------------------------------")
                print("Loss",np.mean(loss))
                print("F1 Score",f1)            
                print("Sensitivity",sensitivity)
                print("Specificity",specificity)
                print("Accuracy",accuracy)
                print("Per Class Accuracy",waccuracy)
                print("Weighted Accuracy",np.mean(waccuracy))
                print("AUC",auc)
                print("Mean AUC", np.mean(auc))  
                # Save results in dict
                allData['f1Best'][cv] = f1
                allData['sensBest'][cv,:] = sensitivity
                allData['specBest'][cv,:] = specificity
                allData['accBest'][cv] = accuracy
                allData['waccBest'][cv,:] = waccuracy
                allData['aucBest'][cv,:] = auc    
        else:
            # TODO: Regression
            print("Not Implemented")            

# Mean results over all folds
np.set_printoptions(precision=4)
print("-------------------------------------------------")
print("Mean over all Folds")
print("-------------------------------------------------")
print("F1 Score",np.array([np.mean(allData['f1Best'])]),"+-",np.array([np.std(allData['f1Best'])]))       
print("Sensitivtiy",np.mean(allData['sensBest'],0),"+-",np.std(allData['sensBest'],0))  
print("Specificity",np.mean(allData['specBest'],0),"+-",np.std(allData['specBest'],0))  
print("Mean Specificity",np.array([np.mean(allData['specBest'])]),"+-",np.array([np.std(np.mean(allData['specBest'],1))]))  
print("Accuracy",np.array([np.mean(allData['accBest'])]),"+-",np.array([np.std(allData['accBest'])]))  
print("Per Class Accuracy",np.mean(allData['waccBest'],0),"+-",np.std(allData['waccBest'],0))
print("Weighted Accuracy",np.array([np.mean(allData['waccBest'])]),"+-",np.array([np.std(np.mean(allData['waccBest'],1))])) 
print("AUC",np.mean(allData['aucBest'],0),"+-",np.std(allData['aucBest'],0))    
print("Mean AUC",np.array([np.mean(allData['aucBest'])]),"+-",np.array([np.std(np.mean(allData['aucBest'],1))]))      
# Perhaps print results for meta learning
if mdlParams['learn_on_preds']:   
    print("-------------------------------------------------")
    print("Mean over all Folds (meta learning)")
    print("-------------------------------------------------")
    print("F1 Score",np.array([np.mean(allData['f1Best_meta'])]),"+-",np.array([np.std(allData['f1Best_meta'])]))       
    print("Sensitivtiy",np.mean(allData['sensBest_meta'],0),"+-",np.std(allData['sensBest_meta'],0))  
    print("Specificity",np.mean(allData['specBest_meta'],0),"+-",np.std(allData['specBest_meta'],0))  
    print("Accuracy",np.array([np.mean(allData['accBest_meta'])]),"+-",np.array([np.std(allData['accBest_meta'])]))  
    print("Per Class Accuracy",np.mean(allData['waccBest_meta'],0),"+-",np.std(allData['waccBest_meta'],0))
    print("Weighted Accuracy",np.array([np.mean(allData['waccBest_meta'])]),"+-",np.array([np.std(np.mean(allData['waccBest_meta'],1))])) 
    print("AUC",np.mean(allData['aucBest_meta'],0),"+-",np.std(allData['aucBest_meta'],0))    
    print("Mean AUC",np.array([np.mean(allData['aucBest_meta'])]),"+-",np.array([np.std(np.mean(allData['aucBest_meta'],1))]))     
# Save dict with results
with open(mdlParams['saveDirBase'] + "/" + pklFileName, 'wb') as f:
    pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)              
