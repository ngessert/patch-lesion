import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models as tv_models
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np
from scipy import io
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
import sklearn.preprocessing
import utils
from sklearn.utils import class_weight
import models
import custom_models

# add configuration file
# Dictionary for model configuration
mdlParams = {}

# Import machine config
pc_cfg = importlib.import_module('pc_cfgs.'+sys.argv[1])
mdlParams.update(pc_cfg.mdlParams)

# Import model config
model_cfg = importlib.import_module('cfgs.'+sys.argv[2])
mdlParams_model = model_cfg.init(mdlParams)
mdlParams.update(mdlParams_model)

# Set visible devices
cuda_str = ""
for i in range(len(mdlParams['numGPUs'])):
    cuda_str = cuda_str + str(mdlParams['numGPUs'][i])
    if i is not len(mdlParams['numGPUs'])-1:
        cuda_str = cuda_str + ","
print("Devices to use:",cuda_str)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str


# Indicate training
mdlParams['trainSetState'] = 'train'

# Path name from filename
mdlParams['saveDirBase'] = mdlParams['saveDir'] + sys.argv[2]

if len(sys.argv) > 3:
    # Set visible devices
    if 'gpu' in sys.argv[3]:
        mdlParams['numGPUs']= [[int(s) for s in re.findall(r'\d+',sys.argv[3])][-1]]
        cuda_str = ""
        for i in range(len(mdlParams['numGPUs'])):
            cuda_str = cuda_str + str(mdlParams['numGPUs'][i])
            if i is not len(mdlParams['numGPUs'])-1:
                cuda_str = cuda_str + ","
        print("Devices to use:",cuda_str)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_str      

# Dicts for paths
mdlParams['im_list'] = {}
mdlParams['tar_list'] = {}

# Check if there is a validation set, if not, evaluate train error instead
if 'valIndCV' in mdlParams or 'valInd' in mdlParams:
    eval_set = 'valInd'
    print("Evaluating on validation set during training.")
else:
    eval_set = 'trainInd'
    print("No validation set, evaluating on training set during training.")

# Scaler, scales targets to a range of 0-1
if mdlParams['scale_targets']:
    mdlParams['scaler'] = sklearn.preprocessing.MinMaxScaler().fit(mdlParams['targets'][mdlParams['trainInd'],:][:,mdlParams['tar_range'].astype(int)])

# Check if there were previous ones that have alreary bin learned
prevFile = Path(mdlParams['saveDirBase'] + '/CV.pkl')
#print(prevFile)
if prevFile.exists():
    print("Part of CV already done")
    with open(mdlParams['saveDirBase'] + '/CV.pkl', 'rb') as f:
        allData = pickle.load(f)
else:
    allData = {}
    allData['f1Best'] = {}
    allData['sensBest'] = {}
    allData['specBest'] = {}
    allData['accBest'] = {}
    allData['waccBest'] = {}
    allData['aucBest'] = {}
    allData['convergeTime'] = {}
    allData['bestPred'] = {}
    allData['targets'] = {}
 
# Take care of CV
for cv in range(mdlParams['numCV']):  
    # Check if this fold was already trained
    if cv in allData['f1Best']:
        print('Fold ' + str(cv) + ' already trained.')
        continue        
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
    # Create basepath if it doesnt exist yet
    if not os.path.isdir(mdlParams['saveDirBase']):
        os.mkdir(mdlParams['saveDirBase'])
    # Check if there is something to load
    load_old = 0
    if os.path.isdir(mdlParams['saveDir']):
        # Check if a checkpoint is in there
        if len([name for name in os.listdir(mdlParams['saveDir'])]) > 0:
            load_old = 1
            print("Loading old model")
        else:
            # Delete whatever is in there (nothing happens)
            filelist = [os.remove(mdlParams['saveDir'] +'/'+f) for f in os.listdir(mdlParams['saveDir'])]
    else:
        os.mkdir(mdlParams['saveDir'])
    # Save training progress in here
    save_dict = {}
    save_dict['acc'] = []
    save_dict['loss'] = []
    save_dict['wacc'] = []
    save_dict['auc'] = []
    save_dict['sens'] = []
    save_dict['spec'] = []
    save_dict['f1'] = []
    save_dict['step_num'] = []
    if mdlParams['print_trainerr']:
        save_dict_train = {}
        save_dict_train['acc'] = []
        save_dict_train['loss'] = []
        save_dict_train['wacc'] = []
        save_dict_train['auc'] = []
        save_dict_train['sens'] = []
        save_dict_train['spec'] = []
        save_dict_train['f1'] = []
        save_dict_train['step_num'] = []        
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
        if isinstance(mdlParams['extra_fac'], float):
            class_weights = np.power(class_weights,mdlParams['extra_fac'])
        else:
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

    if mdlParams['balance_classes'] == 12 or mdlParams['balance_classes'] == 13:
        #print(np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1).size(0))
        strat_sampler = utils.StratifiedSampler(mdlParams)
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], sampler=strat_sampler, num_workers=8, pin_memory=True) 
    else:
        modelVars['dataloader_trainInd'] = DataLoader(dataset_train, batch_size=mdlParams['batchSize'], shuffle=True, num_workers=8, pin_memory=True) 
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
    # multi gpu support
    if len(mdlParams['numGPUs']) > 1:
        modelVars['model'] = nn.DataParallel(modelVars['model'])
    #modelVars['model']  = modelVars['model'].to(modelVars['device'])  
    modelVars['model'] = modelVars['model'].cuda()
    #summary(modelVars['model'], modelVars['model'].input_size)# (mdlParams['input_size'][2], mdlParams['input_size'][0], mdlParams['input_size'][1]))
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

    if 'convGRU' in mdlParams['model_type']:
        modelVars['optimizer'] = optim.Adam([{'params':modelVars['model'].cnn.parameters()},{'params':modelVars['model'].convgru.parameters(),'lr':mdlParams['learning_rate_congru']}], lr=mdlParams['learning_rate'])
    elif 'CNN_GRU_TP' in mdlParams['model_type']:
        modelVars['optimizer'] = optim.Adam([{'params':modelVars['model'].cnn_features.parameters()},{'params':modelVars['model'].classifier.parameters(),'lr':mdlParams['learning_rate_congru']},{'params':modelVars['model'].cnn_features_global.parameters()},{'params':modelVars['model'].gru.parameters(),'lr':mdlParams['learning_rate_congru']}], lr=mdlParams['learning_rate'])            
    elif 'CNN_GRU' in mdlParams['model_type']:
        if mdlParams['aux_classifier']:
            modelVars['optimizer'] = optim.Adam([{'params':modelVars['model'].cnn_features.parameters()},{'params':modelVars['model'].cnn.classifier.parameters(),'lr':mdlParams['learning_rate_congru']},{'params':modelVars['model'].classifier.parameters(),'lr':mdlParams['learning_rate_congru']},{'params':modelVars['model'].gru.parameters(),'lr':mdlParams['learning_rate_congru']}], lr=mdlParams['learning_rate']) 
        else:
            modelVars['optimizer'] = optim.Adam([{'params':modelVars['model'].cnn_features.parameters()},{'params':modelVars['model'].classifier.parameters(),'lr':mdlParams['learning_rate_congru']},{'params':modelVars['model'].gru.parameters(),'lr':mdlParams['learning_rate_congru']}], lr=mdlParams['learning_rate'])            
    elif 'CNN_FC' in mdlParams['model_type']:
        if mdlParams['combine_features'] == 'conv1':
            modelVars['optimizer'] = optim.Adam([{'params':modelVars['model'].cnn_features.parameters()},{'params':modelVars['model'].conv1x1.parameters(),'lr':mdlParams['learning_rate_congru']},{'params':modelVars['model'].classifier.parameters(),'lr':mdlParams['learning_rate_congru']}], lr=mdlParams['learning_rate'])            
        else:
            if mdlParams['initial_attention'] and not mdlParams['end_attention']:
                modelVars['optimizer'] = optim.Adam([{'params':modelVars['model'].cnn_features.parameters()},{'params':modelVars['model'].classifier.parameters(),'lr':mdlParams['learning_rate_congru']},{'params':modelVars['model'].initial_attention_layer.parameters(),'lr':mdlParams['learning_rate_congru']}], lr=mdlParams['learning_rate'])
            elif not mdlParams['initial_attention'] and mdlParams['end_attention']:
                modelVars['optimizer'] = optim.Adam([{'params':modelVars['model'].cnn_features.parameters()},{'params':modelVars['model'].classifier.parameters(),'lr':mdlParams['learning_rate_congru']},{'params':modelVars['model'].end_attention_layer.parameters(),'lr':mdlParams['learning_rate_congru']}], lr=mdlParams['learning_rate'])
            elif mdlParams['initial_attention'] and mdlParams['end_attention']:
                modelVars['optimizer'] = optim.Adam([{'params':modelVars['model'].cnn_features.parameters()},{'params':modelVars['model'].classifier.parameters(),'lr':mdlParams['learning_rate_congru']},{'params':modelVars['model'].initial_attention_layer.parameters(),'lr':mdlParams['learning_rate_congru']},{'params':modelVars['model'].end_attention_layer.parameters(),'lr':mdlParams['learning_rate_congru']}], lr=mdlParams['learning_rate'])
            else:
                modelVars['optimizer'] = optim.Adam([{'params':modelVars['model'].cnn_features.parameters()},{'params':modelVars['model'].classifier.parameters(),'lr':mdlParams['learning_rate_congru']}], lr=mdlParams['learning_rate'])            
    else:
        modelVars['optimizer'] = optim.Adam(modelVars['model'].parameters(), lr=mdlParams['learning_rate'])

    # Decay LR by a factor of 0.1 every 7 epochs
    modelVars['scheduler'] = lr_scheduler.StepLR(modelVars['optimizer'], step_size=mdlParams['lowerLRAfter'], gamma=1/np.float32(mdlParams['LRstep']))

    # Define softmax
    modelVars['softmax'] = nn.Softmax(dim=1)

    # Set up training
    # loading from checkpoint
    if load_old:
        # Find last, not last best checkpoint
        files = glob(mdlParams['saveDir']+'/*')
        global_steps = np.zeros([len(files)])
        for i in range(len(files)):
            # Use meta files to find the highest index
            if 'best' in files[i]:
                continue
            if 'checkpoint-' not in files[i]:
                continue                
            # Extract global step
            nums = [int(s) for s in re.findall(r'\d+',files[i])]
            global_steps[i] = nums[-1]
        # Create path with maximum global step found
        chkPath = mdlParams['saveDir'] + '/checkpoint-' + str(int(np.max(global_steps))) + '.pt'
        print("Restoring: ",chkPath)
        # Load
        state = torch.load(chkPath)
        # Initialize model and optimizer
        modelVars['model'].load_state_dict(state['state_dict'])
        modelVars['optimizer'].load_state_dict(state['optimizer'])     
        start_epoch = state['epoch']
        mdlParams['lastBestInd'] = int(np.max(global_steps))
    else:
        start_epoch = 1
        mdlParams['lastBestInd'] = -1

    # Num batches
    numBatchesTrain = int(math.floor(len(mdlParams['trainInd'])/mdlParams['batchSize']))
    print("Train batches",numBatchesTrain)

    # Track metrics for saving best model
    mdlParams['valBest'] = 1000

    # Start training       
    # TODO: write away stuff ? summaries

    # Run training
    start_time = time.time()
    print("Start training...")
    for step in range(start_epoch, mdlParams['training_steps']+1):
        # One Epoch of training
        if step >= mdlParams['lowerLRat']-mdlParams['lowerLRAfter']:
            modelVars['scheduler'].step()
        modelVars['model'].train()      
        for j, (inputs, labels, indices) in enumerate(modelVars['dataloader_trainInd']):    
            #print(indices)                  
            # Run optimization         
            inputs = inputs.cuda()
            labels = labels.cuda()        
            # zero the parameter gradients
            modelVars['optimizer'].zero_grad()             
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):             
                if 'hpc' in sys.argv[1]:
                    print("Epoch",step,"iteration",j) 
                if mdlParams['aux_classifier']:
                    outputs, outputs_aux = modelVars['model'](inputs) 
                    loss1 = modelVars['criterion'](outputs, labels)
                    labels_aux = labels.repeat(mdlParams['multiCropTrain'])
                    loss2 = modelVars['criterion'](outputs_aux, labels_aux) 
                    loss = loss1 + mdlParams['aux_classifier_loss_fac']*loss2     
                else:                     
                    outputs = modelVars['model'](inputs)            
                    loss = modelVars['criterion'](outputs, labels)         
                # Perhaps adjust weighting of the loss by the specific index
                if mdlParams['balance_classes'] == 6 or mdlParams['balance_classes'] == 7 or mdlParams['balance_classes'] == 8:
                    #loss = loss.cpu()
                    indices = indices.numpy()
                    loss = loss*torch.cuda.FloatTensor(mdlParams['loss_fac_per_example'][indices].astype(np.float32))
                    loss = torch.mean(loss)
                    #loss = loss.cuda()
                # backward + optimize only if in training phase
                loss.backward()                 
                modelVars['optimizer'].step()                                 
        if step % mdlParams['display_step'] == 0 or step == 1:
            # Duration so far
            duration = time.time() - start_time
            # Calculate evaluation metrics
            if mdlParams['classification']:
                # Adjust model state
                modelVars['model'].eval()
                # Get metrics
                loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, _ = utils.getErrClassification_mgpu(mdlParams, eval_set, modelVars)
                # Save in mat
                save_dict['loss'].append(loss)
                save_dict['acc'].append(accuracy)
                save_dict['wacc'].append(waccuracy)
                save_dict['auc'].append(auc)
                save_dict['sens'].append(sensitivity)
                save_dict['spec'].append(specificity)
                save_dict['f1'].append(f1)
                save_dict['step_num'].append(step)
                if os.path.isfile(mdlParams['saveDir'] + '/progression_'+eval_set+'.mat'):
                    os.remove(mdlParams['saveDir'] + '/progression_'+eval_set+'.mat')                
                io.savemat(mdlParams['saveDir'] + '/progression_'+eval_set+'.mat',save_dict)                
            eval_metric = -np.mean(waccuracy)
            # Check if we have a new best value
            if eval_metric < mdlParams['valBest']:
                mdlParams['valBest'] = eval_metric
                if mdlParams['classification']:
                    allData['f1Best'][cv] = f1
                    allData['sensBest'][cv] = sensitivity
                    allData['specBest'][cv] = specificity
                    allData['accBest'][cv] = accuracy
                    allData['waccBest'][cv] = waccuracy
                    allData['aucBest'][cv] = auc
                oldBestInd = mdlParams['lastBestInd']
                mdlParams['lastBestInd'] = step
                allData['convergeTime'][cv] = step
                # Save best predictions
                allData['bestPred'][cv] = predictions
                allData['targets'][cv] = targets
                # Delte previously best model
                if os.path.isfile(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.pt'):
                    os.remove(mdlParams['saveDir'] + '/checkpoint_best-' + str(oldBestInd) + '.pt')
                # Save currently best model
                state = {'epoch': step,'state_dict': modelVars['model'].state_dict(),'optimizer': modelVars['optimizer'].state_dict()}
                torch.save(state, mdlParams['saveDir'] + '/checkpoint_best-' + str(step) + '.pt')               
                            
            # If its not better, just save it delete the last checkpoint if it is not current best one
            # Save current model
            state = {'epoch': step,'state_dict': modelVars['model'].state_dict(),'optimizer': modelVars['optimizer'].state_dict()}
            torch.save(state, mdlParams['saveDir'] + '/checkpoint-' + str(step) + '.pt')                           
            # Delete last one
            if step == mdlParams['display_step']:
                lastInd = 1
            else:
                lastInd = step-mdlParams['display_step']
            if os.path.isfile(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.pt'):
                os.remove(mdlParams['saveDir'] + '/checkpoint-' + str(lastInd) + '.pt')                 
            # Print
            if mdlParams['classification']:
                print("\n")
                print('Fold: %d Epoch: %d/%d (%d h %d m %d s)' % (cv,step,mdlParams['training_steps'], int(duration/3600), int(np.mod(duration,3600)/60), int(np.mod(np.mod(duration,3600),60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
                print("Loss on ",eval_set,"set: ",loss," Accuracy: ",accuracy," F1: ",f1," (best WACC: ",-mdlParams['valBest']," at Epoch ",mdlParams['lastBestInd'],")")
                print("Auc",auc,"Mean AUC",np.mean(auc))
                print("Per Class Acc",waccuracy,"Weighted Accuracy",np.mean(waccuracy))
                print("Sensitivity: ",sensitivity,"Specificity",specificity)
                print("Confusion Matrix")
                print(conf_matrix)
                # Potentially peek at test error
                if mdlParams['peak_at_testerr']:              
                    loss, accuracy, sensitivity, specificity, _, f1, _, _, _, _, _ = utils.getErrClassification_mgpu(mdlParams, 'testInd', modelVars)
                    print("Test loss: ",loss," Accuracy: ",accuracy," F1: ",f1)
                    print("Sensitivity: ",sensitivity,"Specificity",specificity)
                # Potentially print train err
                if mdlParams['print_trainerr'] and 'train' not in eval_set:                
                    loss, accuracy, sensitivity, specificity, conf_matrix, f1, auc, waccuracy, predictions, targets, _ = utils.getErrClassification_mgpu(mdlParams, 'trainInd', modelVars)
                    # Save in mat
                    save_dict_train['loss'].append(loss)
                    save_dict_train['acc'].append(accuracy)
                    save_dict_train['wacc'].append(waccuracy)
                    save_dict_train['auc'].append(auc)
                    save_dict_train['sens'].append(sensitivity)
                    save_dict_train['spec'].append(specificity)
                    save_dict_train['f1'].append(f1)
                    save_dict_train['step_num'].append(step)
                    if os.path.isfile(mdlParams['saveDir'] + '/progression_trainInd.mat'):
                        os.remove(mdlParams['saveDir'] + '/progression_trainInd.mat')                
                    scipy.io.savemat(mdlParams['saveDir'] + '/progression_trainInd.mat',save_dict_train)                     
                    print("Train loss: ",loss," Accuracy: ",accuracy," F1: ",f1)
                    print("Sensitivity: ",sensitivity,"Specificity",specificity)
    # Free everything in modelvars
    modelVars.clear()
    # After CV Training: print CV results and save them
    print("Best F1:",allData['f1Best'][cv])
    print("Best Sens:",allData['sensBest'][cv])
    print("Best Spec:",allData['specBest'][cv])
    print("Best Acc:",allData['accBest'][cv])
    print("Best Per Class Accuracy:",allData['waccBest'][cv])
    print("Best Weighted Acc:",np.mean(allData['waccBest'][cv]))
    print("Best AUC:",allData['aucBest'][cv])
    print("Best Mean AUC:",np.mean(allData['aucBest'][cv]))    
    print("Convergence Steps:",allData['convergeTime'][cv])

    # Write to File
    with open(mdlParams['saveDirBase'] + '/CV.pkl', 'wb') as f:
        pickle.dump(allData, f, pickle.HIGHEST_PROTOCOL)           
            