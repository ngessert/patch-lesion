import os
import torch
#import pandas as pd
from skimage import io, transform
import scipy
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, utils
from sklearn.metrics import confusion_matrix, auc, roc_curve, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import math
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import types


# Define ISIC Dataset Class
class ISICDataset(Dataset):
    """ISIC dataset."""

    def __init__(self, mdlParams, indSet):
        """
        Args:
            mdlParams (dict): Configuration for loading
            indSet (string): Indicates train, val, test
        """
        # Number of classes
        self.numClasses = mdlParams['numClasses']
        # Model input size
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))    
        # Load size/downsampled size, reversed order for loading
        self.input_size_load = (np.int32(mdlParams['input_size_load'][1]),np.int32(mdlParams['input_size_load'][0]))   
        # Whether or not to use ordered cropping 
        self.orderedCrop = mdlParams['orderedCrop']   
        # Number of crops for multi crop eval
        self.multiCropEval = mdlParams['multiCropEval']   
        # Whether during training same-sized crops should be used
        self.same_sized_crop = mdlParams['same_sized_crops']    
        # Only downsample
        self.only_downsmaple = mdlParams.get('only_downsmaple',False)   
        # Potential class balancing option 
        self.balancing = mdlParams['balance_classes']
        # Whether data should be preloaded
        self.preload = mdlParams['preload']
        # Potentially subtract a mean
        self.subtract_set_mean = mdlParams['subtract_set_mean']
        # Potential switch for evaluation on the training set
        self.train_eval_state = mdlParams['trainSetState']   
        # Potential setMean to deduce from channels
        self.setMean = mdlParams['setMean'].astype(np.float32)
        # Current indSet = 'trainInd'/'valInd'/'testInd'
        self.indices = mdlParams[indSet]  
        self.indSet = indSet
        if self.balancing == 3 and indSet == 'trainInd':
            # Sample classes equally for each batch
            # First, split set by classes
            not_one_hot = np.argmax(mdlParams['labels_array'],1)
            self.class_indices = []
            for i in range(mdlParams['numClasses']):
                self.class_indices.append(np.where(not_one_hot==i)[0])
                # Kick out non-trainind indices
                self.class_indices[i] = np.setdiff1d(self.class_indices[i],mdlParams['valInd'])
                # And test indices
                if 'testInd' in mdlParams:
                    self.class_indices[i] = np.setdiff1d(self.class_indices[i],mdlParams['testInd'])
            # Now sample indices equally for each batch by repeating all of them to have the same amount as the max number
            indices = []
            max_num = np.max([len(x) for x in self.class_indices])
            # Go thourgh all classes
            for i in range(mdlParams['numClasses']):
                count = 0
                class_count = 0
                max_num_curr_class = len(self.class_indices[i])
                # Add examples until we reach the maximum
                while(count < max_num):
                    # Start at the beginning, if we are through all available examples
                    if class_count == max_num_curr_class:
                        class_count = 0
                    indices.append(self.class_indices[i][class_count])
                    count += 1
                    class_count += 1
            print("Largest class",max_num,"Indices len",len(indices))
            print("Intersect val",np.intersect1d(indices,mdlParams['valInd']),"Intersect Testind",np.intersect1d(indices,mdlParams['testInd']))
            # Set labels/inputs
            self.labels = mdlParams['labels_array'][indices,:]
            self.im_paths = np.array(mdlParams['im_paths'])[indices].tolist()     
            # Normal train proc
            if self.same_sized_crop:
                cropping = transforms.RandomCrop(self.input_size)
            elif self.only_downsmaple:
                cropping = transforms.Resize(self.input_size)
            else:
                cropping = transforms.RandomResizedCrop(self.input_size[0])
            # All transforms
            self.composed = transforms.Compose([
                    cropping,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                    ])                                
        elif self.orderedCrop and (indSet == 'valInd' or self.train_eval_state  == 'eval' or indSet == 'testInd'):
            # Complete labels array, only for current indSet, repeat for multiordercrop
            inds_rep = np.repeat(mdlParams[indSet], mdlParams['multiCropEval'])
            self.labels = mdlParams['labels_array'][inds_rep,:]
            # Path to images for loading, only for current indSet, repeat for multiordercrop
            self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
            print(len(self.im_paths))
            # Set up crop positions for every sample
            self.cropPositions = np.tile(mdlParams['cropPositions'], (mdlParams[indSet].shape[0],1))
            print("CP",self.cropPositions.shape)
            #print("CP Example",self.cropPositions[0:len(mdlParams['cropPositions']),:])          
            # Set up transforms
            self.norm = transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
            self.trans = transforms.ToTensor()
        elif indSet == 'valInd' or indSet == 'testInd':
            if self.multiCropEval == 0:
                if self.only_downsmaple:
                    self.cropping = transforms.Resize(self.input_size)
                else:
                    self.cropping = transforms.Compose([transforms.CenterCrop(np.int32(self.input_size[0]*1.5)),transforms.Resize(self.input_size)])
                # Complete labels array, only for current indSet
                self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
                # Path to images for loading, only for current indSet
                self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()                   
            else:
                self.cropping = transforms.RandomResizedCrop(self.input_size[0])
                # Complete labels array, only for current indSet, repeat for multiordercrop
                inds_rep = np.repeat(mdlParams[indSet], mdlParams['multiCropEval'])
                self.labels = mdlParams['labels_array'][inds_rep,:]
                # Path to images for loading, only for current indSet, repeat for multiordercrop
                self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
            print(len(self.im_paths))  
            # Set up transforms
            self.norm = transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
            self.trans = transforms.ToTensor()                   
        else:
            # Normal train proc
            if self.same_sized_crop:
                cropping = transforms.RandomCrop(self.input_size)
            elif self.only_downsmaple:
                cropping = transforms.Resize(self.input_size)
            else:
                cropping = transforms.RandomResizedCrop(self.input_size[0])
            # Color distortion
            if mdlParams.get('full_color_distort') is not None:
                color_distort = transforms.ColorJitter(brightness=32. / 255.,saturation=0.5, contrast = 0.5, hue = 0.2) 
            else:
                color_distort = transforms.ColorJitter(brightness=32. / 255.,saturation=0.5) 
            # All transforms
            self.composed = transforms.Compose([
                    cropping,
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    color_distort,
                    transforms.ToTensor(),
                    transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                    ])                  
            # Complete labels array, only for current indSet
            self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
            # Path to images for loading, only for current indSet
            self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()
        # Potentially preload
        if self.preload:
            self.im_list = []
            for i in range(len(self.im_paths)):
                if self.input_size_load[0] < 450:
                    self.im_list.append(Image.open(self.im_paths[i]).resize(self.input_size_load,Image.BICUBIC))
                else:
                    self.im_list.append(Image.open(self.im_paths[i]))
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # Load image
        if self.preload:
            x = self.im_list[idx]
        else:
            if self.input_size_load[0] < 450:
                x = Image.open(self.im_paths[idx]).resize(self.input_size_load,Image.BICUBIC)
            else:
                x = Image.open(self.im_paths[idx])
        # Get label
        y = self.labels[idx,:]
        # Transform data based on whether train or not train. If train, also check if its train train or train inference
        if self.orderedCrop and (self.indSet == 'valInd' or self.indSet == 'testInd' or self.train_eval_state == 'eval'):
            # Apply ordered cropping to validation or test set
            # First, to pytorch tensor (0.0-1.0)
            x = self.trans(x)
            # Normalize
            x = self.norm(x)
            # Get current crop position
            x_loc = self.cropPositions[idx,0]
            y_loc = self.cropPositions[idx,1]
            # Then, apply current crop
            #print("Before",x.size(),"xloc",x_loc,"y_loc",y_loc)
            #print((x_loc-np.int32(self.input_size[0]/2.)),(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)),(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1])
            x = x[:,(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1]]
            #print("After",x.size())
        elif self.indSet == 'valInd' or self.indSet == 'testInd':
            # Normalize
            x = self.cropping(x)        
            # First, to pytorch tensor (0.0-1.0)
            x = self.trans(x)
            x = self.norm(x)                
        else:
            # Apply
            x = self.composed(x)  
        # Transform y
        y = np.argmax(y)
        y = np.int64(y)
        return x, y, idx

# Sampler for balanced sampling
class StratifiedSampler(torch.utils.data.sampler.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, mdlParams):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.dataset_len = len(mdlParams['trainInd'])
        self.numClasses = mdlParams['numClasses']
        self.trainInd = mdlParams['trainInd']
        # Sample classes equally for each batch
        # First, split set by classes
        not_one_hot = np.argmax(mdlParams['labels_array'][mdlParams['trainInd'],:],1)
        self.class_indices = []
        for i in range(mdlParams['numClasses']):
            self.class_indices.append(np.where(not_one_hot==i)[0])
        self.current_class_ind = 0
        self.current_in_class_ind = np.zeros([mdlParams['numClasses']],dtype=int)

    def gen_sample_array(self):
        # Shuffle all classes first
        for i in range(self.numClasses):
            np.random.shuffle(self.class_indices[i])
        # Construct indset
        indices = np.zeros([self.dataset_len],dtype=np.int32)
        ind = 0
        while(ind < self.dataset_len):
            indices[ind] = self.class_indices[self.current_class_ind][self.current_in_class_ind[self.current_class_ind]]
            # Take care of in-class index
            if self.current_in_class_ind[self.current_class_ind] == len(self.class_indices[self.current_class_ind])-1:
                self.current_in_class_ind[self.current_class_ind] = 0
                # Shuffle
                np.random.shuffle(self.class_indices[self.current_class_ind])
            else:
                self.current_in_class_ind[self.current_class_ind] += 1
            # Take care of overall class ind
            if self.current_class_ind == self.numClasses-1:
                self.current_class_ind = 0
            else:
                self.current_class_ind += 1
            ind += 1
        return indices

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.dataset_len 

# Define ISIC Dataset Class
class ISICDataset_custom(Dataset):
    """ISIC dataset for custom models."""

    def __init__(self, mdlParams, indSet):
        """
        Args:
            mdlParams (dict): Configuration for loading
            indSet (string): Indicates train, val, test
        """
        # True/False on ordered cropping for eval
        # Model input size
        self.input_size = (np.int32(mdlParams['input_size'][0]),np.int32(mdlParams['input_size'][1]))    
        # Load size/downsampled size, reversed order for loading
        self.input_size_load = (np.int32(mdlParams['input_size_load'][1]),np.int32(mdlParams['input_size_load'][0]))   
        # Whether or not to use ordered cropping for validation
        self.orderedCrop = mdlParams['orderedCrop']   
        # Whether or not to use ordered cropping for validation
        self.orderedCropTrain = mdlParams['orderedCropTrain']       
        # Potential patch dropout
        self.patch_dropout = mdlParams.get('patch_dropout',0)  
        # For the global path with CNN_GRU
        self.with_global_path = mdlParams['with_global_path']
        # Number of crops for multi crop eval
        self.multiCropEval = mdlParams['multiCropEval']   
        # How many times should this be repeated for each?
        self.numRandValSeq = mdlParams['numRandValSeq']
        # Number of crops for multi crop train
        self.multiCropTrain = mdlParams['multiCropTrain']           
        # Whether during training same-sized crops should be used
        self.same_sized_crop = mdlParams['same_sized_crops']          
        # Potential class balancing option 
        self.balancing = mdlParams['balance_classes']
        # Whether data should be preloaded
        self.preload = mdlParams['preload']
        # Potentially subtract a mean
        self.subtract_set_mean = mdlParams['subtract_set_mean']
        # Potential switch for evaluation on the training set
        self.train_eval_state = mdlParams['trainSetState']   
        # Potential setMean to deduce from channels
        self.setMean = mdlParams['setMean'].astype(np.float32)
        # Randomly permute time sequence
        self.randPerm = mdlParams['randPerm']
        # Current indSet = 'trainInd'/'valInd'/'testInd'
        self.indices = mdlParams[indSet]  
        self.indSet = indSet
        if self.orderedCrop and (indSet == 'valInd' or self.train_eval_state  == 'eval' or indSet == 'testInd'):
            if self.numRandValSeq > 0:
                # Complete labels array, only for current indSet, repeat for multiordercrop
                inds_rep = np.repeat(mdlParams[indSet], self.numRandValSeq)
                self.labels = mdlParams['labels_array'][inds_rep,:]
                # Path to images for loading, only for current indSet, repeat for multiordercrop
                self.im_paths = np.array(mdlParams['im_paths'])[inds_rep].tolist()
                print(len(self.im_paths))
                # Set up crop positions for every sample
                self.cropPositions = mdlParams['cropPositions']       
                # Set up transforms
                self.norm = transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                self.trans = transforms.ToTensor()       
            else:     
                # Cropping is done in fwd pass
                self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
                # Path to images for loading, only for current indSet, repeat for multiordercrop
                self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()
                # Crop positions
                self.cropPositions = mdlParams['cropPositions']     
                # Set up transforms
                self.norm = transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                self.trans = transforms.ToTensor()
        elif indSet == 'valInd' or indSet == 'testInd':
            if self.multiCropEval == 0:
                self.cropping = transforms.Compose([transforms.CenterCrop(np.int32(self.input_size[0]*2)),transforms.Resize(self.input_size)])
                # Complete labels array, only for current indSet
                self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
                # Path to images for loading, only for current indSet
                self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()                   
            else:
                self.cropping = transforms.RandomCrop(self.input_size)
                # Cropping is done in fwd pass
                self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
                # Path to images for loading, only for current indSet, repeat for multiordercrop
                self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()
            print(len(self.im_paths))  
            # Set up transforms
            self.norm = transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
            self.trans = transforms.ToTensor()                   
        else:
            # Normal train proc
            # Crop positions for ordered cropping
            self.cropPositions = mdlParams['cropPositionsTrain']                 
            # Color distortion
            if mdlParams.get('full_color_distort') is not None:
                color_distort = transforms.ColorJitter(brightness=32. / 255.,saturation=0.5, contrast = 0.5, hue = 0.2) 
            else:
                color_distort = transforms.ColorJitter(brightness=32. / 255.,saturation=0.5) 
            if self.orderedCropTrain:
                # All transforms, before cropping
                self.composed = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        color_distort,
                        transforms.ToTensor(),
                        transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                        ])      
            else:
                # Part before cropping
                self.composed_before = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        color_distort
                        ])
                self.composed_after = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(torch.from_numpy(self.setMean).float(),torch.from_numpy(np.array([1.,1.,1.])).float())
                        ])                  
                self.cropping = transforms.RandomCrop(self.input_size)                
            
            # Complete labels array, only for current indSet
            self.labels = mdlParams['labels_array'][mdlParams[indSet],:]
            # Path to images for loading, only for current indSet
            self.im_paths = np.array(mdlParams['im_paths'])[mdlParams[indSet]].tolist()
        # Potentially preload
        if self.preload:
            self.im_list = []
            for i in range(len(self.im_paths)):
                if self.input_size_load[0] < 450:
                    self.im_list.append(Image.open(self.im_paths[i]).resize(self.input_size_load,Image.BICUBIC))
                else:
                    self.im_list.append(Image.open(self.im_paths[i]))
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # Load image
        if self.preload:
            x = self.im_list[idx]
        else:
            if self.input_size_load[0] < 450:
                x = Image.open(self.im_paths[idx]).resize(self.input_size_load,Image.BICUBIC)
            else:
                x = Image.open(self.im_paths[idx])
        # Get label
        y = self.labels[idx,:]
        # Transform data based on whether train or not train. If train, also check if its train train or train inference
        if self.orderedCrop and (self.indSet == 'valInd' or self.indSet == 'testInd' or self.train_eval_state == 'eval'):
            # Apply ordered cropping to validation or test set
            # Sequence for eval
            if self.with_global_path:
                x_seq = torch.zeros([self.multiCropEval+1,3,self.input_size[0],self.input_size[0]])
                x_seq[-1,:,:,:] = self.norm(self.trans(x.resize(self.input_size,Image.BICUBIC)))
            else:
                x_seq = torch.zeros([self.multiCropEval,3,self.input_size[0],self.input_size[0]])
            x = self.trans(x)
            # Normalize
            x = self.norm(x)
            # Permute sequence?
            if self.numRandValSeq > 0:
                crop_indices = np.random.permutation(self.multiCropEval)
            else:
                crop_indices = range(self.multiCropEval)            
            for j, i in enumerate(crop_indices): 
                # Get current crop position
                x_loc = self.cropPositions[i,0]
                y_loc = self.cropPositions[i,1]                           
                x_seq[j,:,:,:] = x[:,(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1]]
            #print("After",x.size())
        elif self.indSet == 'valInd' or self.indSet == 'testInd':
            # Random cropped
            # Sequence for eval
            x_seq = torch.zeros([self.multiCropTrain,3,self.input_size[0],self.input_size[0]])
            for i in range(self.multiCropTrain): 
                x_seq[i,:,:,:] = self.cropping(x)        
            # Normalize
            x = self.trans(x)
            x = self.norm(x)                
        else:
            # Training
            if self.orderedCropTrain:
                # Sequence for training
                if self.with_global_path:
                    x_seq = torch.zeros([self.multiCropTrain+1,3,self.input_size[0],self.input_size[0]])
                    x_seq[-1,:,:,:] = self.composed(x.resize(self.input_size,Image.BICUBIC))
                else:
                    x_seq = torch.zeros([self.multiCropTrain,3,self.input_size[0],self.input_size[0]])                
                # Preprocessing, applied before cropping
                x = self.composed(x)  
                # Permute squence?
                if self.randPerm:
                    crop_indices = np.random.permutation(self.multiCropTrain)
                else:
                    crop_indices = range(self.multiCropTrain)
                # Generate crops
                for j, i in enumerate(crop_indices):
                    # Get current crop position
                    # Patch dropout
                    if self.patch_dropout > 0:
                        if np.random.rand() < self.patch_dropout:
                            continue
                    x_loc = self.cropPositions[i,0]
                    y_loc = self.cropPositions[i,1]                    
                    x_seq[j,:,:,:] = x[:,(x_loc-np.int32(self.input_size[0]/2.)):(x_loc-np.int32(self.input_size[0]/2.))+self.input_size[0],(y_loc-np.int32(self.input_size[1]/2.)):(y_loc-np.int32(self.input_size[1]/2.))+self.input_size[1]]
            else:
                # Sequence for training
                x_seq = torch.zeros([self.multiCropTrain,3,self.input_size[0],self.input_size[0]])
                # Part of preprocessing: before
                x = self.composed_before(x)
                for i in range(self.multiCropTrain):                
                    x_seq[i,:,:,:] = self.composed_after(self.cropping(x))

                    
        # Transform y
        y = np.argmax(y)
        y = np.int64(y)
        return x_seq, y, idx        

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        #print(N)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def getErrClassification_mgpu(mdlParams, indices, modelVars, exclude_class=None):
    # Set up sizes
    if indices == 'trainInd':
        numBatches = int(math.floor(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
    else:
        numBatches = int(math.ceil(len(mdlParams[indices])/mdlParams['batchSize']/len(mdlParams['numGPUs'])))
    # Consider multi-crop case
    if 'multiCropEval' in mdlParams and mdlParams['multiCropEval'] > 0 and mdlParams.get('model_type_cnn') is None:
        loss_all = np.zeros([numBatches])
        allInds = np.zeros([len(mdlParams[indices])])
        predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
        targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])        
        loss_mc = np.zeros([len(mdlParams[indices])])
        predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])
        targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['multiCropEval']])   
        for i, (inputs, labels, inds) in enumerate(modelVars['dataloader_'+indices]):
            # Get data
            inputs = inputs.to(modelVars['device'])
            labels = labels.to(modelVars['device'])       
            # Not sure if thats necessary
            modelVars['optimizer'].zero_grad()    
            with torch.set_grad_enabled(False):
                # Get outputs
                if mdlParams['aux_classifier']:
                    outputs, outputs_aux = modelVars['model'](inputs)
                    if mdlParams['eval_aux_classifier']:
                        outputs = outputs_aux
                else:
                    outputs = modelVars['model'](inputs)
                preds = modelVars['softmax'](outputs)      
                # Loss
                loss = modelVars['criterion'](outputs, labels)           
            # Write into proper arrays
            loss_mc[i] = np.mean(loss.cpu().numpy())
            predictions_mc[i,:,:] = np.transpose(preds.cpu().numpy())
            tar_not_one_hot = labels.data.cpu().numpy()
            tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
            tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
            targets_mc[i,:,:] = np.transpose(tar)
        # Targets stay the same
        targets = targets_mc[:,:,0]
        if mdlParams['voting_scheme'] == 'vote':
            # Vote for correct prediction
            print("Pred Shape",predictions_mc.shape)
            predictions_mc = np.argmax(predictions_mc,1)    
            print("Pred Shape",predictions_mc.shape) 
            for j in range(predictions_mc.shape[0]):
                predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
            print("Pred Shape",predictions.shape) 
        elif mdlParams['voting_scheme'] == 'average':
            predictions = np.mean(predictions_mc,2)
    else:    
        if mdlParams.get('model_type_cnn') is not None and mdlParams['numRandValSeq'] > 0:
            loss_all = np.zeros([numBatches])
            allInds = np.zeros([len(mdlParams[indices])])
            predictions = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])
            targets = np.zeros([len(mdlParams[indices]),mdlParams['numClasses']])        
            loss_mc = np.zeros([len(mdlParams[indices])])
            predictions_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['numRandValSeq']])
            targets_mc = np.zeros([len(mdlParams[indices]),mdlParams['numClasses'],mdlParams['numRandValSeq']])   
            for i, (inputs, labels, inds) in enumerate(modelVars['dataloader_'+indices]):
                # Get data
                inputs = inputs.to(modelVars['device'])
                labels = labels.to(modelVars['device'])       
                # Not sure if thats necessary
                modelVars['optimizer'].zero_grad()    
                with torch.set_grad_enabled(False):
                    # Get outputs
                    if mdlParams['aux_classifier']:
                        outputs, outputs_aux = modelVars['model'](inputs)
                        if mdlParams['eval_aux_classifier']:
                            outputs = outputs_aux
                    else:
                        outputs = modelVars['model'](inputs)
                    preds = modelVars['softmax'](outputs)      
                    # Loss
                    loss = modelVars['criterion'](outputs, labels)           
                # Write into proper arrays
                loss_mc[i] = np.mean(loss.cpu().numpy())
                predictions_mc[i,:,:] = np.transpose(preds)
                tar_not_one_hot = labels.data.cpu().numpy()
                tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1
                targets_mc[i,:,:] = np.transpose(tar)
            # Targets stay the same
            targets = targets_mc[:,:,0]
            if mdlParams['voting_scheme'] == 'vote':
                # Vote for correct prediction
                print("Pred Shape",predictions_mc.shape)
                predictions_mc = np.argmax(predictions_mc,1)    
                print("Pred Shape",predictions_mc.shape) 
                for j in range(predictions_mc.shape[0]):
                    predictions[j,:] = np.bincount(predictions_mc[j,:],minlength=mdlParams['numClasses'])   
                print("Pred Shape",predictions.shape) 
            elif mdlParams['voting_scheme'] == 'average':
                predictions = np.mean(predictions_mc,2)
        else:
            for i, (inputs, labels, indices) in enumerate(modelVars['dataloader_'+indices]):
                # Get data
                inputs = inputs.to(modelVars['device'])
                labels = labels.to(modelVars['device'])       
                # Not sure if thats necessary
                modelVars['optimizer'].zero_grad()    
                with torch.set_grad_enabled(False):
                    # Get outputs
                    if mdlParams['aux_classifier']:
                        outputs, outputs_aux = modelVars['model'](inputs)
                        if mdlParams['eval_aux_classifier']:
                            outputs = outputs_aux
                    else:
                        outputs = modelVars['model'](inputs)
                    #print("in",inputs.shape,"out",outputs.shape)
                    preds = modelVars['softmax'](outputs)      
                    # Loss
                    loss = modelVars['criterion'](outputs, labels)     
                # Write into proper arrays                
                if i==0:
                    loss_all = np.array([loss.cpu().numpy()])
                    predictions = preds.cpu().numpy()
                    tar_not_one_hot = labels.data.cpu().numpy()
                    tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                    tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1   
                    targets = tar    
                    #print("Loss",loss_all)         
                else:                 
                    loss_all = np.concatenate((loss_all,np.array([loss.cpu().numpy()])),0)
                    predictions = np.concatenate((predictions,preds.cpu().numpy()),0)
                    tar_not_one_hot = labels.data.cpu().numpy()
                    tar = np.zeros((tar_not_one_hot.shape[0], mdlParams['numClasses']))
                    tar[np.arange(tar_not_one_hot.shape[0]),tar_not_one_hot] = 1                   
                    targets = np.concatenate((targets,tar),0)
                    #allInds[(i*len(mdlParams['numGPUs'])+k)*bSize:(i*len(mdlParams['numGPUs'])+k+1)*bSize] = res_tuple[3][k]
            predictions_mc = predictions
    #print("Check Inds",np.setdiff1d(allInds,mdlParams[indices]))
    # Calculate metrics
    if exclude_class is not None:
        predictions = np.concatenate((predictions[:,:exclude_class],predictions[:,exclude_class+1:]),1)
        targets = np.concatenate((targets[:,:exclude_class],targets[:,exclude_class+1:]),1)    
        num_classes = mdlParams['numClasses']-1
    else:
        num_classes = mdlParams['numClasses']
    # Accuarcy
    acc = np.mean(np.equal(np.argmax(predictions,1),np.argmax(targets,1)))
    # Confusion matrix
    conf = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1))
    if conf.shape[0] < num_classes:
        conf = np.ones([num_classes,num_classes])
    # Class weighted accuracy
    wacc = conf.diagonal()/conf.sum(axis=1)    
    # Sensitivity / Specificity
    sensitivity = np.zeros([num_classes])
    specificity = np.zeros([num_classes])
    if num_classes > 2:
        for k in range(num_classes):
                sensitivity[k] = conf[k,k]/(np.sum(conf[k,:]))
                true_negative = np.delete(conf,[k],0)
                true_negative = np.delete(true_negative,[k],1)
                true_negative = np.sum(true_negative)
                false_positive = np.delete(conf,[k],0)
                false_positive = np.sum(false_positive[:,k])
                specificity[k] = true_negative/(true_negative+false_positive)
                # F1 score
                f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1),average='weighted')                
    else:
        tn, fp, fn, tp = confusion_matrix(np.argmax(targets,1),np.argmax(predictions,1)).ravel()
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        # F1 score
        f1 = f1_score(np.argmax(predictions,1),np.argmax(targets,1))
    # AUC
    fpr = {}
    tpr = {}
    roc_auc = np.zeros([num_classes])
    if num_classes > 9:
        print(predictions)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return np.mean(loss_all), acc, sensitivity, specificity, conf, f1, roc_auc, wacc, predictions, targets, predictions_mc 

def modify_densenet_avg_pool(model):
    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = torch.mean(torch.mean(x,2), 2)
        #x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)

    return model    
    
