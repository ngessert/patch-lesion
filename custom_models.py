import torch
import torch.nn as nn
import numpy as np
import models
from torchvision import models as tvmodels

# helper
def modify_densenet(model,mdlParams):
    # Add new inital convolutional layer
    if mdlParams['bidirectional']:
        conv_new = nn.Conv2d(2*mdlParams['convGRU_FM_hidden'][-1],mdlParams['convGRU_FM_first_cnn_layer'],kernel_size=7,stride=2,padding=3,bias=False)
    else:
        conv_new = nn.Conv2d(mdlParams['convGRU_FM_hidden'][-1],mdlParams['convGRU_FM_first_cnn_layer'],kernel_size=7,stride=2,padding=3,bias=False)
    # Perhaps initialize its weights with the other ones
    if mdlParams['convGRU_init_first_cnn_layer']:
        conv_old = model.features.conv0
        if np.mod(conv_new.in_channels,conv_old.in_channels) != 0:
            raise ValueError('New initial conv must have an input FM size divisible by 3')
        weight_ratio = int(conv_new.in_channels/conv_old.in_channels)
        for i in range(weight_ratio):
            conv_new.weight.data[:,i*conv_old.in_channels:(i+1)*conv_old.in_channels,:,:] = conv_old.weight.data
    # Modify forward pass
    model.features.conv0 = conv_new
    return model

class SEPatchLayer(nn.Module):
    def __init__(self, channel):
        super(SEPatchLayer, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(channel, channel),
                nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        if len(x.shape) == 5:
            b, p, _, _, _ = x.size()
            y = torch.mean(torch.mean(torch.mean(x,2),2),2)
            y = self.fc(y).view(b, p, 1, 1, 1)
        else:
            b, p, _ = x.size()
            y = torch.mean(x,2)
            y = self.fc(y).view(b, p, 1)    
        if return_attention:
            return x * y, y
        else:        
            return x * y

class SEPatchLayer_Pool(nn.Module):
    def __init__(self, channel_in, out_size):
        super(SEPatchLayer_Pool, self).__init__()
        if len(out_size) == 3:
            self.fc = nn.Sequential(
                    nn.Linear(channel_in*out_size[0]*out_size[1]*out_size[2],channel_in),
                    nn.Sigmoid()
            )
            self.pool = nn.AdaptiveAvgPool3d(out_size)
        else:
            self.fc = nn.Sequential(
                    nn.Linear(channel_in*out_size[0],channel_in),
                    nn.Sigmoid()
            )
            self.pool = nn.AdaptiveAvgPool1d(out_size[0])            

    def forward(self, x):
        if len(x.shape) == 5:
            b, p, _, _, _ = x.size()
            y = self.pool(x).view(b,-1)
            y = self.fc(y).view(b, p, 1, 1, 1)
        else:
            b, p, _ = x.size()
            y = self.pool(x).view(b,-1)
            y = self.fc(y).view(b, p, 1)            
        return x * y        

class CNN_FC(nn.Module):
    def __init__(self,mdlParams):
        super(CNN_FC, self).__init__()
        # Some necessary vars
        self.crop_number = mdlParams['multiCropTrain']
        self.combine_features = mdlParams['combine_features']
        self.cnn_output_point = mdlParams['CNN_Output_Point']
        self.initial_attention = mdlParams['initial_attention']
        if mdlParams.get('end_pool') is not None:
            self.end_pool = mdlParams['end_pool']
        else:
            self.end_pool = False
        self.end_attention = mdlParams['end_attention']
        # CNN first,up to feature vector
        self.cnn = models.getModel(mdlParams['model_type_cnn'])()
        #print(self.cnn)
        if 'Dense' in mdlParams['model_type_cnn']:
            if self.cnn_output_point == 'end':
                self.cnn_features = self.cnn.features
            elif self.cnn_output_point == 'transition3':
                self.cnn_features = nn.Sequential(*list(self.cnn.features.children())[:10])
        elif 'InceptionV3' in mdlParams['model_type_cnn']:
            if self.cnn_output_point == 'end':
                self.cnn_features = nn.Sequential(self.cnn.Conv2d_1a_3x3,self.cnn.Conv2d_2a_3x3,self.cnn.Conv2d_2b_3x3,nn.MaxPool2d(kernel_size=3,stride=2),self.cnn.Conv2d_3b_1x1,self.cnn.Conv2d_4a_3x3,nn.MaxPool2d(kernel_size=3,stride=2),
                                                    self.cnn.Mixed_5b,self.cnn.Mixed_5c,self.cnn.Mixed_5d,self.cnn.Mixed_6a,self.cnn.Mixed_6b,self.cnn.Mixed_6c,self.cnn.Mixed_6d,self.cnn.Mixed_6e,
                                                    self.cnn.Mixed_7a,self.cnn.Mixed_7b,self.cnn.Mixed_7c)
        else:
            if self.cnn_output_point == 'end':
                self.cnn_features = nn.Sequential(self.cnn.layer0,self.cnn.layer1,self.cnn.layer2,self.cnn.layer3,self.cnn.layer4)
        #print(self.cnn_features)
        # The classifier
        if self.combine_features == 'add' or self.combine_features == 'conv1':
            self.classifier = nn.Linear(mdlParams['CNN_Features'],mdlParams['numClasses'])
        else:
            self.classifier = nn.Linear(self.crop_number*mdlParams['CNN_Features'],mdlParams['numClasses'])         
        if self.combine_features == 'conv1':      
            self.conv1x1 = nn.Conv1d(self.crop_number,1,1,1)
        if self.initial_attention:
            if 'attention_size' in mdlParams:
                self.initial_attention_layer = SEPatchLayer_Pool(mdlParams['multiCropTrain'],mdlParams['attention_size'][0])
            else:
                self.initial_attention_layer = SEPatchLayer(mdlParams['multiCropTrain'])
        if self.end_attention:
            if 'attention_size' in mdlParams:
                self.end_attention_layer = SEPatchLayer_Pool(mdlParams['multiCropTrain'],mdlParams['attention_size'][1]) 
            else:
                self.end_attention_layer = SEPatchLayer(mdlParams['multiCropTrain']) 

    def forward(self, x):
        # Perhaps, attention first
        if self.initial_attention:
            x = self.initial_attention_layer(x)        
        # Frist: reshape time into batch dim
        cnn_in = torch.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])
        # CNN
        #print("cnn in",cnn_in.shape)
        cnn_features = self.cnn_features(cnn_in)
        #print("cnn out",cnn_features.shape)
        # Pool
        #print("CNn Features",cnn_features.shape)
        cnn_features = torch.mean(torch.mean(cnn_features,2),2)
        #print("CNn Features",cnn_features.shape)
        # Reshape
        class_in = torch.reshape(cnn_features,[int(cnn_features.shape[0]/self.crop_number),self.crop_number,cnn_features.shape[1]])
        if self.end_attention:
            class_in = self.end_attention_layer(class_in)
        if self.end_pool:
            class_in = class_in.view([class_in.shape[0]*class_in.shape[1],class_in.shape[2]])
            output_all = self.classifier(class_in)
            output_all = output_all.view([int(output_all.shape[0]/self.crop_number),self.crop_number,output_all.shape[1]])
            output = torch.mean(output_all,dim=1)
        else:
            if self.combine_features == 'add':
                class_in = torch.sum(class_in,dim=1)
            elif self.combine_features == 'conv1':
                class_in = torch.squeeze(self.conv1x1(class_in),dim=1)
            else:
                class_in = torch.squeeze(torch.cat(torch.chunk(class_in,class_in.shape[1],1),2),dim=1)
                #print("After",class_in.shape)
            # Get last, to classifier
            output = self.classifier(class_in)
        return output

    def forward_analysis(self, x):
        # Perhaps, attention first
        if self.initial_attention:
            x, att_in = self.initial_attention_layer(x,return_attention=True)  
        else:
            att_in = None      
        # Frist: reshape time into batch dim
        cnn_in = torch.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])
        # CNN
        cnn_features = self.cnn_features(cnn_in)
        # Pool
        #print("CNn Features",cnn_features.shape)
        cnn_features = torch.mean(torch.mean(cnn_features,2),2)
        #print("CNn Features",cnn_features.shape)
        # Reshape
        class_in = torch.reshape(cnn_features,[int(cnn_features.shape[0]/self.crop_number),self.crop_number,cnn_features.shape[1]])
        pooled_out = class_in
        if self.end_attention:
            class_in, att_out = self.end_attention_layer(class_in,return_attention=True)
        else:
            att_out = None
        if self.end_pool:
            class_in = class_in.view([class_in.shape[0]*class_in.shape[1],class_in.shape[2]])
            output_all = self.classifier(class_in)
            output_all = output_all.view([int(output_all.shape[0]/self.crop_number),self.crop_number,output_all.shape[1]])
            output = torch.mean(output_all,dim=1)
        else:
            if self.combine_features == 'add':
                class_in = torch.sum(class_in,dim=1)
            elif self.combine_features == 'conv1':
                class_in = torch.squeeze(self.conv1x1(class_in),dim=1)
            else:
                class_in = torch.squeeze(torch.cat(torch.chunk(class_in,class_in.shape[1],1),2),dim=1)
                #print("After",class_in.shape)
            # Get last, to classifier
            output = self.classifier(class_in)
        return output_all, output, att_in, att_out, pooled_out   


    
class CNN_GRU(nn.Module):
    def __init__(self,mdlParams):
        super(CNN_GRU, self).__init__()
        # Some necessary vars
        self.crop_number = mdlParams['multiCropTrain']
        self.cell_type = mdlParams['cell_type']
        self.cnn_output_point = mdlParams['CNN_Output_Point']
        self.aux_classifier = mdlParams['aux_classifier']
        # CNN first,up to feature vector
        self.cnn = models.getModel(mdlParams['model_type_cnn'])()
        if 'Dense' in mdlParams['model_type_cnn']:
            if self.cnn_output_point == 'end':
                self.cnn_features = self.cnn.features
            elif self.cnn_output_point == 'transition3':
                self.cnn_features = nn.Sequential(*list(self.cnn.features.children())[:10])
        elif 'InceptionV3' in mdlParams['model_type_cnn']:
            if self.cnn_output_point == 'end':
                self.cnn_features = nn.Sequential(self.cnn.Conv2d_1a_3x3,self.cnn.Conv2d_2a_3x3,self.cnn.Conv2d_2b_3x3,nn.MaxPool2d(kernel_size=3,stride=2),self.cnn.Conv2d_3b_1x1,self.cnn.Conv2d_4a_3x3,nn.MaxPool2d(kernel_size=3,stride=2),
                                                    self.cnn.Mixed_5b,self.cnn.Mixed_5c,self.cnn.Mixed_5d,self.cnn.Mixed_6a,self.cnn.Mixed_6b,self.cnn.Mixed_6c,self.cnn.Mixed_6d,self.cnn.Mixed_6e,
                                                    self.cnn.Mixed_7a,self.cnn.Mixed_7b,self.cnn.Mixed_7c)                
        else:
            if self.cnn_output_point == 'end':
                self.cnn_features = nn.Sequential(self.cnn.layer0,self.cnn.layer1,self.cnn.layer2,self.cnn.layer3,self.cnn.layer4)
        # The GRU
        if self.cell_type == 'A':
            self.gru = nn.GRU(mdlParams['CNN_Features'],mdlParams['GRU_FM_Hidden'],mdlParams['GRU_num_layers'],batch_first=True,bidirectional=mdlParams['bidirectional'])
        # The classifier
        if mdlParams['bidirectional']:
            self.classifier = nn.Linear(2*mdlParams['GRU_FM_Hidden'],mdlParams['numClasses'])
        else:
            self.classifier = nn.Linear(mdlParams['GRU_FM_Hidden'],mdlParams['numClasses'])
        if self.aux_classifier:
            num_ftrs = self.cnn.classifier.in_features
            self.cnn.classifier = nn.Linear(num_ftrs, mdlParams['numClasses'])                 

    def forward(self, x):
        # Frist: reshape time into batch dim
        cnn_in = torch.reshape(x,[x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4]])
        # CNN
        cnn_features = self.cnn_features(cnn_in)
        # Pool
        #print("CNn Features",cnn_features.shape)
        cnn_features = torch.mean(torch.mean(cnn_features,2),2)
        #print("CNn Features",cnn_features.shape)
        # Reshape
        gru_input = torch.reshape(cnn_features,[int(cnn_features.shape[0]/self.crop_number),self.crop_number,cnn_features.shape[1]])
        #print("Gru In",gru_input.shape)
        # Feed into gru
        if self.cell_type == 'A':
            gru_out, _ = self.gru(gru_input)
            #print("Gru Out",gru_out.shape)
            # Get last, to classifier
            output = self.classifier(gru_out[:,-1,:])
        if self.aux_classifier:
            # Also return original cnn output   
            output_aux = self.cnn.classifier(cnn_features)    
            return output, output_aux
        else:
            return output


class CNN_GRU_TP(nn.Module):
    def __init__(self,mdlParams):
        super(CNN_GRU_TP, self).__init__()
        # Some necessary vars
        self.crop_number = mdlParams['multiCropTrain']
        self.cell_type = mdlParams['cell_type']
        self.cnn_output_point = mdlParams['CNN_Output_Point']
        # CNN first,up to feature vector
        self.cnn = models.getModel(mdlParams['model_type_cnn'])()
        if self.cnn_output_point == 'end':
            self.cnn_features = self.cnn.features
        elif self.cnn_output_point == 'transition3':
            self.cnn_features = nn.Sequential(*list(self.cnn.features.children())[:10])
        # The GRU
        if self.cell_type == 'A':
            self.gru = nn.GRU(mdlParams['CNN_Features'],mdlParams['GRU_FM_Hidden'],mdlParams['GRU_num_layers'],batch_first=True,bidirectional=mdlParams['bidirectional'])
        # The second CNN for global, make this better
        self.cnn_global = tvmodels.densenet121(pretrained=True)
        if self.cnn_output_point == 'end':
            self.cnn_features_global = self.cnn_global.features
        elif self.cnn_output_point == 'transition3':
            self.cnn_features_global = nn.Sequential(*list(self.cnn_global.features.children())[:10])
        # The GRU
        # The classifier
        if mdlParams['bidirectional']:
            self.classifier = nn.Linear(2*mdlParams['GRU_FM_Hidden']+mdlParams['CNN_Features'],mdlParams['numClasses'])
        else:
            self.classifier = nn.Linear(mdlParams['GRU_FM_Hidden']+mdlParams['CNN_Features'],mdlParams['numClasses'])
  

    def forward(self, x):
        # Split into high-res patches + low-res global
        x_seq = x[:,:-1,:,:,:]
        x_glob = x [:,-1,:,:,:]
        # First CNN gets seq + GRU
        # Frist: reshape time into batch dim
        cnn_in = torch.reshape(x_seq,[x_seq.shape[0]*x_seq.shape[1],x_seq.shape[2],x_seq.shape[3],x_seq.shape[4]])
        # CNN
        cnn_features = self.cnn_features(cnn_in)
        # Pool
        #print("CNn Features",cnn_features.shape)
        cnn_features = torch.mean(torch.mean(cnn_features,2),2)
        #print("CNn Features",cnn_features.shape)
        # Reshape
        gru_input = torch.reshape(cnn_features,[cnn_features.shape[0]/self.crop_number,self.crop_number,cnn_features.shape[1]])
        #print("Gru In",gru_input.shape)
        # Feed into gru
        if self.cell_type == 'A':
            gru_out, _ = self.gru(gru_input)
            #print("Gru Out",gru_out.shape)
            # Get last, to classifier
            output_seq = gru_out[:,-1,:]        
        # Second CNN gets global image
        cnn_features_global = self.cnn_features_global(x_glob)
        # Pool
        cnn_features_global = torch.mean(torch.mean(cnn_features_global,2),2)
        # Fuse at output 
        output = self.classifier(torch.cat((output_seq,cnn_features_global),1))
        return output