import csv
import numpy as np
import os 
import sys
from glob import glob
import pickle
from shutil import copyfile

meta_file = (r'\\path_to_meta_folder\meta\meta.csv').replace(os.sep,'/')
test_inds_orig = (r'\\path_to_folder\meta\test_indexes.csv').replace(os.sep,'/')
val_inds_orig = (r'\\path_to_folder\meta\valid_indexes.csv').replace(os.sep,'/')
train_inds_orig = (r'\\path_to_folder\meta\train_indexes.csv').replace(os.sep,'/')

src_fold_im = (r'\\path_to_folder\images').replace(os.sep,'/')
tar_fold_im = (r'\\path_to_folder\sevenpoint').replace(os.sep,'/')
tar_label_csv = (r'\\path_to_folder\labels.csv').replace(os.sep,'/')
tar_ind_pkl = (r'\\path_to_folder\indices_sp.pkl').replace(os.sep,'/')

# New indices
copy_images =False
indices = {}
indices['trainIndCV'] = []
indices['valIndCV'] = []
# Add these later
train_inds = []
test_inds = []
# Get all orig indices
orig_train_inds = []
orig_val_inds  = []
orig_test_inds = []
# Train
with open(train_inds_orig, newline='') as csvfile:
    labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in labels_str:
        if 'ind' in row[0]:
            continue
        orig_train_inds.append(int(float(row[0])))
# Val
with open(val_inds_orig, newline='') as csvfile:
    labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in labels_str:
        if 'ind' in row[0]:
            continue
        orig_val_inds.append(int(float(row[0])))
# Test
with open(test_inds_orig, newline='') as csvfile:
    labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in labels_str:
        if 'ind' in row[0]:
            continue
        orig_test_inds.append(int(float(row[0])))
# To array
orig_train_inds = np.array(orig_train_inds)
orig_val_inds  = np.array(orig_val_inds)
orig_test_inds = np.array(orig_test_inds)
# Open labels file
csv_label_file = open(tar_label_csv, 'w')
csv_label_file.write("image,MEL,NV,BCC,AKIEC,BKL,DF,VASC,Diagtype\n")
# Track labels
all_labels = []

# Go through meta data file and copy images to target location
with open(meta_file, newline='') as csvfile:
    labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in labels_str:
        # Skip first line
        if 'case_num' in row[0]:
            continue
        # Check diagnosis and assign to class
        curr_class = np.zeros([7],dtype=np.int32)
        if 'basal cell' in row[1]:
            curr_class[2] = 1
        elif 'nevus' in row[1]:
            curr_class[1] = 1
        elif 'dermatofibroma' in row[1]:
            curr_class[5] = 1
        elif 'lentigo' in row[1]:
            curr_class[4] = 1
        elif 'melanoma' in row[1]:
            curr_class[0] = 1
        elif 'seborrheic keratosis' in row[1]:
            curr_class[4] = 1
        elif 'vascular lesion' in row[1]:
            curr_class[6] = 1
        else:
            print("No fitting class for diagnosis",row[1])
        # If we found a class for the example, write to csv file, copy image, add index
        if np.sum(curr_class) == 1:
            # Write label
            csv_label_file.write(row[16].replace('/','_').replace('.jpg','') + "," + str(curr_class[0]) + "," +  str(curr_class[1]) + "," +  str(curr_class[2]) + "," +  str(curr_class[3]) + "," +  str(curr_class[4]) + "," +  str(curr_class[5]) + "," +  str(curr_class[6]) + "," + row[14] +"\n")
            # Copy image
            if copy_images:
                copyfile(src_fold_im + '/' + row[16],tar_fold_im + '/' + row[16].replace('/','_'))
            all_labels.append(curr_class)

all_labels = np.array(all_labels)
print(np.sum(all_labels,0))
#print(orig_train_inds)
all_images = sorted(glob(tar_fold_im+'/*'))
for ind, file in enumerate(all_images):
    csvfile = open(meta_file, newline='')
    labels_str = csv.reader(csvfile, delimiter=',', quotechar='|')
    found = False
    for row in labels_str:
        # Find example
        #print(row[16].replace('/','_'))
        #print(file)
        if row[16].replace('/','_') in file:       
            # Define index
            if int(row[0]) in orig_train_inds or int(row[0]) in orig_val_inds:
                train_inds.append(ind)
                found = True
            elif int(row[0]) in orig_test_inds:
                test_inds.append(ind)
                found = True
    if not found:
        print("Did not find",file)
    csvfile.close()



# Save indices
indices['trainIndCV'].append(np.array(train_inds))
indices['valIndCV'].append(np.array(test_inds))
with open(tar_ind_pkl,'wb') as f:
    pickle.dump(indices,f,pickle.HIGHEST_PROTOCOL) 