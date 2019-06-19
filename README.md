## Skin Lesion Classification Using CNNs with Patch-Based Attention and Diagnosis-Guided Loss Weighting

This repository contains the code for the above mentioned paper.

The paper is available here: https://ieeexplore.ieee.org/abstract/document/8710336 
Preprint: https://arxiv.org/abs/1905.02793

The insights in this paper are partially based on our submission to the ISIC2018 Skin Lesion Diagnosis challenge where we achieved second place while being the best approach using only publicly available data ([Leaderboards](https://challenge2018.isic-archive.com/leaderboards/)).

Associated challenge arxiv paper: https://arxiv.org/abs/1808.01694

The code for the challenge submission is also available here: https://github.com/ngessert/isic2018

### Patch-based attention

If you are only interested in patch-based attention, go to models_custom.py.

### Dataset Preparation

The HAM10000 dataset is available here: https://isic-archive.com/

The images' and labels' directory strucutre should look like this: /task3/images/HAM10000/ISIC_0024306.jpg and /task3/labels/HAM10000/labels.csv. The labels in the CSV file should be structured as follows: first column contains the image ID ("ISIC_0024306"), then the one-hot encoded labels follow (ISIC challenge standard format).

In addition, we use the sevenpoint dataset which is available here: https://github.com/jeremykawahara/derm7pt

Use gen_sevenpoint.py to convert the downloaded dataset to our format. Put the resulting images into /task3/images/sevenpoint and the label csv file into /task3/labels/sevenpoint

In terms of train/val/test split, use the indices_tbe.pkl file for HAM10000 and indices_sp.pkl for the sevenpoint dataset.

### Configuration

You can find example configs in /cfgs/. You can vary the parameters in these files. This includes balancing strategies, data input strategies and different models (see paper for details).

### Training

To train a model, run: `python train.py linux example_densenet_sevenpoint gpu0`

### Evaluation

To evaluate a model, run: `python eval.py linux example_densenet_sevenpoint multiorder5 average $HOME/data/isic/example_densenet_sevenpoint lastgpu0` 

