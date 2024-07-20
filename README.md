# TC-Net
Tree Completion Net: A Novel Vegetation Point Clouds Completion Model Based on Deep Learning
## Introduction
To improve the integrity of vegetation point clouds, the missing vegetation point can be compensated through vegetation point clouds completion technology. Further, it can enhance the accuracy of these point clouds’ applications, particularly in terms of quantitative calculations, such as for the urban living vegetation volume (LVV). However, owing to the factors like the mutual occlusion between ground objects, sensor perspective, and penetration ability limitations resulting in missing single tree point clouds’ structures, the existing completion techniques cannot be directly applied to the single tree point clouds completion. This study combines the cutting-edge deep learning techniques, e.g., the self-supervised and multi-scale Encoder (Decoder), to propose a tree completion net (TC-Net) model suitable for the single tree structure completion. The intention is to resolve the problem owing to the point clouds completion technology based on deep learning, being still in its initial stage and a lack of good tree completion approach. In the model, the self-supervised module employs a data-driven approach to learn predicting the missing structure of a single tree point clouds, whereas the multi-scale Encoder (Decoder) is employed to capture the semantic features of different spatial scales of a single tree point clouds and gradually predict the missing part of the point clouds. Among the constructing processes of the single tree complete (incomplete) pairs to train TC-Net model, the commonly used random spherical loss pattern in point clouds completion is not suitable for the single tree structure completion. Inspired by the attenuation of electromagnetic waves through a uniform medium, this study proposes an uneven density loss pattern. Also, this study uses local similarity visualization method, which is different from ordinary Chamfer distance (CD) value and can better assists in visually assessing the effects of point cloud completion. Experiments have shown that the TC-Net trained model, based on the uneven density loss pattern, can effectively discover the missing areas, and then compensate the structure of the incomplete single tree point clouds in real scenarios. The average CD value of the incomplete point clouds have decreased of over 2.0 after completion. The best completion effect is that the CD value of the incomplete point cloud decreases from 23.89 to 13.08 after completion. With respect to the application of urban area LVV calculation, the average MAE, RSME, and MAPE of the LVV values in the experimental area, calculated through incomplete point clouds and completed point clouds, have decreased from 9.57, 7.78, and 14.11% to 1.86, 2.84, and 5.23%, respectively, thus verifying the effectiveness of TC-Net.
## Requirements of Operating Environment
Python>=3.8, pytorch>=1.13, CUDA11.7, tqdm>=4.66.1, numpy>=1.24.3
## Model Training
`python Train_TCNet.py --data_root ../../Data/OwnTree/ --route_type 1 --missing_type 3 --netG ../../Predict/Shapenet_Original/230317_TCNet_routetype1_missingtype3/point_netG200.pth --resume_epoch 200 --niter 401 --save_model_root ../Predict/OwnTree/230317_TCNet_routetype1_missingtype3_Tree_pretrain/ --batchSize 32 --workers 16 --cuda Ture --gpu 1` 

Parameter Description:

--data_root: Dataset location
--route_type 1 --missing_type 3: Parameters for uneven density missing
--netG: Pre-trained weights on Shapenet-part, data located at "PCSS\Data\Shapenet_Original".
--resume_epoch 200 --niter 401: Start training from the 200th epoch of pre-training, for 200 iterations.
--D_choose 0: The original PF-Net has auxiliary GAN loss, which is not used here.
--save_model: Location to save training weights and records.
--batchSize 16 --workers 64 --cuda True --gpu 1: Parameters related to GPU load.
