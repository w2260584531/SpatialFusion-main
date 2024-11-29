# Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST


![]([https://github.com/JinmiaoChenLab/GraphST/blob/main/GraphST.jpg](https://github.com/w2260584531/SpatialFusion-main/blob/main/model.png))

## Overview
GraphST is a versatile graph self-supervised contrastive learning model that incorporates spatial location information and gene expression profiles to accomplish three key tasks, spatial clustering, spatial transcriptomics (ST) data integration, and single-cell RNA-seq (scRNA-seq) transfer onto ST. GraphST combines graph neural networks (GNNs) with self-supervised contrastive learning to learn spot representations in the ST data by modeling gene expressions and spatial locaiton information. After the representation learning, the non-spatial alignment algorithm is used to cluster the spots into different spatial domains. Each cluster is regarded as a spatial domain, containing spots with similar gene expression profiles and spatially proximate. GraphST can jointly analyze multiple ST samples while correcting batch effects, which is achieved by smoothing features between spatially adjacent spots across samples. For the scRNA-seq transfer onto ST data, a mapping matrix is trained via an augmentation-free contrastive learning mechanism, where the similarity of spatially adjacent spots are maximized while those of spatially non-adjacent spots are minimized. With the learned mapping matrix, arbitrary cell attributes (e.g., cell type and sample type) can be flexibly projected onto spatial space.   

## Requirements
You'll need to install the following packages in order to run the codes.
* python==3.8
* torch>=1.8.0
* cudnn>=10.2
* numpy==1.22.3
* scanpy==1.9.1
* anndata==0.8.0
* rpy2==3.4.1
* pandas==1.4.2
* scipy==1.8.1
* scikit-learn==1.1.1
* tqdm==4.64.0
* matplotlib==3.4.2
* R==4.0.3


