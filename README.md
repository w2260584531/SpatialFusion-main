# Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST


![](https://github.com/w2260584531/SpatialFusion-main/blob/main/model.png)

## Overview
In recent years, the rapid development of Spatial Transcriptomics (ST) has provided unprecedented opportunities to uncover tissue heterogeneity and intercellular spatial interactions. However, existing methods for spatial domain identification and cell type deconvolution still face challenges in terms of accuracy, robustness, and computational efficiency. To address these issues, we propose a novel deep learning model, SpatialFusion, which integrates gene expression, spatial coordinates, and functional information to enhance both spatial domain identification and cell type deconvolution performance. The core innovation of SpatialFusion lies in its combination of Graph Neural Networks (GNN) and attention mechanisms, capturing complex intercellular spatial relationships through multi-dimensional embeddings of spatial structure and functional similarity. The model optimizes the spatial domain identification process through a dual-encoding strategy (co-learning of spatial graphs and feature maps) and self-supervised contrastive learning tasks. This approach demonstrates significant advantages in spatial domain identification, cell type deconvolution, and data robustness across multiple datasets.In experiments using the human dorsolateral prefrontal cortex (DLPFC) dataset, SpatialFusion outperformed existing models in terms of accuracy and resolution, particularly excelling at capturing complex layer-specific expression profiles. In the cell type deconvolution task, SpatialFusion showed strong robustness to noise and low cell density, accurately mapping the spatial distribution of different cell types. Furthermore, in the analysis of the breast cancer tumor microenvironment, SpatialFusion successfully revealed the spatial heterogeneity of the tumor microenvironment and identified potential therapeutic targets, such as the differentially expressed genes COX6C and CCND1, offering new insights for precision medicine and cancer research.

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


