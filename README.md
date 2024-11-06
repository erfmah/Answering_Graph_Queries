# Deep Generative Models for Subgraph Prediction
**Authors**: Erfaneh Mahmoudzadeh, Parmis Naddaf, Kiarash Zahirnia, Oliver Schulte

This is a PyTorch implementation of Subgraph Prediction Via Inference from a Model.
## Overview
Graph Neural Networks (GNNs) are important across different domains, such as social network analysis and recommendation systems, due to their ability to model complex relational data. This paper introduces subgraph queries as a new task for deep graph learning. Unlike traditional graph prediction tasks that focus on individual components like link prediction or node classification, subgraph queries jointly predict the components of a target subgraph based on evidence that is represented by an observed subgraph. For instance, a subgraph query can predict a set of target links and/or node labels.  To answer subgraph queries, we utilize a probabilistic deep Graph Generative Model. Specifically, we inductively train a Variational Graph Auto-Encoder (VGAE) model, augmented to represent a joint distribution over links, node features and labels. Bayesian optimization is used to tune a weighting for the relative importance of links, node features and labels in a specific domain. 
We describe a deterministic and a sampling-based inference method for estimating subgraph probabilities from the VGAE generative graph distribution, without retraining, in zero-shot fashion. For evaluation, we apply the inference methods on a range of subgraph queries on six benchmark datasets. We find that  inference from a model achieves superior predictive performance, surpassing independent prediction baselines with improvements in AUC scores ranging from 0.06 to 0.2 points, depending on the dataset. 

## Run
To execute the main.py file, run the model, and address the graph queries, you must customize the parameters within the command.
For example this command runs the code in Fully inductive setting, with Monte Carlo sampling, GIN encoder for single subgraph prediction on Cora dataset:
```sh
python main.py --fully_inductive True --encoder_type "Multi_GIN" --sampling_method "monte" --method "single" --dataSet "Cora_dgl" 
```


## Semi/Fully Inductive Setting

In fully inductive setting, all query nodes are test nodes. The evidence set comprises all links between test nodes that are not target links. You can run the model fully inductive by assigning any value to "--fully_inductive" parameter:
```sh
python main.py --fully_inductive True
```
Or you can run the model in semi inductive setting where, Each target link connects at least one test node. The evidence set comprises all links from the input graph that are not target links. By default, the model operates in the semi-inductive setting.
## Encoder Types
You can run this model with three different encoders by using following commands:
- [**VGAE-GCN**](https://openreview.net/pdf?id=SJU4ayYgl) Graph Convolutional Neural Network is a popular encoder:
    - ```python main.py --encoder_type "Multi_GCN"```
- [**VGAE-GAT**](https://openreview.net/forum?id=rJXMpikCZ) Graph Attention Networks add link attention weights to graph convolutions:
    - ```python main.py --encoder_type "Multi_GAT" ```
- [**VGAE-GIN**](https://openreview.net/pdf?id=ryGs6iA5Km) The Graph Isomorphism Network is a type of GNN that consists of two steps of         aggregation and combination:
    - ```python main.py --encoder_type "Multi_GIN" ```
- [**VGAE-SAGE**](https://arxiv.org/pdf/1706.02216) GraphSAGE encoder:
    - ```python main.py --encoder_type "Multi_SAGE"```

## Sampling Methods
During the inference step, you can choose from three available sampling methods. You can select the desired sampling method using the following commands:
- **Deterministic inference**:
    - ```python main.py --sampling_method "deterministic" ```
- **Monte Carlo inference**:
    - ```python main.py --sampling_method "monte" ```

## Single/Multi Subgraph Prediction
This model can answer two types of queries:
- **Single Subgraph Queries**, where each query has one target edge and one target node. This is the traditional link prediction setup.
    - ```python main.py --method "single" ```
- **Joint Subgraph Prediction Queries**, where each query has at least one target edge and at least one target node. This is the novel task considered in our paper.
    - ```python main.py --method "multi" ```

## tuning hyperparameters
The weights for diffrenet tasks are for loss function are available in weights.csv file. If you need to tune weights again for new datasets or settings you can use the following command:
    - ```python main.py --tuning "True" ```

## Data
You have the option to select a dataset from a set of available options such as Cora and CiteSeer datasets. To use a specific dataset, you can execute the following command:
```sh
python main.py --dataSet "CiteSeer"
```
You can run Cora, CiteSeer, photos and Computers dataset from dgl library, by using the following command. 
```sh
python main.py --dataSet "dataset_name_dgl"
```
For example for CiteSeer dataset it would be like:
```sh
python main.py --dataSet "CiteSeer_dgl"
```
## Environment
- python=3.8
- scikit-learn==1.2.2
- scipy==1.10.1
- torchmetrics==0.11.4
- python-igraph==0.10.4
- powerlaw=1.4.6
- dgl==1.2a230610+cu117
- dglgo==0.0.2



## Cite
If you find the code useful, please cite our papers.
```sh
cite
```
