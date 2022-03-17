# SNRI - Subgraph Neighboring Relations Infomax for Inductive Link Prediction on Knowledge Graphs

This is the code necessary to run experiments on SNRI.

## Requirements
    dgl==0.4.2
    lmdb==0.98
    networkx==2.4
    scikit-learn==0.22.1
    torch==1.4.0
    tqdm==4.43.0

## Inductive link prediction experiments

Train data and test data are located in `data` folder. We use WN18RR_v4 as a runninng example for illustrating the steps.

To train a SNRI model, run the following command:
`python train.py -d WN18RR_v4 -e snri_wn_v4 --MI_coef 5`

To test the SNIR model, run the following command:
`python test_auc.py -d WN18RR_v4_ind -e snri_wn_v4 --runs 5`
`python test_ranking.py -d WN18RR_v4_ind -e snri_wn_v4`
