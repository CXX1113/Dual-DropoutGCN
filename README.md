# Dual-Dropout Graph Convolutional Network for Predicting Synthetic Lethality in Human Cancers


**Requirements**
1. Python == 3.7
2. PyTorch == 1.1.0
3. Numpy == 1.16.2
4. Pandas == 0.24.2
5. SciPy == 1.2


**Repository Structure**
- Dual-DropoutGCN/data/sl/:
    - ./List_Proteins_in_SL.txt: Contain all the genes involved in SL pairs in our dataset.
    - ./SL_Human - Raw Data.xlsx: Contain original data of SL pairs (including the source of data).
    - ./SL_Human_Approved.txt: Contain SL pairs processed from the "sl_human - raw data.xlsx" file, denoted as SynLethDB.
    - ./computational_pairs.txt: Contain the predicted SLs collected from the "SL_Human - Raw Data.xlsx" file, which will be used to remove them from "SL_Human_Approved.txt" file.
- Dual-DropoutGCN/ddgcn/:
    - ./preprocess.py: Load and preprocess data (e.g., normalization).
    - ./objective.py: Objective function (loss function) of DDGCN model.
    - ./model.py: Implementation of DDGCN model.
    - ./train.py: Train DDGCN. In train.py, if the global variable "NOT_COMPUTATIONAL = True", DDGCN will run on the SynLethDB-NonPred dataset, otherwise it will run on the SynLethDB dataset.
    - ./evaluate.py: Evaluate DDGCN.
    - ./case_study.py: Qualitative case studies.


**How to run our code**

run the following scripts within Dual-DropoutGCN/ddgcn/
1.  ./train.py  # Train and evaluate the DDGCN model.
2.  ./case_study.py  #  Train our DDGCN using all the SL pairs in SynLethDB and predict novel SLs from the unknown pairs.
