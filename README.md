# Identification of Protein-nucleotide Binding Residues with Deep Multi-task and Multi-scale Learning

Accurate identification of protein-nucleotide binding residues is essential for functional annotation of proteins and drug discovery. Advancements in computational methods for predicting binding residues from protein sequences have significantly improved predictive accuracy. However, it remains a challenge for current methodologies to extract discriminative features and assimilate heterogeneous data from different nucleotide types. To address this, we introduce NucMoMTL, a novel predictor specifically designed for identifying protein-nucleotide binding residues. Specifically, NucMoMTL leverages pre-trained unsupervised language models for robust sequence embedding and utilizes deep multi-task and multi-scale learning within orthogonal constraints to extract shared representations, capitalizing on auxiliary information from diverse nucleotides. Evaluation of NucMoMTL on the benchmark datasets demonstrates that it outperforms state-of-the-art methods, achieving an average AUC and AUPRC of 0.956 and 0.541, respectively. NucMoMTL can be explored as a reliable computational tool for identifying protein-nucleotide binding residues and facilitating drug discovery and protein function prediction. 

# 1. Requirements
Python >= 3.10.6   
torch = 2.0.0   
pandas = 2.0.0   
scikit-learn = 1.2.2   
ProtTrans (ProtT5-XL-UniRef50 model)

# 2. Datasets
We provided a total of four benchmark datasets, namely Nuc-798 (proposed by Yu et al [1] in 2013), Nuc-849 (proposed by Chen et al. [2] in 2011), Nuc-1521 (proposed by our work in 2023), and Nuc-207 (proposed by our work in 2023). Among them, Nuc-798, Nuc-849, and Nuc-1521 each consist of five common nucleotide (ATP, ADP, AMP, GTP, GDP) binding proteins constructed at different times. Nuc-207 comprises five uncommon nucleotide (TMP, CTP, CMP, UTP, UMP) binding proteins.

# 3. How to use

## 3.1 Set up environment for ProtTrans
Set ProtTrans follow procedure from https://github.com/agemagician/ProtTrans/tree/master

## 3.2 Train and Test   

### 3.2.1 Extract features
Extract pLMs embedding: cd to the NucMoMTL/Feature_Extract dictionary, and run "python3 extract_prot.py", the pLMs embedding matrixs will be extracted to Dataset/prot_embedding folder.   

## 3.2.2 Train and test
cd to the NucMoMTL home dictionary.  
run "python3 NucMoMTL.py" for training and testing the model for ATP, ADP, AMP, GTP, GDP binding residues prediction.  
run "python3 NucMoMTL_Nuc207.py" for training and testing the model for TMP, CTP, CMP, UTP, UMP binding residues prediction.  
run "python3 NucMoMTL_Nuc207_1521.py" for training and testing the model for ATP, ADP, AMP, GTP, GDP, TMP, CTP, CMP, UTP, UMP binding residues prediction.  

## 3.3 Only For nucleotide binding residues prediction purpose

1. unzip the pre-trained model in pre_model folder；  
   pre_model  
   |--   S2MTLTT5Msl2ADDMTL_S2Nuc-CNN_M_scale_l2ADD_02__T5_1.pkl  
   |--   S2MTLTT5Msl2ADDMTL_S2Nuc-CNN_M_scale_l2ADD_02__T5_2.pkl  
   |--   S2MTLTT5Msl2ADDMTL_S2Nuc-CNN_M_scale_l2ADD_02__T5_3.pkl  
   |--   S2MTLTT5Msl2ADDMTL_S2Nuc-CNN_M_scale_l2ADD_02__T5_4.pkl  
   |--   S2MTLTT5Msl2ADDMTL_S2Nuc-CNN_M_scale_l2ADD_02__T5_5.pkl


   Alternatively, download the pre-trained models from http://pan.njust.edu.cn/#/link/yirwtnIWna3w3u2i7HdG
   
3. write your query sequence (once one sequence) in file with file name 'query.txt' like below：
   
   TTVAQILKAKPDSGRTIYTVTKNDFVYDAIKLMAEKGIGALLVVDGDDIAGIVTERDYARKVVLQERSSKATRVEEIMTAKVRYVEPSQSTDECMALMTEHRMRHLPVLDGGKLIGLISIGDLVKSVIADQQFTIS  

   and put the query.txt into customer_test folder.   
4. cd to the NucMoMTL home dictionary and run python3 predict.py ,the predicted results named "result.xlsx" will be saved in the customer_test folder.
    
