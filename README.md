# NucMoMTL
Identification of Protein-nucleotide Binding Residues Using Deep Multi-task Learning Based on Multi-scale Convolutional Neural Network

Accurate identification of protein-nucleotide binding residues is essential for functional annotation of proteins and drug discovery. Advancements in computational methods for predicting binding residues from protein sequences have considerably improved predictive accuracy. However, it remains a challenge for current methodologies to extract discriminative features and assimilate heterogeneous data from different nucleotide types. To address this, we introduce NucMoMTL, a novel predictor specifically designed for identifying protein-nucleotide binding residues. Specifically, NucMoMTL leverages pre-trained unsupervised language models for robust sequence embedding and utilizes deep multi-task learning and multi-scale learning within orthogonal constraints to extract shared representations, capitalizing on auxiliary information from diverse nucleotides. Evaluation of NucMoMTL on the benchmark datasets demonstrates that it outperforms state-of-the-art methods, achieving an average AUC and AUPRC of 0.956 and 0.540, respectively. NucMoMTL can be explored as a reliable computational tool for identifying protein-nucleotide binding residues and facilitating drug discovery and protein function prediction. 

# 2. Requirements
Python >= 3.10.6

torch = 2.0.0

pandas = 2.0.0

scikit-learn = 1.2.2

ProtTrans (ProtT5-XL-UniRef50 model)

# 3. How to Use
## 3.1Set up environment for ProtTrans
Set ProtTrans follow procedure from https://github.com/agemagician/ProtTrans/tree/master
## 3.2 Extract features
Extract pLMs embedding: cd to the NucMoMTL/Feature_Extract dictionary, 
and run "python3 extract_prot.py", the pLMs embedding matrixs will be extracted to Dataset/prot_embedding fold.
## 3.3 Train and Test
cd to the NucMoMTL home dictionary,and run "python3 NucMoMTL.py" for training and testing the model.

## 3.4 Only For Prediction purpose
1. unzip the pre-trained model in model fold；
2. write your query sequence (once one sequence) in file with file name 'quert.txt' like below：
   4FRY_A_AMP TTVAQILKAKPDSGRTIYTVTKNDFVYDAIKLMAEKGIGALLVVDGDDIAGIVTERDYARKVVLQERSSKATRVEEIMTAKVRYVEPSQSTDECMALMTEHRMRHLPVLDGGKLIGLISIGDLVKSVIADQQFTIS
   4FRY_A_AMP 0000000000000000000000000000000000000000000000000001010110000000000000000000001011100000000000000000001110100000000000000000000000000000
3. run python3 predcitpy
    
