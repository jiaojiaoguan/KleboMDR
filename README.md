# KleboMDR

KleboMDR is a machine learning-based model (linear SVM) to predict multidrug resistance phenotypes in Klebsiella pneumoniae using the metatranscriptome sequencing data. We provided the training code and the features we used.

## Data
The data directory includes three files: the labels for each sample and the expression features for WGS and NGS data.

## Trained model
This directory includes the model trained using the SVM based on 29 samples and the selected key features.

## Code
We provided the code that implemented a nested cross-validation (CV) framework for feature importance ranking, model training, and hyperparameter tuning. If you have any questions, please contact the email: jiaojguan2-c@my.cityu.edu.hk
