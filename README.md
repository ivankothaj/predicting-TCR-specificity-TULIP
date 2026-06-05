# Predicting T Cell Receptor Specificity Using TULIP

## Introduction
- This repository contains modified code of TULIP model used for the scope of the bachelor thesis.
- The original code can be found at: https://github.com/barthelemymp/TULIP-TCR/

## Dependencies
We recommend setting up conda environment with python version 3.10:
- ```conda create -n <envName> python=3.10```

Install following libraries into your conda environment using the commands below:
- ```conda install conda-forge::transformers==4.32.1```
- ```conda install -c conda-forge numpy=1.26.4 pandas=2.1.4```
- ```conda install -c pytorch pytorch=2.1.2 cpuonly```
- ```conda install -c conda-forge openpyxl```
- ```pip install huggingface-hub==0.16.4```
- ```conda install -c conda-forge scikit-learn```
- ```conda install -c conda-forge matplotlib```

We checked these version and they worked on our machine. In case you decide to use different versions, their compatibility is not guaranteed.

## Replicating our results
1. Activate your conda environment using: ```conda activate [environment_name]```
2. Clone this repository using:
```
git clone https://github.com/ivankothaj/predicting-TCR-specificity-TULIP
```
3. Execute: ```python extract_datasets.py```
4. Then, execute: 

    ```python predict.py --test_dir data/BATCAVE_subset.csv``` 

    or 

    ```python predict.py --test_dir data/mutant_subset.csv``` 
    
    depending on which subset you want to run on.
5. Intermediate results are located in the data directory in a file called ***predict_results.csv***.
6. Additionally, the overall prediction performance on the ***BATCAVE_subset.csv*** are displayed by several figures in the data directory.
