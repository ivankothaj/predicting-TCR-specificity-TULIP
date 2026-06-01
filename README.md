# Predicting T Cell Receptor Specificity Using TULIP

## Introduction
This repository contains modified code of TULIP model used for the scope of the bachelor thesis.
The original code can be found at:
```
https://github.com/barthelemymp/TULIP-TCR/
```

## Dependencies
We recommend setting up conda environment with python version 3.10.
Afterwards, it is necessary to install following libraries in the environment:
- ...



## Replicating our results
Follow these steps:
1. activate your conda environment using ***conda activate [environment_name]***
2. clone this repository using: ***git clone https://github.com/ivankothaj/predicting-TCR-specificity-TULIP***
3. run ***python extract_database.py***
4. run ***python predict.py --test_dir data/BATMAN_input_subset.csv***
5. You can observe your results located in data directory in a file called ***predict_results.csv***
6. Additionally, there should be present several figures displaying overall prediction accuracies
