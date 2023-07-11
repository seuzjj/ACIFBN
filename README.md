# CrossKT: Learning Asynchronous Common and Individual Functional Brain Network for AD Diagnosis(MICCAI 2023)



# overview

![image](framework.png)
This code is for the paper: Learning Asynchronous Common and Individual Functional Brain Network for AD Diagnosis. 
Experimental data sourced from ADNI datasets( https://adni.loni.usc.edu/).
 
# Quick start (Train and Test)
We provide the code for our model here (`Model.py`), and training and testing code（`main.py`). The specific model settings can be found in the article's documentation.
We have provided a sample of preprocessed fMRI participant data in the Data folder (randomly generated using Python). Please replace it with real data for the actual experiment.
Before training the model, please configure the dependencies in the `requirements.txt`. Setting and modifying experimental parameters in `config.yaml`.
You can run `main.py` to quickly train and test the program.




# Citation
If you use our method or any part of it in your research, please cite:

@inproceedings{BarsoumICMI2016,
    title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
    author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
    booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
    year={2016}
}
