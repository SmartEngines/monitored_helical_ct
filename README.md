# Helical monitored reconstruction modeling experiments

This repository contains experimental code for modelling the monitored helical CT scanning protocol for reducing the radiation dose.

Open dataset [[COVID-CTset](https://github.com/mr7495/COVID-CTset)] and the [[pre-trained neural network (NN)](https://drive.google.com/drive/folders/1xdk-mCkxCDNwsMAk2SGv203rY1mrbnPB)] used for our experiments are published in [[here](https://pubmed.ncbi.nlm.nih.gov/33821166/)]. For all the experiments the NN labeled as `FPN-fold1` was used. All the data and pre-trained NNs should be downloaded in advance, and the corresponding paths should be initialized within the experimental scripts. One should also make sure to download the `CSV` folder within the `COVID-CTset` dataset containing the patient metadata.

## Code overview

The provided code covers two main experiments of the article: modeling per-slice predictions for construction of stopping rules (part 4 of the main article) and modeling full acquisition process within the framework of monitored reconstruction for different stopping rules (part 5 of the main article). For convenience each of the experiments is divided into `runner` and `workflow` parts. To conduct the experiment the `runner` part of the script should be executed with paths to the data and NN being initialized. The labels of data used in our experiments are specified in the corresponding `.csv` file.

1. `slices_experiment_runner.py`
2. `slices_experiment_workflow.py`
3. `slices_to_proces.csv`
4. `full_experiment_runner.py`
5. `full_experiment_workflow.py`
6. `patients_to_process.csv`
7. `full_experiment_results_process.ipynb`

The output of the first experiment contains a full history of NN predictions for a chosen layer represented by pairs of numbers (active projection count and the prediction of NN for a corresponding intermediate reconstruction of the layer). The results are located in `.csv` files. All the intermediate reconstructions are also saved in the output folder repeating the structure of the initial dataset.

The output of the second experiment is similar to the previous one and also contains a full history of NN predictions with a difference that all the layers are being processed for a chosen patient. The predictions here are represented by the layer index followed by a comma separated string of consecutive predictions within a `.csv` file. As previously all the intermediate reconstructions are saved in the output folder repeating the structure of the initial dataset.

The last script represents the jupyter notebook file implementing the loading of the produced `.csv` files and building the Table 3 of the main paper.

## Environment requirements

To run the scripts the following environment conditions should be met:

1. `Python 3.9` 
2. `Numpy`
3. `Pandas`
4. `OpenCV2`
5. `TensorFlow`
6. `Keras`
7. `Keras Retinanet`
8. `Astra-toolbox v.1.9.9 dev`

We also highly recommend using GPU setup, since 3D reconstruction experiments take a long time being performed on CPU.