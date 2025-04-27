The repository contains example files and jupyter notebooks to train deep learning models on chip-seq data. 

First, clone the repository at https://github.com/de-Boer-Lab/random-promoter-dream-challenge-2022

The model I was using for a 2-head task (for example two TFs FLI1 and PU1 in cell state CMP) is present in 
benchmarks/drosophila 

Please check prixfixe for details about the model and all abstract classes used in the code to train the model. 
Depending on your data, make necessary changes in dataset.py, final_layers_block.py, and utils.py 


This is the list of dependencies required to create an environment. 
(This was added as a response to an environment issue by the author. I have been using the same environment. )

you can try manually creating an environment without directly using the yml file. The following works on linux.
conda create -n test python=3.10 conda activate test conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=11.8 -c pytorch -c nvidia pip install jupyterlab pip install pandas pip install tqdm pip install scikit-learn pip install biopython pip install torchinfo

testing.ipynb contains a Jupyter notebook to make binding predictions on the MPRA sequences, from the model trained on chip seq data. 

Data contains the first few columns for the test, train and Val set for prepared chip-seq data. Directory data_prep contains some of the scripts used to prepare the data inside repository /data

Please note that I used a different environment for visualisation since matplotlib wouldn’t work on the cluster. Visualisation can be found in density_plot_predvsactual.ipynb

training_dream.sh contains the slurm script for training the model. It is sufficiently commented, but for a jupyter notebook for block by block output, check M. Rafi’s original repository, specifically the file 
DREAMNets_buildModel_Train_Predict.ipynb inside benchmarks/drosophila

After running the script training_dream.sh (which contains the python file batch128lr01epoch10.py containing code for training the deep model ), the files and folders created are as follows: model_weight ( directory containing trained parameters), predictions.npz (numpy array with predictions and actual values on test set) and directory log_files (from cluster )
