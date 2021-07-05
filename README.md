# Student and Affiliation Information
Affiliation: University of Amsterdam

Students: 
   * Lucas Fijen (10813268), lucas.fijen@gmail.com
   * Dante Niewenhuis (11058595), d.niewenhuis@hotmail.com 
   * Jonathan Mitnik (10911197), jmitnik@gmail.com 
   * Pieter de Marez Oyens (10002403), oyenspieter@gmail.com

TA: Simon Passenheim, :	simon.passenheim@gmail.com

# Debiasing: Mitigating Algorithmic Bias

<!-- TODO: Add final paper name -->
<!-- TODO: Add proper Markdown citation ? -->
This repository contains the code for the experiment described in our report, 'Debiasing: Mitigating Algorithmic Bias'. This experiment mainly attempts to reproduce the 'Uncovering and Mitigating Algorithmic Bias through Learned Latent Structure' paper by Amini et al, and successfully manages to find the original results.

## Installation

Clone this repository using `git clone https://github.com/JMitnik/FACT.git`.

It is recommended to run the code and installation in the '/code' folder, keeping this as the root.

### Dependencies
<!-- Rename the code directory to our source-code directory: debias -->
To install the necessary dependencies, one requirement is to have conda installed (tested on conda 4.8.2). In the code folder, install all necessary requirements by running the following command in your terminal:

If the environment is Linux:

```bash
# Create an environment
conda env create -f cuda_environment.yml
# Activate your environment
conda activate Msc_AI_FACT_Cuda
```

If the environment is Mac:
```bash
# Create an environment
conda env create -f environment.yml
# Activate your environment
conda activate Msc_AI_FACT
```


If the environment is Windows:
```bash
# Create an environment
conda env create -f windows_environment.yml
# Activate your environment
conda activate Msc_AI_FACT_Windows
```

### Getting the dataset

#### Automatic download

Inside the code directory, run the following code in your terminal:

```bash
python download.py
```

This will by default download the data into code/data.

❗The download size is approximately 1.3 GB.

#### Manual download

The 'h5' dataset for training can be downloaded from [this url](https://www.dropbox.com/s/l5iqduhe0gwxumq/train_face.h5?dl=1) and the eval dataset can be downloaded from [this url](https://www.dropbox.com/s/l0lp6qxeplumouf/PPB.tar?dl=1).

❗It is recommended to put the training dataset h5-file in 'data/h5_train', and the evaluation dataset (so that metadata file and the imgs folder are in this directory)  in 'data/ppb', resulting in 'data/ppb/*-metadata.csv' and 'data/ppb/imgs/*.jpg'.

<!-- TODO: Add manual dataset -->

---

## Running the code via the main.ipynb notebook

To run the notebook, run `jupyter notebook` from the /code folder (as root).

As a user, you can play around with the notebook via the *Config* parameters. Initializing a new config with the Config as defined in `code/setup.py`  will instantiate a number of default parameters, and by overriding the default values, the config can be used for various goals:

* Different parameters for the experimental setup, such as z-dim
* Load different models
* Batch-size for training

The notebook consists of **3 main parts**:
* Training a new model
* Evaluation of a model based on its path (or optionally pass a model to the evaluator init)
* Our *own final results* in this experimental setup.

## Running the code via main.py

Main.py can be run with a majority of the flags (see the Config parameters).

Some of the most important ones:
* run_mode: Running 'training' / 'evaluation' / 'both'
* debias_type: Which method to use for debiassing 'max' / 'max5' / 'gaussian'
* path_to_model: path to a model's root directory (from code/results as root).

## Config parameters
| Parameters               | Type          | Default value                      | Description                                        | Flag (--) |
|--------------------------|---------------|------------------------------------|----------------------------------------------------|-----------|
| run_mode                 | str           | 'both'                             | Mode to run main.py in (train/eval/both)           | [x]       |
| path_to_celeba_images    | str           | 'data/celeba/images'               | Path to separate CelebA images used for training   |           |
| path_to_celeba_bbox_file | str           | 'data/celeba/list_bbox_celeba.txt' | Path to separate CelebA bbox used for training     |           |
| path_to_imagenet_images  | str           | 'data/imagenet'                    | Path to separate Imagenet folder used for training |           |
| path_to_eval_face_images | str           | 'data/ppb/PPB-2017/imgs'           | Path to PPB folder used for evaluation             |           |
| path_to_eval_metadata    | str           | 'data/ppb/PPB-2017-metadata.csv'   | Path to PPB evaluation                             |           |
| path_to_model            | Optional[str] |                                    | Path to stored model                               | [x]       |
| path_to_h5_train         | str           | 'data/h5_train/train_face.h5'      | Path to h5                                         |           |
| debias_type              | str           | 'none'                             | Type of debiasing used                             | [x]       |
| model_name               | str           | 'model.pt'                         | name of the model to evaluate                      |           |
| random_seed              | int           | 0                                  | Random seed for reproducability                    |           |
| device                   | torch.device  | cpu                                | Device to use                                      |           |
| run_folder               | str           | *                                  | Folder name of the run (flag = folder_name)        | [x]       |
| eval_name                | Optional[str] |                                    | eval file name                                     | [x]       |
| batch_size               | int           | 256                                | Batch size                                         | [x]       |
| num_bins                 | int           | 10                                 | Number of bins                                     | [x]       |
| epochs                   | int           | 50                                 | Epochs                                             | [x]       |
| zdim                     | int           | 200                                | Z dimension                                        | [x]       |
| alpha                    | float         | 0.01                               | Alpha value                                        | [x]       |
| stride                   | float         | 0.2                                | stride used for evaluation windows                 | [x]       |
| dataset_size             | int           | -1                                 | Dataset size                                       | [x]       |
| eval_freq                | int           | 5                                  | Eval frequence                                     | [x]       |
| num_workers              | int           | 5                                  | Number workers for Pytorch data-loaders            | [x]       |
| debug_mode               | bool          | False                              | Debug mode                                         | [x]       |
| image_size               | int           | 64                                 | Image size                                         |           |
| eval_nr_windows          | int           | 15                                 | Number windows evaluation                          |           |
| eval_min_size            | int           | 30                                 | Evaluation window minimum                          |           |
| eval_max_size            | int           | 64                                 | Evaluation window maximum                          |           |
| use_h5                   | bool          | False                              | Uses h5 instead of the imagenet files              | [x]       |
| debug_mode               | bool          | False                              | Debug mode prints several statistics               |           |
| eval_dataset             | str           | 'ppb'                              | Dataset for evaluation                             | [x]       |
| save_sub_images          | bool          | False                              | Images to save in debug                            |           |

