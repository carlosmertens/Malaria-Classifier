# Project: Malaria Classifier App
## Content: Convolutional Neural Network (CNN)

### Description

Capstone Project for Machine Learning Engineer Nanodegree. 

### Install

Clone or download the project directory and create a new python environment.

Install the dependency libraries and modules for the project, run the following command line in a new python environment:
```
pip install -r Requirements/installation.txt
```
You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

### Data

The dataset for this project needs to be downloaded from the US National Library of Medicine website:
https://ceb.nlm.nih.gov/repositories/malaria-datasets/ 

- Download the file `cell_images.zip`
- Create an empty folder named `Data` and a sub-folder name `Input-Data`
- In the Input-Data folder create 2 folders named `Healthy` and `Infected`
- Unzipped cell_images.zip and move the 2 folder named `Parasitized` and `Uninfected` into the Data folder
- Move 5 random images from the Parasitized folder into the Input-Data/Infected folder
- Rename the images as `Infected1.png, Infected2.png, … Infected5.png`
- Move 5 random images from the Uninfected folder into the Input-Data/Healthy folder
- Rename the images as `Healthy1.png, Healthy2.png, … Healthy5.png`

### Code

Code was create in the `malaria_app.ipynb` notebook file. The python files `train.py` and `predict.py` are to be run on the command line as part of the application.

### Run

In a terminal or command line window, navigate to the top-level project directory `Malaria-Classifier` (that contains this README) and run one of the following:

To open the Jupyter Notebook software and project file in your browser:

```bash
jupyter notebook malaria.ipynb
```

To run the application, follow the python commands:

For training the model

```
python train.py --save_dir <directory to save the checkpoint> --dataset_dir <path to the dataset images directory> --epochs <epochs to train>
```

For predicting a blood cell image
```
python predict.py --input <path to a cell blood image> --checkpoint <path to checkpoint file loading pre-trained model>
```
### Documentation

The project contains a proposal file `Proposal.pdf` and a project report file `project_report.pdf`


### Authors

* **Carlos Mertens** - *Udacity Student* -

## Acknowledgments

* Udacity, Inc.
