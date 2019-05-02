# Project: Malaria Classifier App
## Capstone Project for Machine Learning Engineer Nanodegree
## Content: Convolutional Neural Network (CNN)

### Description

We could use Machine Learning to prevent or to detect early stages of a number of diseases. Therefore, I have created a classifier that could detect blood cells infected with Malaria parasites. This classifier could be very helpful in developing countries to diagnose the patients swiftly in order to avoid further organs complications or death.

### Install

Clone or download the project directory and create a new python environment.

To install the dependencies `libraries and modules` for the project, run the following command line in a new python environment:
```
pip install -r Requirements/installation.txt
```
You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

### Data

The dataset for this project needs to be downloaded from the US National Library of Medicine website:
https://ceb.nlm.nih.gov/repositories/malaria-datasets/ 

- Download the file `cell_images.zip`
- Create an empty folder named `Data` and a sub-folder name `Input-Data`
- In the `Input-Data` folder create 2 folders named `Healthy` and `Infected`
- Unzipped `cell_images.zip` and move the 2 folder named `Parasitized` and `Uninfected` into the `Data` folder
- Move 5 random images from the `Parasitized` folder into the `Input-Data/Infected` folder
- Rename the images as `Infected1.png, Infected2.png, … Infected5.png`
- Move 5 random images from the `Uninfected` folder into the `Input-Data/Healthy` folder
- Rename the images as `Healthy1.png, Healthy2.png, … Healthy5.png`

### Code

The classifier was  assemble in the `malaria_app.ipynb` notebook file. The python files `train.py` and `predict.py` are to be run on the command line as part of the application.

### Run

In a terminal or command line window, navigate to the top-level project directory `Malaria-Classifier` (that contains this README) and run one of the following:

To open the Jupyter Notebook software and project file in your browser:

```
jupyter notebook malaria.ipynb
```

To run the application, follow the python commands:

For training the classifier from the scratch

```
python train.py --save_dir <to save the checkpoint> --dataset_dir <path to the dataset images> --epochs <# epochs to train>
```

For predicting a blood cell image
```
python predict.py --input <path to a cell blood image> --checkpoint <path to checkpoint file loading pre-trained model>
```

### Documentation

The project contains a proposal file `Proposal.pdf` and a project report file `project_report.pdf`


### Author

* **Carlos Mertens** - *Certificate Machine Learning Engineer*
