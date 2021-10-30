# Cassava-leaf-disease-classification
Computer Vision - Identify the type of disease present on a Cassava Leaf image.

As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields.

And the existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.

In this project, a dataset of 21,367 labeled images are used which is collected during a regular survey in Uganda by Makerere University. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI). This is in a format that most realistically represents what farmers would need to diagnose in real life.

Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

### Dataset

The dataset can be downloaded from [here](https://www.kaggle.com/c/cassava-leaf-disease-classification/data) .

### Get involved 

Start by installing [Python](https://www.python.org/) and [Git](https://git-scm.com/downloads) if you have not already.

Next, clone this project by opening a terminal and typing the following commands (do not type the first $ signs on each line, they just indicate that these are terminal commands):

```sh
$ git clone https://github.com/bprasad26/cassava-leaf-disease-classification.git 
$ cd cassava-leaf-disease-classification
```

Next, create a virtual environment:


```sh
# create virtual environment
$ python3 -m venv leaf_venv
# Activate the virtual environment
$ source leaf_venv/bin/activate
```

Install the required packages
```sh
# install packages from requirements.txt file
$ pip install -r requirements.txt
```

Create folder to store data
```sh
$ mkdir data
```
Download the data from the link given above. Extract it and put it in this folder.

deactivate the virtual environment, once done with your work
```sh
$ deactivate
```
to stay updated with this project
```sh
$ git pull
```

### Files 

* explore.py - exploratory data analysis
* helper.py - helper functions for various tasks
* model_dispatcher - models that will be used for training
* train.py - script for training the model

&nbsp;
