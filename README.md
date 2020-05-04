# Project2-Disaster-Response

## Summary

A data scientist needs to have a good understanding of data engineering techniques and know how to build pipelines that start with processing raw data. This data needs to be manipulated and cleaned before it is fed to a machine learning model.

Hence, the purpose of this project is to create ETL and machine learning pipelines. The outcome of these pipelines is a model that categorizes messages based on predefined categories. 

A web application is then created to host the model, where a user can type in a message and then see the category of that message.
The web application also hosts graphs that explore the dataset

## Installations

The project was completed using the Project Workspace IDE. There were two modifications on the environment to make the code work: These packages were added

* pandas==1.0.3
* scikit-learn==0.22.2

Use the following commands to install:

```
pip install pandas==1.0.3
```
Or

```
conda install pandas=1.0.3
```


```
pip install -U scikit-learn
```
Or

```
conda install scikit-learn=0.22.2
```

## File Descriptions

There are three folders that contain all the data and scripts needed for the project. They also contain output files generated from the scripts. Here are the files:

* data
  * process_data.py: script for ETL pipeline. It takes below 2 datasets, merges and cleans them, and loads them to a database
  * disaster_categories.csv: dataset of ids and categories
  * disaster_messages.csv: dataset of ids and messages and their genres
  * disaster_response.db: output of process_data.py


* models
  * train_classifier.py: loads data from disaster_response.db and trains ml model on them
  * model.joblib: saved model
  

* app
  * run.py: Server in which users can interact with model and look at data visualizations
  * templates
    * go.html: template for message classifications request
    * master.html: master template
    
* GridSearch.ipynb: used to find paramaters of optimal performance

## How to Run

All the files are ready to run the server. just type the following commands:

```
cd app
python run.py
```
The server will be running on 0.0.0.0:3001

The method of how to access the server when running from Project Workspace IDE is explained in Project Details.

To make your own output files, you must first begin with data folder. After cloning this project, do the following commands from project directory

```
cd data
python process_data.py disaster_messages.csv disaster_categories.csv DatabaseName.db
```
where DatabaseName is a name of your choosing. After that do this:

```
cd ../models
python train_classifier.py  ../data/DatabaseName.db ModelName.joblib
```
where ModelName is a name of your choosing. You must change model and database names in run.py if they are different.

Here is the code that needs change:

```
# load data
engine = create_engine('sqlite:///../data/disaster_response.db') # this must be changed
df = pd.read_sql_table('mescat', engine)


# load model
model = joblib.load("../models/model.joblib") # this must be changed
```

Then type the following commands:

```
cd ../app
python run.py
```

To clarify, GridSearchCV was used to find paramaters for optimal performance. However, it will not be found in scripts because it takes a long time to finish running. Hence, it is included in the jupyter notebook provided
