# Disaster Response Pipeline

# Required Libraries for running this project
1. pandas
2. numpy
3. sklearn
4. nltk
5. sqlalchemy
6. pickle
7. flask (web app)
8. plotly (web app)

# Motivation for the project
This project is to help provide analysis for tweets to determine if there is help that needs to be sent in sponse to natural disasters. This project has huge relief accounts based on tweets from disaster zones. 

# How to run the Python scripts
This project first requires a cleaned db to be created before modeling. This is done by the following steps: 
1. Run the following command in the terminal after going to the WebApp folder. 
    1. python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv sqlite:///data/DisasterResponse.db
2. The parameters for this are as follows: 
    1. data/process_data.py
        1. This is the first raw csv required for running the dataset, and is included in the github repo. 
    2. data/disaster_categories.csv
        1. This is the other file that is the raw csv input for the process. 
    3. sqlite:///data/DisasterResponse.db
        1. This is the output sql database that is required for the project. 
3. After running this script, the first step of the process before running the web app will have been completed. You can move onto the second part of the process. Running the model portion of the code. 

After the database has been created, then the next step is to train the model that was created for this project: 
1. Run the following command in the terminal after going to WebApp folder. 
    1. python model/train_classifier.py data/DisasterResponse.db models/classifier.pkl
2. The parameters for the command in 1. are as follows: 
    1. data/DisasterResponse.db 
        1. The db location from the output of the last step. 
    2. models/classsifier.pki
        1. The filepath output for the model that is created in this step. 

After that, the main scripts should be completed, and the web app will be able to be ran, assuming the local environment is correct for running the web app. More on this in the "How to run the web app" section below. 

# How to run the web app
1. Naviage to 'WebApp/app/' folder. 
2. Run 'python run.py'
3. To view the web app at the following website:
   * http://0.0.0.0:3001/

# Files in the repo
### categories.csv
* This file is raw data that is required for running the ML_Pipeline.py file. 
### messages.csv
* This file is raw data that is required for running the ML_Pipeline.py file. 
# ML_Pipeline.py
* This python script is the combined file for running the back end of the web app. 
### WebApp/app/run.py
* This file is required for running the web app. 
### WebApp/app/templates/*
* The files in here are the html files for the web app. 
* Adjusting these files will change how the web app looks at the local website in the 'How to run the web app' section. 
### WebApp/data/disaster_categories.csv
* Raw data used for creating the models for the webapp
### WebApp/data/disaster_categories.csv
* Raw data used for creating the models for the webapp
### WebApp/data/DisasterResponse.db 
* The database that is outputted from running the process_data.py file. 
### WebApp/data/process_data.py
* Running this file will merge, clean and create the WebApp/data/DisasterResponse.db file. 
* The steps for doing this are outlined in the first section of the 'How to run the Python scripts' section.
### WebApp/model/train_classifier.py
* This python script takes the database created in the first section outlined in the 'How to run the Pyton scripts' section, and creates a trained model to a pickle file after the appropriate pipeline functions have been performed. 
* The steps for running this is outlined in the second part of the 'How to run the Python scripts' section.