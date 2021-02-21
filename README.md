# Disaster Response Pipeline Project

### Summary of the Project  
Disaster Response Pipeline Project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages. Machine learning pipeline was created to categorize events so that people can send the messages to an appropriate disaster relief agency. A web app where an emergency worker can input a new message and get classification results in several categories in included in the project. The web app will also display visualizations of the data. 


### Instructions on How to Run Python Scripts and Web App:  
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to the web app (instruction here only applies to Udacity workspace)
    
    - Type `env|grep WORK` in terminal, you'll see outputs for SPACEID and SPACEDOMAIN.  
    - In a new web browser window, type `https://SPACEID-3001.SPACEDOMAIN`   

### Files in the Repository   

    - app
    | - template
    | |- master.html  # main page of web app
    | |- go.html  # classification result page of web app
    |- run.py  # Flask file that runs app

    - data
    |- disaster_categories.csv  # data to process 
    |- disaster_messages.csv  # data to process
    |- process_data.py  # a data cleaning pipeline
    |- InsertDatabaseName.db   # database for cleaned data

    - models
    |- train_classifier.py  # a machine learning pipeline
    |- classifier.pkl  # saved model 

    - README.md
