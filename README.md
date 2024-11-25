# Midterm_project_ML

# Overview

Sleep disorders are a prevalent issue experienced by individuals across various demographics, with significant impacts on their quality of life and overall health. Increased stress responsivity, emotional distress, mood disorders, and long-term health consequences, such as cardiovascular disease and diabetes, can result from sleep disruption [^1]. Therefore understanding and analysing sleep patterns can provide valuable insights into improving health outcomes.

Hence for this midterm project, a synthetic dataset from Kaggle [^2] was utilized to analyse sleep disorders and their effects on daily life.

# About Dataset

## Dataset Overview:
The Sleep Health and Lifestyle Dataset comprises 400 rows and 13 columns, covering a wide range of variables related to sleep and daily habits. It includes details such as gender, age, occupation, sleep duration, quality of sleep, physical activity level, stress levels, BMI category, blood pressure, heart rate, daily steps, and the presence or absence of sleep disorders.

## Key Features of the Dataset:
- Comprehensive Sleep Metrics: Explore sleep duration, quality, and factors influencing sleep patterns.
- Lifestyle Factors: Analyze physical activity levels, stress levels, and BMI categories.
- Cardiovascular Health: Examine blood pressure and heart rate measurements.
- Sleep Disorder Analysis: Identify the occurrence of sleep disorders such as Insomnia and Sleep Apnea.

## Dataset Columns:

1. Person ID: An identifier for each individual.
2. Gender: The gender of the person (Male/Female).
3. Age: The age of the person in years.
4. Occupation: The occupation or profession of the person.
5. Sleep Duration (hours): The number of hours the person sleeps per day.
6. Quality of Sleep (scale: 1-10): A subjective rating of the quality of sleep.
7. Physical Activity Level (minutes/day): Daily minutes spent on physical activity.
8. Stress Level (scale: 1-10): A subjective rating of stress experienced.
9. BMI Category: The BMI category (e.g., Underweight, Normal, Overweight).
10. Blood Pressure (systolic/diastolic): Blood pressure measurement.
11. Heart Rate (bpm): Resting heart rate in beats per minute.
12. Daily Steps: Steps taken per day.
13. Sleep Disorder: Presence or absence of a sleep disorder (None, Insomnia, Sleep Apnea).

## Details about Sleep Disorder Column:

* None: The individual does not exhibit any specific sleep disorder.
* Insomnia: The individual experiences difficulty falling asleep or staying asleep, leading to inadequate or poor-quality sleep.
* Sleep Apnea: The individual suffers from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks.


# Project Components

This project includes several key components that are part of machine-learning-zoomcamp [^3] midterm project requirements (Evaluation Criteria):

- Problem Description
- Exploratory Data Analysis (EDA)
- Model Training
- Exporting Notebook to Script
- Model Deployment
- Reproducibility
- Dependency and Environment Management
- Containerization


# Dependency and Environment Management Guide

To ensure reproducibility and ease of setup across different environments:

- ```bash
  pipenv install
  
- ```bash
  pip shell
  
- ```bash
  pip install -r requirements.txt

or , If this is not possible, the needed packages can be installed by running the following:


- Create a Pipfile in your project directory with the specified Python version:
  ```bash
  pipenv --python 3.9.12
  
- To work within your virtual environment, activate it by running:
  ```bash
  pipenv shell
  
The terminal prompt should be seen to change, indicating that the virtual environment has been activated.

- Install Dependencies:
  ```bash
  pipenv install requests flask ruamel-yaml seaborn matplotlib scikit-learn==1.1.1 numpy==1.22.4

- Verify Installation:
   ```bash
   pipenv graph
   
-  If needed generate requirements.txt file
   ```bash
   pipenv requirements > requirements.txt

  
# Deployment Guide

## To Run Locally:

Run "predict.py" file in a terminal to start the Flask server.
  - ```bash
    python predict.py
    
Open another terminal window and run "predict-test.py" file.
  - ```bash
    python predict-test.py

## To Run with Docker:

- Download and run Docker Desktop from the official website: [Docker](https://www.docker.com/)
- Open the terminal of your project
- Build the Docker image:
  ```bash
   docker build -t sleep-disorder-predictor .

- Run the Docker container:
  ```bash
   docker run -p 9696:9696 sleep-disorder-predictor

The Flask application will be started inside a Docker container, and it will be accessible on port 9696.


- To test the application:
    
    - Ensure that the container is running.
    - Open another terminal window:
        ```bash
          python predict-test.py

This will send a request to your containerized application and print the response for you to see.

## References:
[^1]: Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC5449130/
[^2]: Source: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data
[^3]: Source: https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects

