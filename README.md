# Diabetes-Prediction
Diabetes is a chronic, metabolic disease characterized by elevated levels of blood glucose (or blood sugar), which leads over time to serious damage to the heart, blood vessels, eyes, kidneys and nerves. 
It is estimated that 537 million people are currently living with diabetes all over the world. 
 Early detection of diabetes lowers medical expenses and the possibility of developing more severe health issues.

 In this project, I am trying to develop a model to predict the possibility of a person being diabetic from certain general health and economic parameters so that they can try to make changes
 to their lifestyle to prevent developing serious health issues.

 ## Database

 I Used Diabetes Health Indicators Dataset from kaggle which is a cleaned subset of Behavioral Risk Factor Surveillance System (BRFSS) 2015 Data. 
 The data set contains a balanced subset of BRFSS data.It is a clean dataset of 70,692 survey responses to the CDC's BRFSS2015. It has an equal 50-50 split of respondents 
 with no diabetes and with either prediabetes or diabetes.
21 feature variables relevant to Diabetes prediction has been extracted from the main BRFSS data to make this database.
 
 The Behavioral Risk Factor Surveillance System (BRFSS) is a health-related telephone survey that is collected annually by the CDC. Each year,
 the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. 
 The original dataset contains responses from 441,455 individuals and has 330 features. These features are either questions directly asked of participants,
 or calculated variables based on individual participant responses.

Kaggle dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv 

It is also available on this repository in file : **diabetes_binary_5050split_BRFSS2015.csv**

### Database cleaning and preparation
The features used in the database are mostly binary categorical variables like Smoker, Stroke, AnyHealthcare etc. Also numeric variables like age has been already converted to a categorical value by 
grouping different age groups and assigning a number to each group. So this means extracting a subset of features has lead to mzny feature rows being duplicates of one another. Also some
rows had unusually high values of BMI. The duplicate rows and unusual BMI rows have been removed to create a cleaner dataset. The details of cleaning , model preparation and evaluation can be found in 
the jupyter notebook attached to the repository **notebook.ipynb**

Cleaned data is available in repository as **cleaned_data.csv**


### Model selection and training
I tried out basic machine learning models like logistic regression, decision tree, random forest ang XGBoost on the model. Finally decided on XGBoost as it gives the most AUC value (0.8133). The XGBoost model
has been trained on a fraction of data and tested on the rest (80 - 20 split). The final model has been saved in a pickle file. Training script is found on **train.py** and final model on **xgb_trained_model.bin**
file.

### Model deployment and containerization

I have deployed the diabetis prediction model as a flask app (**predict.py** file). Python dependencies are stored using pipenv package and corresponding pip lock files are included with the repository. Also 
added a Dockerfile for deploying the prediction service as a docker image.

### Steps for deployment

Navigate to code repo and build docker image using command:
```
docker build --rm -t zoomcamp-test . 
```
Next deploy the diabetis prediction service by running a container with image generated from previous step. Here the container is called zoomcamp-test test and service is deployed on localhost:9696

```
docker run -it --rm -p 9696:9696 zoomcamp-test
```

Next step is to send a sample request to the service. This can be done using an app like Postman or programatically using REST HTTP requests

Send a **POST** request to https://localhost:9696/predict with the request body as a persons health parameters in JSON format. 

**Sample request**
```
{
        "Age" :3.0,
        "BMI" :25.0,
        "DiffWalk": 0.0,
        "Education": 4.0,
        "GenHlth":1.0,
        "HeartDiseaseorAttack":0.0,
        "HighBP":1.0,
        "HighChol": 1.0,
        "Income":5,
        "PhysHlth":5
}
```
#### Request parameters
- Age: 

&nbsp;&nbsp;&nbsp;&nbsp;Calculated as a number by grouping different age groups into 5 year categories  
&nbsp;&nbsp;&nbsp;&nbsp; 1 = 18-24  
&nbsp;&nbsp;&nbsp;&nbsp; 2 = 25-29  
&nbsp;&nbsp;&nbsp;&nbsp; 3 = 30-34  
&nbsp;&nbsp;&nbsp;&nbsp; 4 = 35-39  
&nbsp;&nbsp;&nbsp;&nbsp; 5 = 40-44  
&nbsp;&nbsp;&nbsp;&nbsp; 6 = 45-49  
&nbsp;&nbsp;&nbsp;&nbsp; ....  
&nbsp;&nbsp;&nbsp;&nbsp; 13 = 80+

- BMI: Body mass Index of person
- DiffWalk: Does the person has difficulty walking (0-no 1-yes)
- Education:
  
&nbsp;&nbsp;&nbsp;&nbsp; 1 =  Never attended school or only kindergarten  
&nbsp;&nbsp;&nbsp;&nbsp; 2 = Grades 1 through 8 (Elementary)  
&nbsp;&nbsp;&nbsp;&nbsp; 3 = Grades 9 through 11 (Some high school)  
&nbsp;&nbsp;&nbsp;&nbsp; 4 = Grade 12 or GED (High school graduate)  
&nbsp;&nbsp;&nbsp;&nbsp; 5 = College 1 year to 3 years (Some college or technical school)  
&nbsp;&nbsp;&nbsp;&nbsp; 6 = College 4 years or more (College graduate)

- GenHlth: General health in scale 1-5 : 1 excellent 5 poor
- HeartDiseaseorAttack: Does the person have had a heart disease or heart attack before (0-no 1-yes)
- HighBP: Does the person has high blood pressure (0-no 1-yes)
- HighChol: Does the person has high cholestrol levels (0-no 1-yes)
- Income

  
&nbsp;&nbsp;&nbsp;&nbsp; 1 Less than $10,000  
&nbsp;&nbsp;&nbsp;&nbsp; 2 Less than $15,000 ($10,000 to less than $15,000)  
&nbsp;&nbsp;&nbsp;&nbsp; 3 Less than $20,000 ($15,000 to less than $20,000)  
&nbsp;&nbsp;&nbsp;&nbsp; 4 Less than $25,000 ($20,000 to less than $25,000)  
&nbsp;&nbsp;&nbsp;&nbsp; 5 Less than $35,000 ($25,000 to less than $35,000)  
&nbsp;&nbsp;&nbsp;&nbsp; 6 Less than $50,000 ($35,000 to less than $50,000)  
&nbsp;&nbsp;&nbsp;&nbsp; 7 Less than $75,000 ($50,000 to less than $75,000)  
&nbsp;&nbsp;&nbsp;&nbsp; 8 $75,000 or more 

- PhysHlth: For how many days during the past 30 days was the persons physical health not good


