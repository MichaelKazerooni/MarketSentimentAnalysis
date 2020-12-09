## Fine tuning the BERT model for sentiment classification
## Sentiment Analysis on Financial Microblogs using Machine learning and DNN

This repo contains code for implementing a financial news healine sentiment classifier using Machine learning and deep learning models. The market sentiment on news headlines have dramatic affects on the financial markets and can create great trading oportunities. Also,sentiment in the financial markets are very tricky to judge and news have different affects depending on the perspective they are being viewed. Therefore, this makes sentiment classification very difficult. <br/>
## Data
Two data sources were used. <br/>
1- Reuters financial news headlines <br/>
2- News headlines from kaggle (https://www.kaggle.com/kimo26/financial-news-sentiment)

Both data sources have been labeled by industry experts and contain valuable information.

## Application
The final result for this application was a flask based web application that was dockerized and deployed on AWS ECS. <br/>
You can visit the link below to evaluate the app.<br/>
 http://13.57.245.84:5000/
 
Below you can see a demo of the app <br/>
![Alt Text](https://github.com/MichaelKazerooni/MarketSentimentAnalysis/blob/master/sentiment%20analysis%20demo.gif)

## Setup
Clone repository and update python path
```
repo_name= MarketSentimentAnalysis 
username= MichaelKazerooni
git clone https://github.com/$username/$repo_name
cd $repo_name

Create new development branch and switch onto it

## Initial Commit
Lets start with a blank slate: remove `.git` and re initialize the repo
```
cd $repo_name
rm -rf .git   
git init   
git status
```  
You'll see a list of file, these are files that git doesn't recognize. At this point, feel free to change the directory names to match your project.
Now commit these:
```
git add .
git commit -m "Initial commit"
git push origin $branch_name
```

## Requisites
python>=3.7
Pytorch == 1.6
matplotlib==3.2.2
transformers==2.11.0
numpy==1.18.5
future == 0.18.2
flask == 1.1.2




## Build Environment
- After docker is installed on your machine, you can run the Dockerfile using:
  docker build -t <name>.
  This command creates a docker image which can be used to run the project.

  
  
## Run Inference
- In order to run the container created in the build environment, execute the command : docker run -p 5000:5000 <docker_name>
- If you want to train the BERT model, simply run classifier_training.py (preferebly using a GPU-enabled machine)
- To evaluate the BERT model, follow the instructions in the classifier_training.py file.
- To evaluate the traditional machine learning approaches, run the traditional_machine_learning.py file.
- The docker image runs using the fine-tuned Bert model.
- Running the docker image will pop up a server that can be reached by using 0.0.0.0:5000.



