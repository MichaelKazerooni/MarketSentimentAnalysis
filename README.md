## An InsightDataScience Fellow Project

## Sentiment Analysis on Financial Microblogs using Machine learning and DNN

## Setup
Clone repository and update python path
```
repo_name= MarketSentimentAnalysis 
username= MichaelKazerooni
git clone https://github.com/$username/$repo_name
cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
Create new development branch and switch onto it
```
branch_name=dev-readme_requisites-20180905 # Name of development branch, of the form 'dev-feature_name-date_of_creation'}}
git checkout -b $branch_name
```

## Initial Commit
Lets start with a blank slate: remove `.git` and re initialize the repo
```
cd $repo_name
rm -rf .git   
git init   
git status
```  
You'll see a list of file, these are files that git doesn't recognize. At this point, feel free to change the directory names to match your project. i.e. change the parent directory Insight_Project_Framework and the project directory Insight_Project_Framework:
Now commit these:
```
git add .
git commit -m "Initial commit"
git push origin $branch_name
```

## Requisites

- List all packages and software needed to build the environment
- This could include cloud command line tools (i.e. gsutil), package managers (i.e. conda), etc.
- You will need docker installed on your machine


## Build Environment
- After docker is installed on your machine, you can run the Dockerfile using:
  docker build -t <name>.
  This command creates a docker image which can be used to run the project.

  
  
  ## Run Inference
- In order to run the container created in the build environment, execute the command : docker run -p 5000:5000 <docker_name>
- If you want to train the model, simply run classifier_training.py (preferebly using a GPU-enabled machine)

