# Titanic-Project---Machine-Learning-from-Disaster

The sinking of the Titanic is one of the most infamous shipwrecks in history.
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this project, I built a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

# Data Overview
The data has been split into two groups:

training set (train.csv)
test set (test.csv)

The training set was used to build the machine learning model. For the training set, the outcome (also known as the “ground truth”) was provided for each passenger. Feature engineering was also used to create new features.

The test set was used to see how well the model performed on unseen data. For the test set, the ground truth for each passenger was not provided. For each passenger in the test set, the trained model was used to predict whether or not they survived the sinking of the Titanic.

# Data Dictionary
| Variable      | Definition:       | Key   |
| :------------ |:--------------- |:-----|
| Survival      | Survival     |0 = No, 1 = Yes|
| pclass        | Ticket Class     |1 = 1st, 2 = 2nd, 3 = 3rd|
| sex           | sex of passenger |male / female|
| Age           | Age in years |  |
|sibsp          |# of siblings / spouses aboard the Titanic|  |
|Parch          |# of parents / children aboard the Titanic|  |
| Fare          | Passenger fare   |  |
|Cabin          |Cabin Number|  |
| Embarked      | Port of Embarkation|C=Cherbourg/ Q=Queenstown/ S=Southampton|

**Variable Notes:**
- pclass: A proxy for socio-economic status (SES)
- 1st = Upper
- 2nd = Middle
- 3rd = Lower
- age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
- sibsp: The dataset defines family relations in this way...
- Sibling = brother, sister, stepbrother, stepsister
- Spouse = husband, wife (mistresses and fiancés were ignored)
- parch: The dataset defines family relations in this way...
- Parent = mother, father
- Child = daughter, son, stepdaughter, stepson
- Some children travelled only with a nanny, therefore parch=0 for them.


# Setup
Find the commands below to setup this project and execute it. You must install Python3 (<3.10)

For manual installation, you need to have [`Python3`](https://www.python.org/) on your system. Then you can clone this repo and being at the repo's `root :: friendly_API_for_Titanic_model> ...`  follow the steps below:

- Windows:
        
        python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

- Linux & MacOs:
        
        python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

Execution
To run this project, please type the command below being at the repo's root:

    uvicorn src.main:app

## Author:
[Gyimah Gideon](https://www.linkedin.com/in/gideon-gyimah-08268b243/) 
