from fastapi import FastAPI
import pickle, uvicorn, os
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import accuracy_score
from typing import List, Literal

# Config & Setup
## Variables of environment
DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, "asset")
ml_comp_pkl = os.path.join(ASSETSDIRPATH, "ml_comp.pkl")

print(
    f" {'*'*10} Config {'*'*10}\n INFO: DIRPATH = {DIRPATH} \n INFO: ASSETSDIRPATH = {ASSETSDIRPATH} "
)


# API Basic config
app = FastAPI(
    title="Titanic Survivors API",
    version="0.0.1",
    description="Prediction of Titanic Survivors",
)

## Loading of assets
with open(ml_comp_pkl, "rb") as f:
    loaded_items = pickle.load(f)
#print("INFO:    Loaded assets:", loaded_items)

pipeline_of_my_model = loaded_items["pipeline"]
num_cols = loaded_items['numeric_columns']
cat_cols = loaded_items['categorical_columns']

## BaseModel
class ModelInput(BaseModel):
    PeopleInTicket: int
    Age: float
    FarePerPerson: float
    SibSp: int
    Pclass: int
    Fare: float
    Parch: int
    TicketNumber: float
    Embarked: Literal["S", "C", "Q"]
    Sex: Literal["male", "female"]
    Title: Literal["Mr", "Mrs", "Miss", "Master", "FemaleChild", "Royalty", "Officer"]



def make_prediction(
     Pclass, Sex, Age, SibSp,Parch, Fare, Embarked,Title, PeopleInTicket, FarePerPerson,TicketNumber
    
):

    data = {
        "PeopleInTicket": PeopleInTicket,
        "Age": Age,
        "FarePerPerson": FarePerPerson,
        "SibSp": SibSp,
        "Pclass": Pclass,
        "Fare": Fare,
        "Parch": Parch,
        "TicketNumber": TicketNumber,
        "Embarked": Embarked,
        "Title": Title,
        "Sex": Sex,
    }
    
        
    df = pd.DataFrame([data])
    target_idx = {
        0: "deceased",
        1: "survived",
    }

    X= df
    pred =pipeline_of_my_model.predict_proba(X).tolist()
    pred_class = int(np.argmax(pred[0]))
    output = {
        "predicted_class": pred_class,
        "prediction_explanation": target_idx[pred_class],
        "confidence_probability": float(pred[0][pred_class]),
    }

    return output
    
  
## Endpoints
@app.post("/Titanic")
async def predict(input: ModelInput):
    """__descr__
    --details---
    """
    output_pred = make_prediction(
        PeopleInTicket =input.PeopleInTicket,
        Age =input.Age,
        FarePerPerson =input.FarePerPerson,
        SibSp =input.SibSp,
        Pclass =input.Pclass,
        Fare =input.Fare,
        Parch =input.Parch,
        TicketNumber=input.TicketNumber,
        Title = input.Title,
        Embarked =input.Embarked,
        Sex=input.Sex,
    )

    
    return {
        "prediction": output_pred,
        "input": input
    }


# Execution

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        reload=True,
    )
