import pandas as pd

DATA_PATH = "../data/compas-scores-raw.csv"
ATTRIBUTE_COLUMNS = ["LastName", "FirstName", "MiddleName", "Sex_Code_Text", "Ethnic_Code_Text", "DateOfBirth", "AssessmentReason", "Language", "LegalStatus", "CustodyStatus", "MaritalStatus" ]



if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df = df.loc[df["DisplayText"] == "Risk of Recidivism"] #only keeping the Risk of Recidivism Cases
    labels = df["DecileScore"]
    data = df[ATTRIBUTE_COLUMNS]