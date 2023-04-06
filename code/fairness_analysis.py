import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np

DATA_PATH = "../data/propublica_data_for_fairml.csv"
ATTRIBUTE_COLUMNS = ["Number_of_Priors", "score_factor", "Age_Above_FourtyFive", "Age_Below_TwentyFive", "African_American", "Asian", "Hispanic", "Native_American", "Other", "Female", "Misdemeanor"]
NUM_EPOCHS = 10
BATCH_SIZE = 32

def train(X, y, num_epochs=NUM_EPOCHS, batch_size = BATCH_SIZE):
    pass

def accuracy():
    pass


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    labels = df["Two_yr_Recidivism"]    # 0 for no, 1 for yes
    data = df[ATTRIBUTE_COLUMNS]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
    print(labels)