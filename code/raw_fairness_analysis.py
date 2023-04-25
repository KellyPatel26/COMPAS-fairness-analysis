import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os 
import torch
from fairness_analysis import TripleLinearClassifier, SingleLinearClassifier, config_loss, config_optimizer, train
from explainer import Explainer

DATA_PATH = "../data/compas-scores.csv"

def loss_plot(train_losses, test_losses):
    plt.plot(train_losses, label = 'train loss')
    plt.plot(test_losses, label = 'test loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.clf()
    
def acc_plot(train_acc, test_acc):
    plt.plot(train_acc, label = 'train accuracy')
    plt.plot(test_acc, label = 'test accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.clf()

date_cols = ["compas_screening_date","dob",
             "c_jail_in","c_jail_out","c_offense_date",
             "v_screening_date","screening_date",
             "vr_offense_date","r_jail_out","r_jail_in",
             "r_offense_date","c_arrest_date"]
df = pd.read_csv(DATA_PATH, parse_dates=date_cols)
nulls = df.isnull().mean()


column_filter = ["id", "juv_fel_count", "compas_screening_date", "c_offense_date",
                 "sex", "age", "age_cat", "race", "c_charge_degree",
                 "c_charge_desc", "days_b_screening_arrest",
                 "decile_score","is_recid","r_offense_date",
                 "c_case_number", "v_decile_score", "is_violent_recid",
                 "vr_offense_date", "score_text", "juv_misd_count", "juv_other_count",
                 "priors_count"]

df_final = df.loc[:, column_filter].copy()
df_final = df_final[df_final["c_case_number"] != "NaN"]
df_final = df_final.loc[(df_final["days_b_screening_arrest"] <30) & (df_final["days_b_screening_arrest"] > -30)]
df_final = df_final.loc[(df_final["is_recid"]!=-1) & (df["decile_score"]!=-1) & (df_final["v_decile_score"] !=-1)]


df_final["african-american"] = np.where(
        df_final["race"] == "African-American",1,0
    )
df_final["caucasian"] = np.where(
        df_final["race"] == "Caucasian",1,0
    )
df_final["hispanic"] = np.where(
        df_final["race"] == "Hispanic",1,0
    )
df_final["other"] = np.where(
        df_final["race"] == "Other",1,0
    )
df_final["asian"] = np.where(
        df_final["race"] == "Asian",1,0
    )
df_final["native-american"] = np.where(
        df_final["race"] == "Native American",1,0
    )

race_type = pd.CategoricalDtype(categories=['African-American','Caucasian','Hispanic',
                                            "Other",'Asian','Native American'],ordered=True)
df_final["race"] = df_final["race"].astype(race_type)

score_type = pd.CategoricalDtype(categories=["Low","Medium","High"],ordered=True)
df_final["score_text"] = df_final["score_text"].astype(score_type)

age_type = pd.CategoricalDtype(categories=["Less than 25","25 - 45","Greater than 45"],ordered=True)
df_final["age_cat"] = df_final["age_cat"].astype(age_type)

df_final["less25"] = np.where(
    df_final["age_cat"] == "Less than 25", 1, 0
)
df_final["25to45"] = np.where(
    df_final["age_cat"] == "25 - 45", 1, 0
)
df_final["greater45"] = np.where(
    df_final["age_cat"] == "Greater than 45", 1, 0
)

df_final["male"] = np.where(
    df_final["sex"] == "Male", 1, 0
)
df_final["female"] = np.where(
    df_final["sex"] == "Female", 1, 0
)

df_final["misdemeanor"] = np.where(
    df_final["c_charge_degree"] == "M", 1, 0
)
df_final["felony"] = np.where(
    df_final["c_charge_degree"] == "F", 1, 0
)

print(df_final["c_charge_degree"])
for col in ["sex","c_charge_degree"]:
    df_final[col] = df_final[col].astype("category")

# exclude traffic tickets & municipal ordinance violations
df_final = df_final[df_final["c_charge_degree"] != "O"]
df_final = df_final[df_final["score_text"] != "NaN"]

df_final = df_final[df_final["c_offense_date"] < df_final["compas_screening_date"]]

# check if person reoffended within 2 years
def two_years(col,col_recid):
    # first we subtract the time columns
    df_final["days"] = df_final[col] - df_final["compas_screening_date"]
    # as it returns a time delta we convert it to int with .days
    df_final["days"] = df_final["days"].apply(lambda x:x.days)
    
    # then we assign the values 0,1 and 3 with np.where ( two years are 730 days )
    df_final["two"] = np.where(df_final[col_recid]==0,0,
                np.where((df_final[col_recid]==1) & (df_final["days"] < 730),1,3))
    
    return df_final["two"]
    
# check if person recided
df_final["two_years_r"] = two_years("r_offense_date","is_recid")
# check if person recided violentley
df_final["two_years_v"] = two_years("vr_offense_date","is_violent_recid")

df_final_c = df_final[df_final["two_years_r"] !=3].copy()
df_final_v = df_final[df_final["two_years_v"] != 3].copy()

# binarise decile scores
df_final_c["binary_decile_score"] = np.where(df_final_c["decile_score"] >=5,1,0)
df_final_v["binary_v_decile_score"] = np.where(df_final_v["v_decile_score"] >=5,1,0)

df_final_c.reset_index(drop=True,inplace=True)
df_final_v.reset_index(drop=True,inplace=True)

train_columns = [
    "juv_fel_count", "juv_misd_count", "juv_other_count",
    "priors_count", "african-american", "caucasian", "hispanic",
    "other", "asian", "native-american", "less25", "greater45",
    "25to45", "male", "female", "felony", "misdemeanor"
]

data = df_final_c[train_columns]
labels = df_final_c["binary_decile_score"]
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)
_, input_dimension = X_train.shape
model = TripleLinearClassifier(input_dim=input_dimension)

loss = config_loss()
optimizer = config_optimizer(model)
train_losses, test_losses, train_acc, test_acc = train(model, loss, optimizer, X_train, y_train, X_test, y_test)
loss_plot(train_losses=train_losses, test_losses=test_losses)
acc_plot(train_acc=train_acc, test_acc=test_acc)

explainer = Explainer(model)

lime = explainer.lime(X_test[-8].unsqueeze(0))
shapley = explainer.shapley(X_test[-8].unsqueeze(0))

explainer.plot([lime, shapley],
                ['lime', 'shapley'],
                train_columns)

