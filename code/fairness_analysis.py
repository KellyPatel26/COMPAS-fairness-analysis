import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from explainer import Explainer
# from raiwidgets import FairnessDashboard


DATA_PATH = "../data/propublica_data_for_fairml.csv"
ATTRIBUTE_COLUMNS = ["Number_of_Priors", "score_factor", "Age_Above_FourtyFive", "Age_Below_TwentyFive", "African_American", "Asian", "Hispanic", "Native_American", "Other", "Female", "Misdemeanor"]
SENSITVE_COLUMNS = ["African_American", "Asian", "Hispanic", "Native_American", "Other"]
NUM_EPOCHS = 500

class SingleLinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=18, output_dim=1):
        super(SingleLinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = f.relu(self.linear(x))
        return x

def config_loss():
    return torch.nn.BCEWithLogitsLoss()

def config_optimizer(model):
    return torch.optim.Adam(model.parameters())

def train(model, loss, optimizer, X_train, y_train, X_test, y_test):
    train_losses = []
    test_losses = []

    for epoch in range(NUM_EPOCHS):
        out = model(X_train)
        l = loss(out, y_train)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        test_out = model(X_test)
        test_l = loss(test_out, y_test)
        train_losses.append(l.item())
        test_losses.append(test_l.item())

        print("Epoch {}: Training loss: {}, Test loss: {}".format(epoch, l.item(), test_l.item()))
    return train_losses, test_losses


class TripleLinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=18, output_dim=1):
        super(TripleLinearClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 10)
        self.linear2 = torch.nn.Linear(10, 5)
        self.linear3 = torch.nn.Linear(5, output_dim)

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = f.relu(self.linear3(x))
        return x

def config_loss():
    return torch.nn.BCEWithLogitsLoss()

def config_optimizer(model):
    return torch.optim.Adam(model.parameters())

def train(model, loss, optimizer, X_train, y_train, X_test, y_test):
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

    for epoch in range(NUM_EPOCHS):
        out = model(X_train)
        l = loss(out, y_train)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        test_out = model(X_test)
        test_l = loss(test_out, y_test)
        train_losses.append(l.item())
        out_classes = torch.where(out > 0.5, 1, 0)
        train_acc.append(torch.sum(out_classes == y_train).item()/len(out))
        test_losses.append(test_l.item())
        test_classes = torch.where(test_out > 0.5, 1, 0)
        test_acc.append(torch.sum(test_classes == y_test).item()/len(test_out))

        print("Epoch {}: Training loss: {}, Test loss: {}\n Training Acc: {} Test Acc: {}"
              .format(epoch, l.item(), test_l.item(), train_acc[-1], test_acc[-1]))
    return train_losses, test_losses, train_acc, test_acc

def loss_plot(train_losses, test_losses):
    plt.plot(train_losses, label = 'train loss')
    plt.plot(test_losses, label = 'test loss')
    plt.legend()
    plt.show()
    
def acc_plot(train_acc, test_acc):
    plt.plot(train_acc, label = 'train accuracy')
    plt.plot(test_acc, label = 'test accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    labels = df["Two_yr_Recidivism"]    # 0 for no, 1 for yes
    data = df[ATTRIBUTE_COLUMNS]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    _, input_dimension = X_train.shape

    model = TripleLinearClassifier(input_dim=input_dimension)
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
    y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)

    loss = config_loss()
    optimizer = config_optimizer(model)
    train_losses, test_losses, train_acc, test_acc = train(model, loss, optimizer, X_train, y_train, X_test, y_test)
    loss_plot(train_losses=train_losses, test_losses=test_losses)

 
    # A_test = torch.tensor(data[SENSITVE_COLUMNS].values).float()
    # y_true = torch.tensor(labels.values).float()
    # print(A_test)
    # model2 = SingleLinearClassifier(input_dim=A_test.shape[1])

    # y_pred = model2(A_test).detach().squeeze()
    # print(np.isscalar(y_pred[0]), y_pred[0])

    # FairnessDashboard(sensitive_features=A_test, 
    #                   y_true=y_true.tolist(),
    #                   y_pred=y_pred.tolist())
    # explainer = Explainer(model)
    # print(X_test[-1].shape)
    # print(y_test)
    # y_test_ints = y_test.type(torch.int64)
    # print(model(X_test[-1]))
    # print(explainer.lime(X_test[-1].unsqueeze(0)))




  

    

    


