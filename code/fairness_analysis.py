import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DATA_PATH = "../data/propublica_data_for_fairml.csv"
ATTRIBUTE_COLUMNS = ["Number_of_Priors", "score_factor", "Age_Above_FourtyFive", "Age_Below_TwentyFive", "African_American", "Asian", "Hispanic", "Native_American", "Other", "Female", "Misdemeanor"]
NUM_EPOCHS = 500

class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
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

def loss_plot(train_losses, test_losses):
    plt.plot(train_losses, label = 'train loss')
    plt.plot(test_losses, label = 'test loss')
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

    model = LinearClassifier(input_dim=input_dimension)
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
    y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)

    loss = config_loss()
    optimizer = config_optimizer(model)
    train_losses, test_losses = train(model, loss, optimizer, X_train, y_train, X_test, y_test)

    loss_plot(train_losses=train_losses, test_losses=test_losses)


  

    

    


