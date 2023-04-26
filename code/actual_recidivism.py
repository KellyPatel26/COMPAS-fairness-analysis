from utils import loss_plot, acc_plot, plot_bar, get_base_dfs, get_compas_wrong_dfs
from fairness_analysis import TripleLinearClassifier, SingleLinearClassifier, config_loss, config_optimizer, train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from explainer import Explainer
import torch
import numpy as np

df_final, _ = get_base_dfs()

train_columns = [
    "juv_fel_count", "juv_misd_count", "juv_other_count",
    "priors_count", "african-american", "caucasian", "hispanic",
    "other", "asian", "native-american", "less25", "greater45",
    "25to45", "male", "female", "felony", "misdemeanor"
]

data = df_final[train_columns]
labels = df_final["two_years_r"]
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
model = SingleLinearClassifier(input_dim=input_dimension)

loss = config_loss()
optimizer = config_optimizer(model)
train_losses, test_losses, train_acc, test_acc = train(model, loss, optimizer, X_train, y_train, X_test, y_test)
loss_plot(train_losses=train_losses, test_losses=test_losses)
acc_plot(train_acc=train_acc, test_acc=test_acc)

explainer = Explainer(model)

feature_count, pos_neg_count = explainer.count_lime(X_test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Lime - All Features Included',
         save_name='lime_all_feat_c.png')

feature_count, pos_neg_count = explainer.count_shap(X_test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - All Features Included',
         save_name='shap_all_feat_c.png')


