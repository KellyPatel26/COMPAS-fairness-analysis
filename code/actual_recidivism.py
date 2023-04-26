from utils import loss_plot, acc_plot, plot_bar, get_base_dfs, get_compas_wrong_dfs, get_compas_gender_dfs, get_compas_race_dfs
from fairness_analysis import TripleLinearClassifier, SingleLinearClassifier, config_loss, config_optimizer, train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from explainer import Explainer
import torch
import numpy as np

_, df_final = get_base_dfs()

train_columns = [
    "juv_fel_count", "juv_misd_count", "juv_other_count",
    "priors_count", "african-american", "caucasian", "hispanic",
    "other", "asian", "native-american", "less25", "greater45",
    "25to45", "felony", "misdemeanor"
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
         title='Lime - All Data, Gender Removed.',
         save_name='lime_all_feat_no_gender_c.png')

feature_count, pos_neg_count = explainer.count_shap(X_test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - All Data, Gender Removed.',
         save_name='shap_all_feat_no_gender_c.png')

afr, cauc, his, oth, asi, nat = get_compas_race_dfs(df_final)

test = afr[train_columns]
test = scaler.transform(test)
test = torch.from_numpy(test.astype(np.float32))

feature_count, pos_neg_count = explainer.count_lime(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Lime - African-American Data, Gender Removed.',
         save_name='lime_afr_feat_no_gender_c.png')

feature_count, pos_neg_count = explainer.count_shap(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - African-American Data, Gender Removed.',
         save_name='shap_afr_feat_no_gender_c.png')

test = cauc[train_columns]
test = scaler.transform(test)
test = torch.from_numpy(test.astype(np.float32))

feature_count, pos_neg_count = explainer.count_lime(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Lime - Caucasian Data, Gender Removed.',
         save_name='lime_cauc_feat_no_gender_c.png')

feature_count, pos_neg_count = explainer.count_shap(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - Caucasian, Gender Removed.',
         save_name='shap_cauc_feat_no_gender_c.png')

test = his[train_columns]
test = scaler.transform(test)
test = torch.from_numpy(test.astype(np.float32))

feature_count, pos_neg_count = explainer.count_lime(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Lime - Hispanic Data, Gender Removed.',
         save_name='lime_his_feat_no_gender_c.png')

feature_count, pos_neg_count = explainer.count_shap(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - Hispanic, Gender Removed.',
         save_name='shap_his_feat_no_gender_c.png')


test = oth[train_columns]
test = scaler.transform(test)
test = torch.from_numpy(test.astype(np.float32))

feature_count, pos_neg_count = explainer.count_lime(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Lime - Other Data, Gender Removed.',
         save_name='lime_oth_feat_no_gender_c.png')

feature_count, pos_neg_count = explainer.count_shap(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - Other, Gender Removed.',
         save_name='shap_oth_feat_no_gender_c.png')

test = asi[train_columns]
test = scaler.transform(test)
test = torch.from_numpy(test.astype(np.float32))

feature_count, pos_neg_count = explainer.count_lime(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Lime - Asian Data, Gender Removed.',
         save_name='lime_asi_feat_no_gender_c.png')

feature_count, pos_neg_count = explainer.count_shap(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - Asian, Gender Removed.',
         save_name='shap_asi_feat_no_gender_c.png')

test = nat[train_columns]
test = scaler.transform(test)
test = torch.from_numpy(test.astype(np.float32))

feature_count, pos_neg_count = explainer.count_lime(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Lime - Native-American Data, Gender Removed.',
         save_name='lime_nat_feat_no_gender_c.png')

feature_count, pos_neg_count = explainer.count_shap(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - Native-American, Gender Removed.',
         save_name='shap_nat_feat_no_gender_c.png')

m, f = get_compas_gender_dfs(df_final)

test = m[train_columns]
test = scaler.transform(test)
test = torch.from_numpy(test.astype(np.float32))

feature_count, pos_neg_count = explainer.count_lime(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Lime - Male Data, Gender Removed.',
         save_name='lime_m_feat_no_gender_c.png')

feature_count, pos_neg_count = explainer.count_shap(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - Male Data, Gender Removed.',
         save_name='shap_m_feat_no_gender_c.png')

test = f[train_columns]
test = scaler.transform(test)
test = torch.from_numpy(test.astype(np.float32))

feature_count, pos_neg_count = explainer.count_lime(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Lime - Female Data, Gender Removed.',
         save_name='lime_f_feat_no_gender_c.png')

feature_count, pos_neg_count = explainer.count_shap(test,
                                                    4,
                                                    input_dimension)

plot_bar(x=np.arange(input_dimension),
         height=feature_count,
         pos_neg=pos_neg_count,
         labels=train_columns,
         title='Shapley - Female Data, Gender Removed.',
         save_name='shap_f_feat_no_gender_c.png')






