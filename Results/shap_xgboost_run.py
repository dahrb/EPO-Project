import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from Classes import TextProcess #PatentExtract
from Classes import Experiments
import numpy as np

#save arrays and shap for local interpretation if it works


X_train_2_pf = pd.read_pickle('X_Train_2_pf')
y_train_2_pf = pd.read_pickle('y_Train_2_pf')

X_test_2_pf2019_2022 = pd.read_pickle('X_test_2_pf2019_2022')
y_test_2_pf2019_2022 = pd.read_pickle('y_test_2_pf2019_2022')

print('read in')

# Patent Ref Exp 2
steps_pf_2 = Pipeline([('pre_process',TextProcess(stopwords= False, numbers= False, lemmatisation= False)),('vect', TfidfVectorizer(tokenizer=lambda x:x,preprocessor=lambda x:x, use_idf=False, norm='l2', ngram_range=(1,4),min_df=2)), ('clf', xgb.XGBClassifier(random_state=42, objective='binary:logistic', n_estimators=300, learning_rate=0.05, gamma=0.2))])

exp = Experiments(opposition=False)
exp.training_helper_preprocess(X_train_2_pf)

X_train_2_pf = exp.X_train
y_train_2_pf = y_train_2_pf[0].to_numpy()

steps_pf_2.fit(X_train_2_pf, y_train_2_pf)

print('model done')

train_2 = steps_pf_2['pre_process'].transform(X_train_2_pf)
train_2 = steps_pf_2['vect'].transform(train_2)

#X_test_2_pf2019_2022
exp = Experiments(opposition=False)
exp.training_helper_preprocess(X_test_2_pf2019_2022)

X_test_2_pf2019_2022 = exp.X_train
y_test_2_pf2019_2022 = y_test_2_pf2019_2022[0].to_numpy()

test_2 = steps_pf_2['pre_process'].transform(X_test_2_pf2019_2022)
test_2 = steps_pf_2['vect'].transform(test_2)

print('start array')

train_2_array = pd.DataFrame(train_2.toarray(), columns = steps_pf_2['vect'].get_feature_names_out())
test_2_array = pd.DataFrame(test_2.toarray(), columns = steps_pf_2['vect'].get_feature_names_out())

print('created arrays')

explainer = shap.TreeExplainer(steps_pf_2['clf'], feature_names = steps_pf_2['vect'].get_feature_names_out())
shap_values = explainer(train_2)
shap_values_test = explainer(test_2)

print('shap done')

shap.summary_plot(shap_values, #Use Shap values array
                features=train_2_array, # Use training set features
                feature_names =steps_pf_2['vect'].get_feature_names_out(),
                plot_type = 'dot',
                show=False,
                color_bar = True,
                max_display = 20,
)

plt.savefig("shap_train.jpg")

plt.clf()

#global importance - test set

shap.summary_plot(shap_values_test, #Use Shap values array
                features=test_2_array, # Use training set features
                feature_names =steps_pf_2['vect'].get_feature_names_out(),
                plot_type = 'dot',
                show=False,
                color_bar = True,
                max_display = 20,
)

plt.savefig('shap_test.jpg')

plt.clf()

y_pred = steps_pf_2['clf'].predict(test_2)
misclassified = np.where(y_test_2_pf2019_2022 != y_pred)
correct = np.where(y_test_2_pf2019_2022 == y_pred)
correct_affirmed = np.where((y_test_2_pf2019_2022 == y_pred)&(y_test_2_pf2019_2022==1))
np.random.seed(42)
correct_case  = np.random.choice(correct[0])
correct_affirm_case = np.random.choice(correct_affirmed[0])
incorrect_case  = np.random.choice(misclassified[0])

#correct reversed
print("CORRECT OUTCOME: ",y_test_2_pf2019_2022[correct_case])
print("PREDICTED OUTCOME: ",y_pred[correct_case])

correct_force_plot = shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_values_test.values[correct_case], 
                features=test_2_array.loc[correct_case],
                feature_names=steps_pf_2['vect'].get_feature_names_out(),
                show=False)

file ='shap1.html'
shap.save_html(file, correct_force_plot)

#correct affirmed
print("CORRECT OUTCOME: ",y_test_2_pf2019_2022[correct_affirm_case])
print("PREDICTED OUTCOME: ",y_pred[correct_affirm_case])

correct_affirm_force_plot = shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_values_test.values[correct_affirm_case], 
                features=test_2_array.loc[correct_affirm_case],
                feature_names=steps_pf_2['vect'].get_feature_names_out(),
                show=False)

file ='shap2.html'
shap.save_html(file, correct_affirm_force_plot)


#Reverse case - predicted as Affirmed

print("CORRECT OUTCOME: ",y_test_2_pf2019_2022[incorrect_case])
print("PREDICTED OUTCOME: ",y_pred[incorrect_case])

incorrect_force_plot = shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_values_test.values[incorrect_case], 
                features=test_2_array.loc[incorrect_case],
                feature_names=steps_pf_2['vect'].get_feature_names_out(),
                show=False)


file ='shap3.html'
shap.save_html(file, incorrect_force_plot)

#Affirmed Case - predicted reverse
print("CORRECT OUTCOME: ",y_test_2_pf2019_2022[1])
print("PREDICTED OUTCOME: ",y_pred[1])


force_plot = shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_values_test.values[1], 
                features=test_2_array.loc[1],
                feature_names=steps_pf_2['vect'].get_feature_names_out(),
                show=False)

file ='shap4.html'
shap.save_html(file, force_plot)