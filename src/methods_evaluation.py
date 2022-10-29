import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.ensemble import ExtraTreesClassifier # ET
from sklearn.ensemble import AdaBoostClassifier # ADB
from sklearn.svm import SVC # svc
from sklearn.neural_network import MLPClassifier # MLP
from xgboost import XGBClassifier # XGB
from sklearn.gaussian_process import GaussianProcessClassifier # GPC
from sklearn.naive_bayes import GaussianNB # GNB
from sklearn.linear_model import LogisticRegression # LR
from sklearn.ensemble import GradientBoostingClassifier # GBC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import scipy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import statistics


class methods:
    def __init__(self, parameters, HP, dataset_name):
        self.datapath = parameters[dataset_name]
        self.resname= dataset_name
        self.test_size = parameters['test_size']
        self.label = parameters['label']
        self.random_state = parameters['random_state']
        self.models_list = parameters['models_list']
        self. test_num = parameters['test_num']
        self.model_RF = RandomForestClassifier(n_estimators=10,
                                               max_depth=None,
                                               min_samples_split=2,
                                               random_state=0)
        self.model_GNB = GaussianNB(priors=None,
                                    var_smoothing=1e-09)
        self.model_ADB = AdaBoostClassifier(n_estimators=50,
                                            learning_rate=1.0,
                                            algorithm='SAMME.R',
                                            random_state=0)
        self.model_ET = ExtraTreesClassifier(n_estimators=100,
                                             criterion='gini',
                                             min_samples_split=2)
        self.model_GB = GradientBoostingClassifier(n_estimators=100,
                                                   learning_rate=0.1,
                                                   loss='deviance')
        self.model_MLP = MLPClassifier(hidden_layer_sizes=(100,),
                                       activation='relu',
                                       solver='adam',
                                       alpha=0.0001)
        self.model_XGB = XGBClassifier(random_state=1,
                                       learning_rate=0.05,
                                       n_estimators=7,
                                       maxdepth=5,
                                       eta=0.05,
                                       objective='binary:logistic')
        self.model_LR = LogisticRegression(solver='newton-cg',
                                           C=100)
        self.Data_Prepration()
        self.results_dataframe = pd.DataFrame(columns=['method', 'Accuracy', 'sensitivity', 'specificity', 'f1'])
        self.experience_path = HP

    def Data_Prepration(self):
        df_input= pd.read_excel(self.datapath)
        Y = df_input.pop(self.label)
        X = df_input
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size,
                                                            random_state=self.random_state)
        self.cv = KFold(n_splits=10, random_state=1, shuffle=True)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        print("Data Preparation Done!")
    def models_Prepration(self):
        for i in self.models_list:

            if i=="RF":
                self.model_RF.fit(self.X_train, self.y_train)
                self.Res_RF = self.predict_point(self.X_test,
                                                 self.model_RF,
                                                 self.test_num,
                                                 self.y_test)
                self.report(i)
                self.results_dataframe.loc[len(self.results_dataframe.index)] = [i,
                                                                                 self.Res_RF[0],
                                                                                 self.Res_RF[1],
                                                                                 self.Res_RF[2],
                                                                                 self.Res_RF[3]]

            elif i == "GNB":
                self.model_GNB.fit(self.X_train,
                                   self.y_train)
                self.Res_GNB = self.predict_point(self.X_test,
                                                  self.model_GNB,
                                                  self.test_num,
                                                  self.y_test)
                self.report(i)
                self.results_dataframe.loc[len(self.results_dataframe.index)] = [i,
                                                                                 self.Res_GNB[0],
                                                                                 self.Res_GNB[1],
                                                                                 self.Res_GNB[2],
                                                                                 self.Res_GNB[3]]
            elif i == "ADB":
                self.model_ADB.fit(self.X_train,
                                   self.y_train)
                self.Res_ADB = self.predict_point(self.X_test,
                                                  self.model_ADB,
                                                  self.test_num,
                                                  self.y_test)
                self.report(i)
                self.results_dataframe.loc[len(self.results_dataframe.index)] = [i,
                                                                                 self.Res_ADB[0],
                                                                                 self.Res_ADB[1],
                                                                                 self.Res_ADB[2],
                                                                                 self.Res_ADB[3]]
            elif i == "ET":
                self.model_ET.fit(self.X_train,
                                  self.y_train)
                self.Res_ET = self.predict_point(self.X_test,
                                                 self.model_ET,
                                                 self.test_num,
                                                 self.y_test)
                self.report(i)
                self.results_dataframe.loc[len(self.results_dataframe.index)] = [i,
                                                                                 self.Res_ET[0],
                                                                                 self.Res_ET[1],
                                                                                 self.Res_ET[2],
                                                                                 self.Res_ET[3]]
            elif i == "GB":
                self.model_GB.fit(self.X_train,
                                  self.y_train)
                self.Res_GB = self.predict_point(self.X_test,
                                                 self.model_GB,
                                                 self.test_num,
                                                 self.y_test)
                self.report(i)
                self.results_dataframe.loc[len(self.results_dataframe.index)] = [i,
                                                                                 self.Res_GB[0],
                                                                                 self.Res_GB[1],
                                                                                 self.Res_GB[2],
                                                                                 self.Res_GB[3]]
            elif i == "MLP":
                self.model_MLP.fit(self.X_train,
                                   self.y_train)
                self.Res_MLP = self.predict_point(self.X_test,
                                                  self.model_MLP,
                                                  self.test_num,
                                                  self.y_test)
                self.report(i)
                self.results_dataframe.loc[len(self.results_dataframe.index)] = [i,
                                                                                 self.Res_MLP[0],
                                                                                 self.Res_MLP[1],
                                                                                 self.Res_MLP[2],
                                                                                 self.Res_MLP[3]]
            elif i == "XGB":
                self.model_XGB.fit(self.X_train,
                                   self.y_train)
                self.Res_XGB = self.predict_point(self.X_test,
                                                  self.model_XGB,
                                                  self.test_num,
                                                  self.y_test)
                self.report(i)
                self.results_dataframe.loc[len(self.results_dataframe.index)] = [i,
                                                                                 self.Res_XGB[0],
                                                                                 self.Res_XGB[1],
                                                                                 self.Res_XGB[2],
                                                                                 self.Res_XGB[3]]
            elif i == "LR":
                self.model_LR.fit(self.X_train,
                                  self.y_train)
                self.Res_LR = self.predict_point(self.X_test,
                                                 self.model_LR,
                                                 self.test_num,
                                                 self.y_test)
                self.report(i)
                self.results_dataframe.loc[len(self.results_dataframe.index)] = [i,
                                                                                 self.Res_LR[0],
                                                                                 self.Res_LR[1],
                                                                                 self.Res_LR[2],
                                                                                 self.Res_LR[3]]
        self.results_dataframe.to_excel(self.experience_path+'/'+self.resname+'.xlsx')



    def predict_point(self, X, model, num_samples, y_test):   #Predict Test Data
      Ou__=[]
      accuracy=[]
      specificity=[]
      sensitivity=[]
      Mode=[]
      # for i in range(num_samples):
        #p_ = model.predict(X)
      _o = self.PRF(X, y_test, model)
      return _o

      #   accuracy.append(accuracy_o)
      #   specificity.append(specificity_o)
      #   sensitivity.append(sensitivity_o)
      # LI__= [accuracy,
      #        specificity,
      #        sensitivity]
      # ou_nps= []
      # ou_npm= []
      # for k in LI__:
      #     print(k)
      #     #ou_npm.append(scipy.stats.sem(k))
      #     #ou_nps.append(statistics.pstdev(k))
      #     ou_nps.append(statistics.stdev(k))
      #
      # for k in LI__:
      #     ou_npm.append(statistics.mean(k))
      # return(ou_npm, ou_nps)

    def PRF(self, X, y_test, model):  # Calculate precision/ Recall/ F1 Score
        recall = []
        #tn, fp, fn, tp = confusion_matrix(y_test, X).ravel()
        accuracy = cross_val_score(model, X, y_test, scoring='accuracy', cv=self.cv, n_jobs=-1)
        #accuracy = accuracy_score(X__, y_test)
        specificity = cross_val_score(model, X, y_test, scoring='precision', cv=self.cv, n_jobs=-1)
        sensitivity = cross_val_score(model, X, y_test, scoring='recall', cv=self.cv, n_jobs=-1)
        f1 = cross_val_score(model, X, y_test, scoring='f1', cv=self.cv, n_jobs=-1)
        acc_mean = str(accuracy.mean())[:4]
        acc_sd = str(accuracy.std())[:4]
        specificity_mean = str(specificity.mean())[:4]
        specificity_sd = str(specificity.std())[:4]
        sensitivity_mean = str(sensitivity.mean())[:4]
        sensitivity_sd = str(sensitivity.std())[:4]
        f1_mean = str(f1.mean())[:4]
        f1_sd = str(f1.std())[:4]
        acc = acc_mean+' ± '+acc_sd
        spec = specificity_mean+' ± '+specificity_sd
        sens = sensitivity_mean+' ± '+sensitivity_sd
        f1_o = f1_mean + ' ± ' + f1_sd


        return [acc, spec, sens, f1_o]

    def report(self, name):
        print("Module - " + name + " Done!")


