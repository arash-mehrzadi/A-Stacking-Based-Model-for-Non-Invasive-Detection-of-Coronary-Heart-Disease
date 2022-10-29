import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class feature_selection:
    def __init__(self, parameters, HP):
        self.data_path = 'data/Z-Alizadeh sani dataset_preprocessed.xlsx'
        self.experience_path = HP
        self.label = parameters['label']
        self.k_fold_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 17, 22]
        self.methods_list = ["CHI2", "RFECV", "XGB", "mutual", "variance", "svc"]
        self.CHI2 = SelectKBest(score_func=chi2)
        self.RFECV = RFECV(SVC(kernel="linear"), step=1, cv=20)
        self.XGB = XGBClassifier(random_state=1,
                                 learning_rate=0.05,
                                 n_estimators=7,
                                 maxdepth=5,
                                 eta=0.05,
                                 objective='binary:logistic')
        self.mutual = SelectKBest(score_func=mutual_info_classif)
        self.variance = VarianceThreshold(threshold=(.8 * (1 - .8)))
        self.svc = LinearSVC(C=0.01, penalty="l1", dual=False)
        self.results_dataframe = pd.DataFrame(columns=['method', 'Accuracy', 'sensitivity', 'specificity'])
        self.test_size = 0.33
        self.random_state = 42
        self.features = {}
        self.res_dic = {}
        self.data_import()
        self.methods_run()


    def data_import(self):
        df_input = pd.read_excel(self.data_path)
        self.y = df_input.pop(self.label)
        self.X = df_input

        print("data import Done!")

    def methods_run(self):
        for i in self.methods_list:

            if i == "CHI2":
                features = self.CHI2.fit(self.X, self.y)
                self.features['CHI2'] = features.get_feature_names_out()
                selected_x_data = self.X[self.features['CHI2']]
                self.eval_(selected_x_data, self.y, i)
                selected_x_data[self.label] = self.y
                selected_x_data.to_excel(self.experience_path+'/'+i+'_fs_dataset.xlsx', index=False)

                self.report(i)
            elif i == "RFECV":
                features = self.RFECV.fit(self.X, self.y)
                features = list((self.X.columns[features.get_support()]))
                self.features['RFECV'] = features
                selected_x_data = self.X[self.features['RFECV']]
                self.eval_(selected_x_data, self.y, i)
                selected_x_data[self.label] = self.y
                selected_x_data.to_excel(self.experience_path+'/'+i+'_fs_dataset.xlsx', index=False)
                self.report(i)
            elif i == "XGB":
                features = self.XGB.fit(self.X, self.y)

                f_list = {'score': list(features.feature_importances_), 'name': list(self.X.columns)}
                dfxg = pd.DataFrame(f_list)
                dfxg = dfxg[dfxg.score != 0]
                dfxg = dfxg.sort_values(by=['score'], ascending=False)
                self.features['XGB'] = list(dfxg['name'])
                selected_x_data = self.X[self.features['XGB']]
                self.eval_(selected_x_data, self.y, i)
                selected_x_data[self.label] = self.y
                selected_x_data.to_excel(self.experience_path+'/'+i+'_fs_dataset.xlsx', index=False)
                self.report(i)
            elif i == "mutual":
                features = self.mutual.fit(self.X, self.y)
                self.features['mutual'] = features.get_feature_names_out()
                selected_x_data = self.X[self.features['mutual']]
                self.eval_(selected_x_data, self.y, i)
                selected_x_data[self.label] = self.y
                selected_x_data.to_excel(self.experience_path+'/'+i+'_fs_dataset.xlsx', index=False)
                self.report(i)
            elif i == "variance":
                features = self.variance.fit_transform(self.X)
                self.features['variance'] = list((self.X.columns[self.variance.get_support()]))
                selected_x_data = self.X[self.features['variance']]
                self.eval_(selected_x_data, self.y, i)
                selected_x_data[self.label] = self.y
                selected_x_data.to_excel(self.experience_path+'/'+i+'_fs_dataset.xlsx', index=False)
                self.report(i)
            elif i == "svc":
                features = self.svc.fit(self.X, self.y)
                model = SelectFromModel(features, prefit=True)
                self.features['svc'] = list((self.X.columns[model.get_support()]))
                selected_x_data = self.X[self.features['svc']]
                self.eval_(selected_x_data, self.y, i)
                selected_x_data[self.label] = self.y
                selected_x_data.to_excel(self.experience_path+'/'+i+'_fs_dataset.xlsx', index=False)
                self.report(i)

        print("Hi")
        fdf = pd.DataFrame.from_dict(self.features, orient='index').transpose()
        fdf.to_excel(self.experience_path+'/feature_selection_RES.xlsx', index=False)
        fdf = pd.DataFrame.from_dict(self.res_dic, orient='index').transpose()
        fdf.to_excel(self.experience_path + '/feature_selection_RES_kvalue.xlsx', index=False)
        #self.check= self.res_dic


    def eval_(self, X, y, name):
        #print(name+'  '+str(len(X.columns)))

        k_report_list = []
        svc_clf = SVC(C=1.0, kernel='linear')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        svc_clf.fit(X_train, y_train)
        for i in self.k_fold_list:
            cv = KFold(n_splits=i, random_state=1, shuffle=True)
            k_report_list.append(self.predict_point(X_test, svc_clf, y_test, cv))
        self.res_dic[name] = k_report_list


    def predict_point(self, X, model, y_test, cv):

      _o = self.PRF(X, y_test, model, cv)
      return _o



    def PRF(self, X, y_test, model, cv):
        accuracy = cross_val_score(model, X, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
        acc_mean = str(accuracy.mean())[:4]
        acc_sd = str(accuracy.std())[:4]
        acc = acc_mean+' ± '+acc_sd
        print(acc)
        return acc


    def report(self, name):
        print("Module - " + name + " Done!")
