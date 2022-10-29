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
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

l = {
    'RF': RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2,random_state=0),
    'GNB': GaussianNB(priors=None,var_smoothing=1e-09),
    'ADB': AdaBoostClassifier(n_estimators=100, algorithm='SAMME.R', random_state=0),
    'ET': ExtraTreesClassifier(n_estimators=100, criterion='gini', min_samples_split=2),
    'GB': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, loss='deviance'),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='adam',alpha=0.00001),
    'XGB': XGBClassifier(random_state=1, learning_rate=0.05, n_estimators=7, maxdepth=5,eta=0.05, objective='binary:logistic'),
    'LR': LogisticRegression(solver='newton-cg', C=100),
    'SVC': SVC(gamma='auto'),
    'GPC': GaussianProcessClassifier(kernel=1.0 * RBF(1.0),random_state=0),
}



def algorithm1(R, S):
    '''
    The Process of Building Base-Level Model
    
        parameters:
            R: nine folds as train set
            S: One fold as test set

        return:
            train: [trainRF , trainET , ...trainGB]
            test: [testRF , testET , ...testGB
    
    '''
    kf1 = KFold(n_splits=10)
    train = {}
    train_y = {}
    test = {}
    test_y = {}
    
    #R = (R.iloc[:-(len(R)%10),:])
    
    for classifier in l:
        
        kf_splits_1 = kf1.split(R)
        train_l = []
        test_l = []
        for train_index1, validation_index1 in kf_splits_1:
            
            R_kt = R.iloc[train_index1,:] # train set
            R_kv = R.iloc[validation_index1,:] # validation set
    
            R_kt_x = R_kt.iloc[:,:-1]
            R_kt_y = R_kt.iloc[:,-1]
            
            R_kv_x = R_kv.iloc[:,:-1]
            R_kv_y = R_kv.iloc[:,-1]
            
            S_x = S.iloc[:,:-1]
            S_y = S.iloc[:,-1]
            
            # use R_kt to train ξl
            l[classifier].fit(R_kt_x,R_kt_y)
            
            train_l.append(l[classifier].predict(R_kt_x))
            test_l.append(l[classifier].predict(S_x))
            a = 0 # for debug
            
        train_l = np.sum(np.array(train_l), axis=0)
        test_l = (np.sum(np.array(test_l), axis=0)/10) 
        
        train[classifier] = train_l
        train_y[classifier] = np.array(R_kt_y)
        test[classifier] = test_l
        test_y[classifier] = np.array(S_y)
        
    
    return train,  train_y, test, test_y
    

def model_run(data_path, HP, en):
  df = pd.read_excel(data_path)[:300]
  k=10
  kf = KFold(n_splits=k, shuffle=True, random_state=None)

  kf_splits = kf.split(df.iloc[:,:]) # x is training and y is tests


  alg_1_result = []
  for train_index, test_index in kf_splits:
      r = df.iloc[train_index,:]
      s = df.iloc[test_index,:]
      
      # print(r.shape, s.shape)
      
      (train, train_y, test, test_y) = algorithm1(r,s)
      alg_1_result.append([train, train_y, test, test_y])

  clfs_list= ['RF','GNB','ADB','ET','GB','MLP','XGB','LR','SVC', 'GPC']
  algo_trainl={}
  algo_testl={}
  algo_train_y_l={}
  algo_test_y_l={}
  for i in clfs_list:
    rmxt=[]
    rmyt=[]
    rmxv=[]
    rmyv=[]
    for j in range(10):
      rmxt.append(alg_1_result[j][0][i])
      rmyt.append(alg_1_result[j][1][i])
      rmxv.append(alg_1_result[j][2][i])
      rmyv.append(alg_1_result[j][3][i])
    algo_trainl[i] = np.array(rmxt).transpose()
    algo_train_y_l[i] = np.array(rmyt[0])
    algo_testl[i] = np.array(rmxv).transpose()
    algo_test_y_l[i] = np.array(rmyv[0])

    C_res= {}
  a=1
  for i in l:
    for j in algo_trainl.keys():
      #print(j)
      clf= l[i]
      clf.fit(algo_trainl[j], algo_train_y_l[j])
      y_pred = clf.predict(algo_testl[j])
      #print(y_pred)
      #print('ssd')
      acc= accuracy_score(algo_test_y_l[j], y_pred)
      specificity = precision_score(algo_test_y_l[j], y_pred)
      sensitivity = recall_score(algo_test_y_l[j], y_pred)
      f1 = f1_score(algo_test_y_l[j], y_pred)
      C_res[i+'_C_'+str(a)]= [acc, specificity, sensitivity, f1]
      a+=1

  res_= pd.DataFrame(C_res)
  res_.to_excel(HP+'/report_'+en+'.xlsx')
