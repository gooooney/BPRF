import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

class Basic:
    def __init__(self, csv_file:str) -> None:
        self.csv_file = csv_file
        self.event_file = 'CARE_To_Compare/Wind Farm A/event_info.csv'
        self.df = pd.read_csv(self.csv_file,sep = ';')
        self.event_df = pd.read_csv(self.event_file,sep = ';')
        self.df.drop(['asset_id', 'id', 'status_type_id'],axis=1,inplace=True)
        self.df.ffill(inplace=True)
        self.df.sort_values(by='time_stamp', inplace=True)
        self.df['tag'] = 1


    def get_fault_time(self) -> pd.DataFrame:
        file_name = self.csv_file.split('/')[-1].split('.')[0]
        for i in range(len(self.event_df)):
            event_id = self.event_df.iloc[i]["event_id"]
            event_name = self.event_df.iloc[i]['event_label']
            if event_id == int(file_name) and event_name == 'anomaly':
                start_time = self.event_df.iloc[i]['event_start']
                end_time = self.event_df.iloc[i]['event_end']
                self.df.loc[(self.df['time_stamp'] >= start_time) & (self.df['time_stamp'] <= end_time),'tag'] = 0
                return self.df 
    
    def run(self):
        df = self.get_fault_time().set_index('time_stamp')
        return df 


def pca_model(df:pd.DataFrame):
    pass 



def rf_bayes(df:pd.DataFrame,p_flag = True):
    dt = df.copy()
    dt.drop(columns=['train_test'], inplace=True)
    X,y = dt.iloc[:,:-1],dt.iloc[:,-1]
    # Define the hyperparameter space to be optimized
    search_space = {
        'n_estimators': Integer(10, 200),
        'max_depth': Integer(1, 50),
        'min_samples_split': Integer(2, 100),
        'min_samples_leaf': Integer(1, 10),
        'bootstrap': Categorical([True, False])
    }

    # # Bayesian optimization using BayesSearchCV
    rf = RandomForestClassifier(random_state=0)
    opt = BayesSearchCV(rf, search_space, n_iter=32, random_state=0, cv=5)
    opt.fit(X, y)

    print("最优参数：", opt.best_params_)
    # 最优参数： OrderedDict([('bootstrap', False), ('max_depth', 1), ('min_samples_leaf', 1), ('min_samples_split', 2), ('n_estimators', 124)])
    # rf = RandomForestClassifier(n_estimators=120,random_state=66)
    rf = RandomForestClassifier(**opt.best_params_,random_state=66)
    rf.fit(X, y)
    mdf = pd.DataFrame({"important":rf.feature_importances_})
    mdf.sort_values(by="important", ascending=False, inplace=True)
    # fe = mdf.head(10).index.tolist()
    mdf = pd.DataFrame({"important":rf.feature_importances_})
    mdf = mdf[mdf["important"] > 0]
    mdf.sort_values(by="important", ascending=False, inplace=True)
    fe = mdf.index.tolist()
    dt = dt.iloc[:,fe]
    if p_flag:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(dt.values)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        xpdf = pd.DataFrame(X_pca,columns=['pca1','pca2'])
        xpdf['tag'] = df['tag'].values
        xpdf['train_test'] = df['train_test'].values
        return xpdf
        
    else:
        dt['tag'] = df['tag'].values
        dt['train_test'] = df['train_test'].values
        return dt