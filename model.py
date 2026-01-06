from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lunar import LUNAR

def get_model(model_name=None,outliers_fraction=0.01,flag = False):
    detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                    LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                    LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                    LOF(n_neighbors=50)]
    classifiers = {
        'Angle-based Outlier Detector (ABOD)':
            ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)':
            CBLOF(contamination=outliers_fraction,
                check_estimator=False, random_state=0),
        'Feature Bagging':
            FeatureBagging(LOF(n_neighbors=35),
                        contamination=outliers_fraction,
                        random_state=0),
        'Histogram-base Outlier Detection (HBOS)': HBOS(
            contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=0),
        'K Nearest Neighbors (KNN)': KNN(
            contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',
                        contamination=outliers_fraction),
        'Local Outlier Factor (LOF)':
            LOF(n_neighbors=35, contamination=outliers_fraction),
        'Minimum Covariance Determinant (MCD)': MCD(
            contamination=outliers_fraction, random_state=0),
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
        'Principal Component Analysis (PCA)': PCA(
            contamination=outliers_fraction, random_state=0),
        'Locally Selective Combination (LSCP)': LSCP(
            detector_list, contamination=outliers_fraction,
            random_state=0),
        'LUNAR':LUNAR(model_type='WEIGHT', n_neighbours=5,epsilon=0.1, proportion=1.0, n_epochs=200, lr=0.001, wd=0.1, verbose=0, contamination = outliers_fraction)
    }
    if flag:
        return classifiers
    else:
        return classifiers[model_name]