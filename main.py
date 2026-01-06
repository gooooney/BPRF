import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np  
from utils import Basic
from sklearn.metrics import roc_curve, auc ,accuracy_score, f1_score, precision_score, recall_score
from pyod.utils.utility import standardizer
from model import get_model
import time 



def model_train(data:list,norm=True):
    
    x_train, y_train, x_test, y_test = data
    if norm == True:
        x_train = standardizer(x_train.values)
        x_test = standardizer(x_test.values)
    clf_name = 'LUNAR'
    clf = get_model(clf_name,outliers_fraction=0.01)
    clf.fit(x_train)
    y_test_pred = clf.predict(x_test)  
    y_test_scores = clf.decision_function(x_test)  
    visualize(y_test, y_test_scores, y_test_pred,clf_name)
    # all_model = get_model(flag=True)
    # for i, (clf_name,clf) in enumerate(all_model.items()):

    #     clf.fit(x_train)
    #     # y_train_pred = clf.labels_
    #     # y_train_scores = clf.decision_scores_   

    #     y_test_pred = clf.predict(x_test)  
    #     y_test_scores = clf.decision_function(x_test)  

    #     visualize(y_test, y_test_scores, y_test_pred,clf_name)





def visualize(y_test, y_test_scores, y_test_pred,clf_name = None):

   
    fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
    roc_auc = auc(fpr, tpr) 
    asc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    print(f"Model Name:{clf_name}\naccuracy:{asc:.2f}\nf1_score:{f1:.2f}\nprecision:{precision:.2f}\nrecall:{recall:.2f}\nAUC:{roc_auc:.2f}\n")
    print("-"*50)

    plt.figure(figsize= (20,10))
    plt.suptitle(f'Model Name:{clf_name}')
    plt.subplot(1,2,1)
    plt.hist(y_test, bins=np.arange(-0.5, 2, 1), alpha=0.5, label='True Labels')
    plt.hist(y_test_pred, bins=np.arange(-0.5, 2, 1), alpha=0.5, label='Predicted Labels')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Distribution of True and Predicted Labels')


    plt.subplot(1,2,2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'res/metrics_{clf_name}.png',dpi = 120)
    plt.show()


def get_model_data(df:pd.DataFrame) -> list:
    x_train = df[df['train_test'] == 'train'].drop(columns=['train_test','tag'])
    y_train = df[df['train_test'] == 'train']['tag']
    x_test = df[df['train_test'] == 'prediction'].drop(columns=['train_test','tag'])
    y_test = df[df['train_test'] == 'prediction']['tag']
    return x_train, y_train, x_test, y_test


def main(ndf:pd.DataFrame):
    print("Exception marking situation")
    print(ndf.tag.value_counts())
    print("TAG".center(50,"-"))
    ndf.fillna(0, inplace=True)
    model_train(get_model_data(ndf))



if __name__ == '__main__':
    # read data
    start = time.time()
    base = Basic('CARE_To_Compare/Wind Farm A/datasets/68.csv')
    ndf = base.run()
    main(ndf)

    end = time.time()
    print("Time taken: %.2f s" % (end - start))