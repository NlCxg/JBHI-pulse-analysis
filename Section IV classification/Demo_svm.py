from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import scipy.io
mat_file_path = 'He_vs_dm_stft.mat'


X = scipy.io.loadmat(mat_file_path)
X =  X["data"]
zeros_part = np.zeros(200)
ones_part = np.ones(200)
y = np.concatenate((zeros_part, ones_part))


all_metrics = {'accuracy': [], 'g_mean': [], 'f1_score': [], 'precision': []}
accuracies = []

for kk in range(20):
    svm_classifier = SVC(kernel='rbf')
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svm_classifier.fit(X_train, y_train)

        y_pred = svm_classifier.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        g_mean = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        all_metrics['accuracy'].append(acc)
        all_metrics['g_mean'].append(g_mean)
        all_metrics['f1_score'].append(f1)
        all_metrics['precision'].append(precision)


average_accuracy = np.mean(all_metrics['accuracy'])
std_accuracy = np.std(all_metrics['accuracy'])

average_g_mean = np.mean(all_metrics['g_mean'])
std_g_mean = np.std(all_metrics['g_mean'])

average_f1_score = np.mean(all_metrics['f1_score'])
std_f1_score = np.std(all_metrics['f1_score'])

average_precision = np.mean(all_metrics['precision'])
std_precision = np.std(all_metrics['precision'])


print(
    '& {:.2f}'.format(average_precision * 100) + '$_{\pm' + '{:.2f}'.format(std_precision * 100) + '}$' +
    ' & {:.2f}'.format(average_g_mean * 100) +   '$_{\pm' + '{:.2f}'.format(std_g_mean * 100)    + '}$' +
    ' & {:.2f}'.format(average_f1_score * 100) + '$_{\pm' + '{:.2f}'.format(std_f1_score * 100)  + '}$' +
    ' & {:.2f}'.format(average_accuracy * 100) + '$_{\pm' + '{:.2f}'.format(std_g_mean * 100)    + '}$'
)
