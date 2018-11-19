from sklearn import metrics
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from SVM import SVM
drug_target_data_test=pd.read_csv(os.getcwd()+'/drug_target_data_test.csv')
drug_target_data_train=pd.read_csv(os.getcwd()+'/drug_target_data_train.csv')
tanimoto_kernel_test=np.loadtxt('tanimoto_kernel_test.npy')
mismatch_kernel_test=np.loadtxt('mismatch_kernel_test.npy')
tanimoto_kernel_train=np.loadtxt('tanimoto_kernel_train.npy')
mismatch_kernel_train=np.loadtxt('mismatch_kernel_train.npy')
tanimoto_kernel_test_train=np.loadtxt('tanimoto_kernel_test_train.npy')
mismatch_kernel_test_train=np.loadtxt('mismatch_kernel_test_train.npy')

k_mismatch_train=np.reshape(np.sqrt(np.diag(mismatch_kernel_train))+1e-12,(-1,1))
k_mismatch_test=np.reshape(np.sqrt(np.diag(mismatch_kernel_test))+1e-12,(-1,1))
k_train=np.dot(k_mismatch_train,(np.transpose(k_mismatch_train)))
k_test=np.dot(k_mismatch_test,(np.transpose(k_mismatch_test)))
k_test_train=np.dot(k_mismatch_test,(np.transpose(k_mismatch_train)))
mismatch_kernel_train_norm=mismatch_kernel_train/k_train
mismatch_kernel_test_train_norm=mismatch_kernel_test_train/k_test_train
mismatch_kernel_test_norm=mismatch_kernel_test/k_test

#tensor product kernel
tensor_kernel_gram_train=tanimoto_kernel_train*mismatch_kernel_train_norm
tensor_kernel_gram_test_train=tanimoto_kernel_test_train*mismatch_kernel_test_train_norm
tensor_kernel_gram_test=tanimoto_kernel_test*mismatch_kernel_test_norm

K_tensor_train=np.reshape(np.sqrt(np.diag(tensor_kernel_gram_train))+1e-12,(-1,1))
K_tensor_test=np.reshape(np.sqrt(np.diag(tensor_kernel_gram_test))+1e-12,(-1,1))
K_train_2=np.dot(K_tensor_train,(np.transpose(K_tensor_train)))
K_test_2=np.dot(K_tensor_test,(np.transpose(K_tensor_test)))
K_test_train_2=np.dot(K_tensor_test,(np.transpose(K_tensor_train)))
tensor_kernel_gram_train_norm=tensor_kernel_gram_train/K_train_2
tensor_kernel_gram_test_train_norm=tensor_kernel_gram_test_train/K_test_train_2
tensor_kernel_gram_test_norm=tensor_kernel_gram_test/K_test_2
#polynomial tensor kernel
polynomial_kernel_gram_train=np.power((0.6+tensor_kernel_gram_train_norm),2)
polynomial_kernel_gram_test_train=np.power((0.6+tensor_kernel_gram_test_train_norm),2)
polynomial_kernel_gram_test=np.power((0.6+tensor_kernel_gram_test_norm),2)

K_polynomial_train=np.reshape(np.sqrt(np.diag(polynomial_kernel_gram_train))+1e-12,(-1,1))
K_polynomial_test=np.reshape(np.sqrt(np.diag(polynomial_kernel_gram_test))+1e-12,(-1,1))
Kk_train_2=np.dot(K_polynomial_train,(np.transpose(K_polynomial_train)))
Kk_test_2=np.dot(K_polynomial_test,(np.transpose(K_polynomial_test)))
Kk_test_train_2=np.dot(K_polynomial_test,(np.transpose(K_polynomial_train)))
polynomial_kernel_gram_train_norm=polynomial_kernel_gram_train/Kk_train_2
polynomial_kernel_gram_test_train_norm=polynomial_kernel_gram_test_train/Kk_test_train_2
polynomial_kernel_gram_test_norm=polynomial_kernel_gram_test/Kk_test_2
 #direct sum kernel
direct_kernel_gram_train=tanimoto_kernel_train+mismatch_kernel_train_norm
direct_kernel_gram_test_train=tanimoto_kernel_test_train+mismatch_kernel_test_train_norm
direct_kernel_gram_test=tanimoto_kernel_test+mismatch_kernel_test_norm

K_direct_train=np.reshape(np.sqrt(np.diag(direct_kernel_gram_train))+1e-12,(-1,1))
K_direct_test=np.reshape(np.sqrt(np.diag(direct_kernel_gram_test))+1e-12,(-1,1))
K_train=np.dot(K_direct_train,(np.transpose(K_direct_train)))
K_test=np.dot(K_direct_test,(np.transpose(K_direct_test)))
K_test_train=np.dot(K_direct_test,(np.transpose(K_direct_train)))
direct_kernel_gram_train_norm=direct_kernel_gram_train/K_train
direct_kernel_gram_test_train_norm=direct_kernel_gram_test_train/K_test_train
direct_kernel_gram_test_norm=direct_kernel_gram_test/K_test

#polynomial direct kernel
polynomial_direct_kernel_gram_train=(3+direct_kernel_gram_train_norm)**5
polynomial_direct_kernel_gram_test_train=(3+direct_kernel_gram_test_train_norm)**5
polynomial_direct_kernel_gram_test=(3+direct_kernel_gram_test_norm)**5

K_polynomial_direct_train=np.reshape(np.sqrt(np.diag(polynomial_direct_kernel_gram_train))+1e-12,(-1,1))
K_polynomial_direct_test=np.reshape(np.sqrt(np.diag(polynomial_direct_kernel_gram_test))+1e-12,(-1,1))
Kk_train=np.dot(K_polynomial_direct_train,(np.transpose(K_polynomial_direct_train)))
Kk_test=np.dot(K_polynomial_direct_test,(np.transpose(K_polynomial_direct_test)))
Kk_test_train=np.dot(K_polynomial_direct_test,(np.transpose(K_polynomial_direct_train)))
polynomial_direct_kernel_gram_train_norm=polynomial_direct_kernel_gram_train/Kk_train
polynomial_direct_kernel_gram_test_train_norm=polynomial_direct_kernel_gram_test_train/Kk_test_train
polynomial_direct_kernel_gram_test_norm=polynomial_direct_kernel_gram_test/Kk_test

probs,y_pred=SVM(tensor_kernel_gram_train_norm,tensor_kernel_gram_test_train_norm,drug_target_data_train['labels'])
#probs,y_pred=SVM(polynomial_kernel_gram_train_norm,polynomial_kernel_gram_test_train_norm,drug_target_data_train['labels'])
#probs,y_pred=SVM(direct_kernel_gram_train_norm,direct_kernel_gram_test_train_norm,drug_target_data_train['labels'])
#probs,y_pred=SVM(polynomial_direct_kernel_gram_train_norm,polynomial_direct_kernel_gram_test_train_norm,drug_target_data_train['labels'])
def performance_measures(probs,y_pred,test_labels):
    tn, fp, fn, tp =metrics.confusion_matrix(test_labels, y_pred, labels=[-1, 1]).ravel()
    pred_acc=metrics.accuracy_score(test_labels, y_pred)
    fpr, tpr, threshold = metrics.roc_curve(test_labels, probs[:,1])
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(drug_target_data_test['labels'], y_pred)
    average_precision = metrics.average_precision_score(test_labels, y_pred)
    return tn,fp,fn,tp,pred_acc,tpr,fpr,threshold,roc_auc,precision,recall,average_precision
tn,fp,fn,tp,pred_acc,tpr,fpr,threshold,roc_auc,precision,recall,average_precision=performance_measures(probs,y_pred,drug_target_data_test['labels'])

fig=plt.figure(1)    
plt.subplot(211)
plt.step(recall, precision, color='red', alpha=0.2,where='post',label ='AP={0:0.2f}'.format(average_precision))
plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
plt.legend(loc = 'upper right')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')

plt.subplot(212)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.subplots_adjust( hspace=0.6)
fig.savefig('tensor_kernel.png')
#fig.savefig('tensor_polynomial_kernel.png')
#fig.savefig('direct_kernel.png')
#fig.savefig('direct_polynomial_kernel.png')