from sklearn import svm
def SVM(kernel_gram_train,kernel_gram_test_train,training_labels):
    clf = svm.SVC(C=1,kernel='precomputed',probability=True)
    clf.fit(kernel_gram_train,training_labels)
    probs=clf.predict_proba(kernel_gram_test_train)
    y_pred = clf.predict(kernel_gram_test_train)
    return probs,y_pred
