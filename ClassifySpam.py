import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import preprocessing
from scipy import interp
from matplotlib import rcParams

def aucCV(features,labels):
    
    
    my_labels=[]
    my_labels=labels  

    spam_ind=[]
    ham_ind=[]
    for t in range(len(my_labels)):
        if my_labels[t]==1:
            spam_ind.append(t)
        else:
            ham_ind.append(t)
            
    spam = features[spam_ind]
    ham = features[ham_ind]
    
    spam_sum=0
    c_spam=0
    ham_sum=0
    c_ham=0
    
    spam_mean=[0 for l in range(len(spam[0]))]
    ham_mean=[0 for o in range(len(ham[0]))]     
    
    for i in range(len(spam[0])):
        spam_sum=0
        for m in range(len(spam)):
            if spam[m][i] != 0.0:
                spam_sum = spam_sum + spam[m][i]
                c_spam=c_spam+1
        if c_spam != 0.0:
            spam_mean[i]=spam_sum/c_spam
        else:
            spam_mean[i]=0.0
        c_spam=0
        
    for j in range(len(ham[0])):
        ham_sum=0
        for n in range(len(ham)):
            if ham[n][j] != 0.0:
                ham_sum = ham_sum + ham[n][j]
                c_ham= c_ham+1
        if c_ham != 0.0:
            ham_mean[j]=ham_sum/c_ham
        else:
            ham_mean[j]=0.0
        c_ham=0
        
        
    difer=[0 for l in range(len(spam[0]))]
    for s in range(len(spam[0])):
        difer[s] = ham_mean[s]-spam_mean[s]
          
    
    num_ok = []
    for q in range(len(difer)):
        if difer[q]>0.009 or difer[q]<-0.002:
            num_ok.append(q)
    
    features = features[:,num_ok]
    features = preprocessing.scale(features)
    labels = labels.astype('int')
     
    tprs=[]
    aucs=[]
    mean_fpr=np.linspace(0,1,100)
    cv= StratifiedKFold(n_splits=10)
    i=0
    
    model = MLPClassifier(solver = 'adam',hidden_layer_sizes=(45,7), activation='tanh',
                          max_iter=90, warm_start=True, shuffle=False)
    for train,test in cv.split(features,labels):
        probaXY=model.fit(features[train],labels[train]).predict_proba(features[test])
        fpr, tpr, thresholds = roc_curve(labels[test], probaXY[:,1])
        tprs.append(interp(mean_fpr,fpr,tpr))
        tprs[-1][0]=0.0
        roc_auc=auc(fpr,tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label="ROC fold %d (AUC = %0.2f)" %(i+1, roc_auc))
        i+=1
    
    rcParams.update({'font.size':7})
    mean_tpr=np.std(tprs,axis=0)
    plt.plot([0,1],[0,1], linestyle = '--', lw=2, color='r', alpha=0.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper=np.minimum(mean_tpr+std_tpr, 1)
    tprs_lower=np.maximum(mean_tpr-std_tpr, 0)
    plt.fill_between(mean_fpr,tprs_lower, tprs_upper, color='grey', alpha=0.2, label= r"$\pm$ 1 std. dev.")
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("10 Fold Cross Validation Mean AUC")
    plt.legend(loc="lower right")    
    plt.show()
    #scores = cross_val_score(model, features, labels, cv=10,scoring='roc_auc')
    return aucs

def aucTest(trainFeatures,trainLabels,testFeatures,testLabels):
    
    
    my_labels=[]
    my_labels=trainLabels  

    spam_ind=[]
    ham_ind=[]
    for t in range(len(my_labels)):
        if my_labels[t]==1:
            spam_ind.append(t)
        else:
            ham_ind.append(t)
            
    spam = trainFeatures[spam_ind]
    ham = trainFeatures[ham_ind]
    
    spam_sum=0
    c_spam=0
    ham_sum=0
    c_ham=0
    
    spam_mean=[0 for l in range(len(spam[0]))]
    ham_mean=[0 for o in range(len(ham[0]))]     
    
    for i in range(len(spam[0])):
        spam_sum=0
        for m in range(len(spam)):
            if spam[m][i] != 0.0:
                spam_sum = spam_sum + spam[m][i]
                c_spam=c_spam+1
        if c_spam != 0.0:
            spam_mean[i]=spam_sum/c_spam
        else:
            spam_mean[i]=0.0
        c_spam=0
    
    for j in range(len(ham[0])):
        ham_sum=0
        for n in range(len(ham)):
            if ham[n][j] != 0.0:
                ham_sum = ham_sum + ham[n][j]
                c_ham= c_ham+1
        if c_ham != 0.0:
            ham_mean[j]=ham_sum/c_ham
        else:
            ham_mean[j]=0.0
        c_ham=0
        
        
    difer=[0 for l in range(len(spam[0]))]
    for s in range(len(spam[0])):
        difer[s] = ham_mean[s]-spam_mean[s]
          
    
    num_ok = []
    for q in range(len(difer)):
        if difer[q]>0.009 or difer[q]<-0.002:
            num_ok.append(q)
            
    trainFeatures = trainFeatures[:,num_ok]
    trainFeatures = preprocessing.scale(trainFeatures)
    scaler = preprocessing.StandardScaler().fit(trainFeatures)
    trainFeatures = scaler.transform(trainFeatures)
    trainLabels = trainLabels.astype('int')
                
    testFeatures = testFeatures[:,num_ok]
    testFeatures = preprocessing.scale(testFeatures)
    testFeatures = scaler.transform(testFeatures)
    testLabels = testLabels.astype('int')

    model = MLPClassifier(solver = 'adam',hidden_layer_sizes=(45,7), activation='tanh',
                          max_iter=90, warm_start=True, shuffle=False)
    model.fit(trainFeatures,trainLabels)
    testOutputs = model.predict_proba(testFeatures)[:,1]  
    
    tprs=[]
    mean_fpr=np.linspace(0,1,100)
    i=0
    fpr, tpr, thresholds = roc_curve(testLabels, testOutputs)
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label="AUC = %0.2f" %(roc_auc))
    i+=1
    
    rcParams.update({'font.size':7})
    mean_tpr=np.std(tprs,axis=0)
    plt.plot([0,1],[0,1], linestyle = '--', lw=2, color='r', alpha=0.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr+std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr-std_tpr, 0)
    plt.fill_between(mean_fpr,tprs_lower, tprs_upper, color='grey', alpha=0.2, label= r"$\pm$ 1 std. dev.")
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test Set AUC")
    plt.legend(loc="lower right")    
    plt.show()
       
    return roc_auc
    

if __name__ == "__main__":
    
    data = np.loadtxt('spamTrain.csv',delimiter=',')
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex,:]
    features = data[:,:-1]
    labels = data[:,-1]

    print("10-fold cross-validation mean AUC: ",
          np.mean(aucCV(features,labels)))       
          
    #split the dataset to test and training data
    trainFeatures, testFeatures, trainLabels, testLabels =train_test_split (features, labels, test_size=0.2)    
    print("Test set AUC: ", aucTest(trainFeatures,trainLabels,testFeatures,
                                    testLabels))
