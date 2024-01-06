import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve,auc
import os

def List_opt(Index_0):
    Index_new = []
    for index,i in enumerate(Index_0):
        if i== True:
            Index_new.append(index)
    return np.mat(Index_new)


def Auc_Data_Calc(Prevalue,TrueLabel,patienceName):
    PatiencePictureNum = {}
    patiencePrevalue = []
    patienceTrueLabel = []

    Prevalue =  [np.float32(n) for a in Prevalue for n in a ]
    TrueLabel = [np.float32(n) for a in TrueLabel for n in a ]

    Prevalue = np.array(Prevalue)
    Index_0 = np.isnan(Prevalue)
    Prevalue[Index_0] = 0
    Index_1 = np.isinf(Prevalue)
    Prevalue[Index_1] = 1

    fpr, tpr, threshold = roc_curve(TrueLabel,Prevalue)
    pictureAuc = auc(fpr, tpr)

    patienceName = [n for a in patienceName for n in a]
    tmp = list(set(patienceName))
    tmp.sort(key=patienceName.index)
    for patienceFile in tmp:
        PatiencePictureNum[patienceFile.split('/')[-2]] = len(os.listdir(os.path.dirname(patienceFile)))

    ksum = 0
    for (key,value) in (PatiencePictureNum.items()):
        sumpre = 0.
        sumlabel = 0.
        for i in range(value):
            sumpre = sumpre + Prevalue[i + ksum]
            sumlabel = sumlabel + TrueLabel[i + ksum]
        ksum = ksum + value
        patiencePrevalue.append(sumpre/np.float32(value))
        patienceTrueLabel.append(sumlabel / np.float32(value))

    patience_fpr, patience_tpr, threshold = roc_curve(patienceTrueLabel,patiencePrevalue)

    patienceAuc = auc(patience_fpr, patience_tpr)

    return [pictureAuc,patienceAuc,patienceTrueLabel,patiencePrevalue,PatiencePictureNum.keys()]

def AucResuluts_logs(args,training,auclists,G_serverNames):
    os.makedirs(args.train_save,exist_ok=True)
    # Start log
    if training:
        for G_serverName,auclist in zip(G_serverNames,auclists):
            Train_csv_path = os.path.join(args.train_save, str(G_serverName)+'_Train_Auc.csv')
            with open(Train_csv_path, 'w') as f:
                f.write('name,label,value\n')
            for Trainlabel,TrainValue,Name in zip(auclist[2],auclist[3],auclist[4]):
                with open(Train_csv_path, 'a') as f:
                    f.write('%s,%d,%0.6f\n' % (Name,Trainlabel, TrainValue))
            with open(Train_csv_path, 'a') as f:
                    f.write('Train_auc--Picture:%.4f---patients:%.4f\n' % (auclist[0],auclist[1]))

    else:
        for G_serverName,auclist in zip(G_serverNames,auclists):
            Test_csv_path = os.path.join(args.train_save, str(G_serverName)+'_Test_Auc.csv')
            with open(Test_csv_path, 'w') as f:
                f.write('name,label,value\n')
            for Testlabel,TestValue,Name in zip(auclist[2],auclist[3],auclist[4]):
                with open(Test_csv_path, 'a') as f:
                    f.write('%s,%d,%0.6f\n' % (Name,Testlabel, TestValue))
            with open(Test_csv_path, 'a') as f:
                    f.write('Test_auc--Picture:%.4f---patients:%.4f\n' % (auclist[0],auclist[1]))