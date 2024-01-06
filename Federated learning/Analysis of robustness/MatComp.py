import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scio
import pickle

def Mat_Dealwith(Root_File,Mat_Class):

    Mat_ValueList = scio.loadmat(os.path.join(Root_File,Mat_Class[0]))['Xtrain']
    Mat_LabelList = scio.loadmat(os.path.join(Root_File,Mat_Class[0]))['Ytrain']

    for Center in Mat_Class[1:]:
        Value_MatFile = scio.loadmat(os.path.join(Root_File,Center))['Xtrain']
        Label_MatFile = scio.loadmat(os.path.join(Root_File, Center))['Ytrain']
        Mat_ValueList = np.column_stack((Mat_ValueList,Value_MatFile))
        #Mat_LabelList = np.column_stack((Mat_LabelList, Label_MatFile))

    Index_Data = np.where(Mat_LabelList==1)
    Mat_LabelList = Mat_LabelList[Index_Data]
    Mat_ValueList = Mat_ValueList[Index_Data,][0]

    Mat_Feature = Mat_ValueList

    # pandas format is constructed for the feature matrix
    FeatureName = []
    CenterName = ['A_','B_','C_','D_']
    FeatureNumber = int(Mat_ValueList.shape[1]/len(Mat_Class))

    for i in range(len(Mat_Class)):
        for j in range(1,FeatureNumber+1):
                FeatureName.append(CenterName[i]+str(j))

    #Mat_ValueList = np.row_stack((FeatureName,Mat_ValueList))\
    scio.savemat('R AGC/R_AGC__Mat.mat', {'Data': Mat_ValueList})
    Mat_ValueList = pd.DataFrame(Mat_ValueList,columns=FeatureName)


    return Mat_Class,Mat_ValueList,Mat_LabelList,Mat_Feature

def Corr_calc(MatValue):
    CorMat = MatValue.corr()

    # f,ax = plt.subplots(figsize = (10,6))
    # camp = sns.cubehelix_palette(start=1.5,rot=3,gamma = 0.8,as_cmap=True)
    # hm = sns.heatmap(round(corr, 2), annot=True, cmap=camp, ax=ax, fmt='.2f',
    #                  linewidths=.05)

    scio.savemat('Corr_Mat.mat',{'Data':CorMat})

def Information_Corr(Cor_Mat,Select_number):
    CorrMat = scio.loadmat(os.path.join(Cor_Mat))['Data']
    """
        Screening for common and personality information:
        Commonality: large information between groups
        Adaptive: large information within groups and small information between groups
    """
    CenterCorr = int(len(CorrMat[0])/4)
    CenterName = ['A','B','C','D']
    CorrDict = {}
    Common_score = {}
    Personality_score = {}

    # adaptive
    for i,number in zip(CenterName,range(len(CenterName))):
        CorrDict[i] = CorrMat[number*CenterCorr:(number+1)*CenterCorr,]
    for (center,value),number in zip(CorrDict.items(),range(len(CenterName))):
        print('----------%s center: Analysis of data results-------' % (center))

        for j in value:
            # The within-group and between-group feature matrices were separated:
            Intra_group = value[:,number*CenterCorr:(number+1)*CenterCorr]
            Inter_group = np.delete(value,[range(number*CenterCorr,(number+1)*CenterCorr)],axis=1)

            Intra_group_avg = np.mean(Intra_group, axis=1)
            Inter_group_avg = np.mean(Inter_group, axis=1)

            Intra_Idx = np.argsort(Intra_group_avg)
            Inter_Idx = np.argsort(Inter_group_avg)

            # Per_score = Inter_Idx + Intra_Idx[-1::-1]

            def get_key2(dct, value):
                return [k for (k, v) in dct.items() if v == value]

            # Adaptive score: the score of the index with the largest intra-group similarity + the score of the index
            # with the smallest most inter-group similarity
            Score = range(1,CenterCorr+1)
            Pers_Record = {}

            # Calculate the corresponding score term for each feature
            for sco,idx in zip(Score[-1::-1],Inter_Idx):
                Pers_Record[str(idx)] = sco + (sco-list(Intra_Idx[-1::-1]).index(idx))

            # Find the five indexes with the largest key value
            MaxValue = list(Pers_Record.values())
            MaxValue.sort(reverse=True)
            Select_Pervalue = MaxValue[:Select_number]

            # Find new keys sorted by score (index of personalized features)
            # map(lambda x:x+1,a)
            Center_Idx = []
            for values in Select_Pervalue:
                Center_Idx.append(int(get_key2(Pers_Record, values)[0]))
            Personality_score[center+'_Per'] = list(map(lambda x:x+200*number,Center_Idx))

            # Commonality score: the index with the greatest similarity between groups
            Com_score = Inter_Idx[-1::-1]
            Common_score[center+'_Com'] = list(map(lambda x:x+200*number,Com_score[:Select_number]))

    with open("R AGC/Common_score1.pkl", "wb") as tf:
        pickle.dump(Common_score, tf)
    with open("R AGC/Adaptive_score1.pkl", "wb") as tf:
        pickle.dump(Personality_score, tf)

    print("Adaptive score dictionary：{}".format(Personality_score))
    print("Common score dictionary：{}".format(Common_score))

    return Personality_score,Common_score

def Select_Feature(Personality_score,Common_score,Feature_Mat):

    def Index_Deal(Dict_values):
        Index = []
        for value in Dict_values.values():
            for j in value:
                Index.append(j)
        return Index

    PreIndex = Index_Deal(Personality_score)
    ComIndex = Index_Deal(Common_score)

    Com_Feature = Feature_Mat[:,ComIndex]
    Pre_Feature = Feature_Mat[:,PreIndex]

    # Step1: Compute the covariance matrix and draw the sns heatmap

    def Corr_Pd(Coms_Feature):
        FeatureName = []
        CenterName = ['A_', 'B_', 'C_', 'D_']
        FeatureNumber = int(Coms_Feature.shape[1] / len(Mat_Class))
        for i in range(len(Mat_Class)):
            for j in range(1, FeatureNumber + 1):
                FeatureName.append(CenterName[i] + str(j))
        #Mat_ValueList = np.row_stack((FeatureName,Coms_Feature))
        Mat_ValueList = pd.DataFrame(Coms_Feature, columns=FeatureName)
        return Mat_ValueList

    Com_Feature = Corr_Pd(Com_Feature)
    Pre_Feature = Corr_Pd(Pre_Feature)

    # Step2: Construct heat map and cross-correlation matrix
    def Corr_Picture(Pd_mat,title):
        mat_corr = Pd_mat.corr()
        f,ax = plt.subplots(figsize = (10,6))
        #camp = sns.cubehelix_palette(start=1.5,rot=3,gamma = 0.8,as_cmap=True)
        # hm = sns.heatmap(round(mat_corr, 2), annot=True, cmap=camp, ax=ax, fmt='.2f',
        #                  linewidths=.05)
        hm = sns.heatmap(round(mat_corr, 2), annot=True,  ax=ax, fmt='.2f',
                                         linewidths=.05, cmap="RdBu_r")


        f.subplots_adjust(top=0.93)
        t = f.suptitle('%s Correlation Heatmap'%title, fontsize=14)
        plt.savefig("%s.eps"%title)
        plt.show()

    Corr_Picture(Com_Feature,'Common_Information')
    Corr_Picture(Pre_Feature, 'Adaptive_Information')


if __name__ == '__main__':
    Root_File = r'The feature matrix of the four models after processing'
    Mat_Class = ['CenterA','CenterB','CenterC','CenterD']
    Mat_Class, Mat_ValueList, Mat_LabelList,Mat_Feature = Mat_Dealwith(Root_File,Mat_Class)

    # Compute the result of the mutual information matrix
    Corr_calc(Mat_ValueList)
    Cor_Mat = r'Corr_Mat.mat'

    # Select a few features
    Select_number = 5
    Personality_score,Common_score = Information_Corr(Cor_Mat,Select_number)
    Select_Feature(Personality_score,Common_score,Mat_Feature)

    # Read a file
    # with open("myDictionary.pkl", "rb") as tf:
    #     new_dict = pickle.load(tf)