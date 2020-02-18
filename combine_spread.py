import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def tpr(df, d, c, category_name):
    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
    if len(gt) != 0:
        TPR = len(pred) / len(gt)
        return TPR
    else:
        print("Disease", d, "in category", c, "has zero division error")
        return -1

def preprocess_NIH(split):
    details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/NIH/preprocessed.csv")
    if 'Cardiomegaly' in split.columns:
        split = split.drop(columns=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'])
    split = details.merge(split, left_on='Image Index', right_on='path')
    split.drop_duplicates(subset="path", keep="first", inplace=True)
    split['Patient Age'] = np.where(split['Patient Age'].between(0,19), 19, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(20,39), 39, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(40,59), 59, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(60,79), 79, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age']>=80, 81, split['Patient Age'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
    return split

def plot_median_NIH(df, diseases, category, category_name):
    df = preprocess_NIH(df)
    GAP_total = []
    percentage_total = []
    cate = []
    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Patient Gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)
    
    return GAP_total, percentage_total, cate

def plot_sort_median_NIH(df, diseases, category, category_name):
    df_copy = df
    df = preprocess_NIH(df)
    GAP_total = []
    percentage_total = []
    cate = []
    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Patient Gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)
        
    GAP_total = np.array(GAP_total)
    percentage_total = np.array(percentage_total)

    difference = {}
    for i in range(GAP_total.shape[1]):
        mask = GAP_total[:, i] < 50
        difference[diseases[i]] = np.max(GAP_total[:, i][mask]) - np.min(GAP_total[:, i][mask])
    sort = [(k, difference[k]) for k in sorted(difference, key=difference.get, reverse=False)]
    diseases = []
    for k, _ in sort:
        diseases.append(k)
    df = df_copy
    return plot_median_NIH(df, diseases, category, category_name)

def preprocess_MIMIC(split):
    # total_subject_id = pd.read_csv("total_subject_id_with_gender.csv")
    details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/mimic-cxr-metadata-detail.csv")
    details = details.drop(columns=['dicom_id', 'study_id'])
    details.drop_duplicates(subset="subject_id", keep="first", inplace=True)
    if "subject_id" not in split.columns:
        subject_id = []
        for idx, row in split.iterrows():
            subject_id.append(row['path'].split('/')[1][1:])
        split['subject_id'] = subject_id
        split = split.sort_values("subject_id")
    if "gender" not in split.columns:
        split["subject_id"] = pd.to_numeric(split["subject_id"])
        split = split.merge(details, left_on="subject_id", right_on="subject_id")
    split = split.replace(
        [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
         'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '>=90'],
        [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
         'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])
    return split

def plot_14_MIMIC(df, diseases, category, category_name):
    df = preprocess_MIMIC(df)
    map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
    df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
    GAP_total = []
    percentage_total = []
    cate = []
    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)
        
    return GAP_total, percentage_total, cate

def plot_sort_14_MIMIC(df, diseases, category, category_name):

    df_copy = df
    df = preprocess_MIMIC(df)
    map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
    df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
    GAP_total = []
    percentage_total = []
    cate = []
    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)
        
    GAP_total = np.array(GAP_total)
    percentage_total = np.array(percentage_total)

    difference = {}
    for i in range(GAP_total.shape[1]):
        mask = GAP_total[:, i] < 50
        difference[diseases[i]] = np.max(GAP_total[:, i][mask]) - np.min(GAP_total[:, i][mask])
    sort = [(k, difference[k]) for k in sorted(difference, key=difference.get, reverse=False)]
    diseases = []
    for k, _ in sort:
        diseases.append(k)
    df = df_copy
    return plot_14_MIMIC(df, diseases, category, category_name)

def plot_14_CXP(df, diseases, category, category_name):
    df = preprocess_CXP(df)
    GAP_total = []
    percentage_total = []
    cate = []
    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Sex':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)
        
    return GAP_total, percentage_total, cate

def plot_sort_14_CXP(df, diseases, category, category_name):
    df_copy = df
    df = preprocess_CXP(df)
    GAP_total = []
    percentage_total = []
    cate = []
    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'Sex':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)
        
    GAP_total = np.array(GAP_total)
    percentage_total = np.array(percentage_total)

    difference = {}
    for i in range(GAP_total.shape[1]):
        mask = GAP_total[:, i] < 50
        difference[diseases[i]] = np.max(GAP_total[:, i][mask]) - np.min(GAP_total[:, i][mask])
    sort = [(k, difference[k]) for k in sorted(difference, key=difference.get, reverse=False)]
    diseases = []
    for k, _ in sort:
        diseases.append(k)
    df = df_copy
    return plot_14_CXP(df, diseases, category, category_name)

def preprocess_CXP(split):
    details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/CheXpert/map.csv")
    if 'Atelectasis' in split.columns:
        details = details.drop(columns=['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices'])
    split = split.merge(details, left_on="Path", right_on="Path")
    split['Age'] = np.where(split['Age'].between(0,19), 19, split['Age'])
    split['Age'] = np.where(split['Age'].between(20,39), 39, split['Age'])
    split['Age'] = np.where(split['Age'].between(40,59), 59, split['Age'])
    split['Age'] = np.where(split['Age'].between(60,79), 79, split['Age'])
    split['Age'] = np.where(split['Age']>=80, 81, split['Age'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81, 'Male', 'Female'], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-", 'M', 'F'])
    return split

def list_to_array(l):
    ret = np.zeros((len(l), len(l[0]), len(l[0][0])))
    for x in range(len(l)):
        for y in range(len(l[0])):
            for z in range(len(l[0][0])):
                ret[x, y, z] = l[x][y][z]
    return ret

def plot():
    plt.rcParams.update({'font.size': 18})
    #NIH data
    diseases_NIH = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Pneumonia', 'Pneumothorax']
    age_decile_NIH = ['60-80', '40-60', '20-40', '80-', '0-20']
    gender_NIH = ['M', 'F']
    pred_NIH = pd.read_csv("./NIH/results/bipred.csv")
    factor_NIH = [gender_NIH, age_decile_NIH]
    # factor_NIH = [gender_NIH]
    factor_str_NIH = ['Patient Gender', 'Patient Age']
    
    #MIMIC data
    # diseases_MIMIC = ['Atelectasis', 'Cardiomegaly',
    #    'Consolidation', 'Edema', 'Pleural Effusion',
    #    'Pneumonia', 'Pneumothorax', 'No Finding']
    diseases_MIMIC = ['Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Pleural Effusion',
       'Pneumonia', 'Pneumothorax']
    age_decile_MIMIC = ['60-80', '40-60', '20-40', '80-', '0-20']
    gender_MIMIC = ['M', 'F']
    pred_MIMIC = pd.read_csv("./MIMIC/results/bipred.csv")
    factor_MIMIC = [gender_MIMIC, age_decile_MIMIC]
    factor_str_MIMIC = ['gender', 'age_decile']

    #CXP data
    # diseases_CXP = ['Atelectasis', 'Cardiomegaly',
    #         'Consolidation' , 'Edema', 'Pleural Effusion', 'Pneumonia',
    #         'Pneumothorax', 'No Finding']
    diseases_CXP = ['Atelectasis', 'Cardiomegaly',
            'Consolidation' , 'Edema', 'Pleural Effusion', 'Pneumonia',
            'Pneumothorax']
    Age_CXP = ['60-80', '40-60', '20-40', '80-', '0-20']
    gender_CXP = ['M', 'F']
    pred_CXP = pd.read_csv("./CXP/results/bipred.csv")
    factor_CXP = [gender_CXP, Age_CXP]
    factor_str_CXP = ['Sex', 'Age']

    ylabel = {'Patient Age': 'AGE',
        'Patient Gender': 'GENDER'
        }

    for f in range(len(factor_NIH)):
        GAP = []
        PERCENT = []
        CATE = []
        gap, percentage, cate = plot_median_NIH(pred_NIH, diseases_NIH, factor_NIH[f], factor_str_NIH[f])
        GAP.append(gap)
        PERCENT.append(percentage)
        CATE.append(np.array(cate))
        gap, percentage, cate = plot_14_MIMIC(pred_MIMIC, diseases_MIMIC, factor_MIMIC[f], factor_str_MIMIC[f])
        GAP.append(gap)
        PERCENT.append(percentage)
        CATE.append(np.array(cate))
        gap, percentage, cate = plot_14_CXP(pred_CXP, diseases_CXP, factor_CXP[f], factor_str_CXP[f])
        GAP.append(gap)
        PERCENT.append(percentage)
        CATE.append(np.array(cate))
        
        GAP = list_to_array(GAP)
        PERCENT = list_to_array(PERCENT)
        CATE = np.array(CATE)

        t = np.arange(3)
        # plt.figure(figsize=(50,4))
        fig, axs = plt.subplots(1, len(diseases_MIMIC)+1, sharey=True, figsize=(25,10))
        # Remove horizontal space between axes
        fig.subplots_adjust(wspace=0)

        # Plot each graph, and manually set the y tick values

        for i in range(len(diseases_MIMIC)):

            for g in range(GAP.shape[1]):
                s = np.multiply(PERCENT[:, g, i], 1000)
                mask = GAP[:, g, i] < 50
                axs[i].scatter(t[mask], GAP[:, g, i][mask], s=s, marker='o', label=CATE[0, g])
            axs[i].set_xticks(t)
            axs[i].set_xticklabels(['NIH','CXR','CXP'], rotation=90)
            axs[i].set(xlabel=diseases_MIMIC[i])
        axs[0].set(ylabel='TPR ' + ylabel[factor_str_NIH[f]] + ' GAP')

        
        diseases_special = ['No Finding']
        special_GAP = []
        special_PERCENT = []
        special_CATE = []
        gap, percentage, cate = plot_14_MIMIC(pred_MIMIC, diseases_special, factor_MIMIC[f], factor_str_MIMIC[f])
        special_GAP.append(gap)
        special_PERCENT.append(percentage)
        special_CATE.append(np.array(cate))
        gap, percentage, cate = plot_14_CXP(pred_CXP, diseases_special, factor_CXP[f], factor_str_CXP[f])
        special_GAP.append(gap)
        special_PERCENT.append(percentage)
        special_CATE.append(np.array(cate))

        special_GAP = list_to_array(special_GAP)
        special_PERCENT = list_to_array(special_PERCENT)
        special_CATE = np.array(special_CATE)

        t = np.arange(2)
        for g in range(special_GAP.shape[1]):
            s = np.multiply(PERCENT[:, g, 0], 1000)
            mask = special_GAP[:, g, 0] < 50
            axs[7].scatter(t[mask], special_GAP[:, g, 0][mask], s=s, marker='o', label=special_CATE[0, g])

        axs[7].set_xticks(t)
        axs[7].set_xticklabels(['CXR','CXP'], rotation=90)
        axs[7].set(xlabel=diseases_special[0])
        axs[7].legend()

        plt.savefig("./results/Combine_Spread_" + factor_str_NIH[f] + ".pdf")


if __name__ == "__main__":
    plot()