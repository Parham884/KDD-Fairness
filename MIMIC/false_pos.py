 # This code provide the results of subgroup-specific underdiagnosis and Intersectional specific chronic underdiagnosis by studing the NoFinding label
# since

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fpr(df, d, c, category_name):
    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 0) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (df[d] == 0) & (df[category_name] == c), :]
    if len(gt) != 0:
        FPR = len(pred) / len(gt)
        return FPR
    else:
        # print("Disease", d, "in category", c, "has zero division error")
        return -1

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

# def FP_NF_CXP(df, diseases, category, category_name):
#     df = preprocess_CXP(df)
#     GAP_total = np.zeros((len(category), len(diseases)))
#     percentage_total = np.zeros((len(category), len(diseases)))
#     cate = []
#     print("FP in CXP====================================")
#     for c in range(len(category)):
#         for d in range(len(diseases)):
#             pred_disease = "bi_" + diseases[d]
#             gt = df.loc[(df[diseases[d]] == 0) & (df[category_name] == category[c]), :]
#             pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name] == category[c]), :]
#             n_gt = df.loc[(df[diseases[d]] == 0) & (df[category_name] != category[c]) & (df[category_name] != 0), :] # why?
#             n_pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name] != category[c]) & (df[category_name] != 0), :]
#             pi_gy = df.loc[(df[diseases[d]] == 0) & (df[category_name] == category[c]), :]
#             pi_y = df.loc[(df[diseases[d]] == 0) & (df[category_name] != 0), :]
#
#             if len(gt) != 0:
#                 FPR = len(pred) / len(gt)
#                 print("False Positive Rate in " + category[c] + " for " + diseases[d] + " is: " + str(FPR))
#             else:
#                 print("False Positive Rate in " + category[c] + " for " + diseases[d] + " is: N\A")
# # We have not finally calculate GAP for FP study, we report the FPR themselve
#         #     if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
#         #         FPR = len(pred) / len(gt)
#         #         n_FPR = len(n_pred) / len(n_gt)
#         #         percentage = len(pi_gy) / len(pi_y)
#         #         if category_name != 'gender':
#         #             temp = []
#         #             for c1 in category:
#         #                 ret = fpr(df, diseases[d], c1, category_name)
#         #                 if ret != -1:
#         #                     temp.append(ret)
#         #             temp.sort()
#         #             if len(temp) % 2 == 0:
#         #                 median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)])/2
#         #             else:
#         #                 median = temp[(len(temp) // 2)]
#         #             GAP = FPR - median
#         #         else:
#         #             GAP = FPR - n_FPR
#         #         GAP_total[c, d] = GAP
#         #         percentage_total[c, d] = percentage
#         #     #    print("GAP in " + category[c] + " for " + diseases[d] + " is: " + str(GAP))
#         #     else:
#         #         GAP_total[c, d] = 51
#         #         percentage_total[c, d] = 0
#         #     #    print("GAP in " + category[c] + " for " + diseases[d] + " is: N/A")
#         #
#         # cate.append(category[c])
#
#     return FPR

def FP_NF_MIMIC(df, diseases, category, category_name):
    df = preprocess_MIMIC(df)
    map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
    df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
    GAP_total = np.zeros((len(category), len(diseases)))
    percentage_total = np.zeros((len(category), len(diseases)))
    cate = []

    # if category_name == 'gender':
    #     FP_sex = pd.DataFrame(diseases, columns=["diseases"])
    #
    # if category_name == 'age_decile':
    #     FP_age = pd.DataFrame(diseases, columns=["diseases"])
    #
    # if category_name == 'race':
    #     FP_race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'insurance':
        FP_insurance = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'race':
        FP_race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'gender':
        FP_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'age_decile':
        FP_age = pd.DataFrame(diseases, columns=["diseases"])

    print("FP in MIMIC====================================")
    i=0
    for c in range(len(category)):
        for d in range(len(diseases)):
            pred_disease = "bi_" + diseases[d]
            gt = df.loc[(df[diseases[d]] == 0) & (df[category_name] == category[c]), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name] == category[c]), :]
            # n_gt = df.loc[(df[diseases[d]] == 0) & (df[category_name] != category[c]) & (df[category_name] != 0), :]
            # n_pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name] != category[c]) & (df[category_name] != 0), :]
            # pi_gy = df.loc[(df[diseases[d]] == 0) & (df[category_name] == category[c]), :]
            # pi_y = df.loc[(df[diseases[d]] == 0) & (df[category_name] != 0), :]


            if len(gt) != 0:
                FPR = len(pred) / len(gt)
                print("False Positive Rate in " + category[c] + " for " + diseases[d] + " is: " + str(FPR))

                if category_name == 'gender':
                    if i == 0:
                        FPR_S = pd.DataFrame([FPR], columns=["M"])
                        FP_sex = pd.concat([FP_sex, FPR_S.reindex(FP_sex.index)], axis=1)

                    if i == 1:
                        FPR_S = pd.DataFrame([FPR], columns=["F"])
                        FP_sex = pd.concat([FP_sex, FPR_S.reindex(FP_sex.index)], axis=1)

                # make sure orders are right

                if category_name == 'age_decile':
                    if i == 0:
                        FPR_A = pd.DataFrame([FPR], columns=["60-80"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)

                    if i == 1:
                        FPR_A = pd.DataFrame([FPR], columns=["40-60"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)

                    if i == 2:
                        FPR_A = pd.DataFrame([FPR], columns=["20-40"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)

                    if i == 3:
                        FPR_A = pd.DataFrame([FPR], columns=["80-"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)

                    if i == 4:
                        FPR_A = pd.DataFrame([FPR], columns=["0-20"])
                        FP_age = pd.concat([FP_age, FPR_A.reindex(FP_age.index)], axis=1)

                if category_name == 'insurance':
                    if i == 0:
                        FPR_Ins = pd.DataFrame([FPR], columns=["Medicare"])
                        FP_insurance = pd.concat([FP_insurance, FPR_Ins.reindex(FP_insurance.index)], axis=1)

                    if i == 1:
                        FPR_Ins = pd.DataFrame([FPR], columns=["Other"])
                        FP_insurance = pd.concat([FP_insurance, FPR_Ins.reindex(FP_insurance.index)], axis=1)


                    if i == 2:
                        FPR_Ins = pd.DataFrame([FPR], columns=["Medicaid"])
                        FP_insurance = pd.concat([FP_insurance, FPR_Ins.reindex(FP_insurance.index)], axis=1)

                if category_name == 'race':
                    if i == 0:
                        FPR_Rac = pd.DataFrame([FPR], columns=["White"])
                        FP_race = pd.concat([FP_race, FPR_Rac.reindex(FP_race.index)], axis=1)

                    if i == 1:
                        FPR_Rac = pd.DataFrame([FPR], columns=["Black"])
                        FP_race = pd.concat([FP_race, FPR_Rac.reindex(FP_race.index)], axis=1)


                    if i == 2:
                        FPR_Rac = pd.DataFrame([FPR], columns=["Hisp"])
                        FP_race = pd.concat([FP_race, FPR_Rac.reindex(FP_race.index)], axis=1)

                    if i == 3:
                        FPR_Rac = pd.DataFrame([FPR], columns=["Other"])
                        FP_race = pd.concat([FP_race, FPR_Rac.reindex(FP_race.index)], axis=1)

                    if i == 4:
                        FPR_Rac = pd.DataFrame([FPR], columns=["Asian"])
                        FP_race = pd.concat([FP_race, FPR_Rac.reindex(FP_race.index)], axis=1)

                    if i == 5:
                        FPR_Rac = pd.DataFrame([FPR], columns=["American"])
                        FP_race = pd.concat([FP_race, FPR_Rac.reindex(FP_race.index)], axis=1)


            else:
                print("False Positive Rate in " + category[c] + " for " + diseases[d] + " is: N\A")

        i= i+1

    if category_name == 'insurance':
        FP_insurance.to_csv("./results/FP_insurance.csv")

    if category_name == 'race':
        FP_race.to_csv("./results/FP_race.csv")

    if category_name == 'gender':
        FP_sex.to_csv("./results/FP_sex.csv")

    if category_name == 'age_decile':
        FP_age.to_csv("./results/FP_age.csv")

    
    return FPR


def FP_NF_MIMIC_Inter(df, diseases, category1, category_name1,category2, category_name2 ):
    df = preprocess_MIMIC(df)
    map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
    df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
    # GAP_total = np.zeros((len(category), len(diseases)))
    # percentage_total = np.zeros((len(category), len(diseases)))
    # cate = []

    if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
        FP_InsSex = pd.DataFrame(category2, columns=["Insurance"])

    if (category_name1 == 'gender')  &  (category_name2 == 'race'):
        FP_RaceSex = pd.DataFrame(category2, columns=["race"])

    if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
        FP_InsRace = pd.DataFrame(category2, columns=["race"])


    print("FP in MIMIC====================================")
    i = 0
    for c1 in range(len(category1)):
        FPR_list = []

        for c2 in range(len(category2)):
            for d in range(len(diseases)):
                pred_disease = "bi_" + diseases[d]
                gt =   df.loc[((df[diseases[d]] == 0)  & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]
                pred = df.loc[((df[pred_disease] == 1) & (df[diseases[d]] == 0) & (df[category_name1] == category1[c1]) & (df[category_name2] == category2[c2])), :]


                if len(gt) != 0:
                    FPR = len(pred) / len(gt)
                    print(len(pred),'--' ,len(gt))
                    print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: " + str(FPR))

                else:
                    print("False Positive Rate in " + category1[c1] +"/" + category2[c2] + " for " + diseases[d] + " is: N\A")


            FPR_list.append(FPR)

        if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
            if i == 0:
                FPR_SIn = pd.DataFrame(FPR_list, columns=["M"])
                FP_InsSex = pd.concat([FP_InsSex, FPR_SIn.reindex(FP_InsSex.index)], axis=1)

            if i == 1:
                FPR_SIn = pd.DataFrame(FPR_list, columns=["F"])
                FP_InsSex = pd.concat([FP_InsSex, FPR_SIn.reindex(FP_InsSex.index)], axis=1)

        if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
            if i == 0:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Medicare"])
                FP_InsRace = pd.concat([FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)

            if i == 1:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Other"])
                FP_InsRace = pd.concat([FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)

            if i == 2:
                FPR_RIn = pd.DataFrame(FPR_list, columns=["Medicaid"])
                FP_InsRace = pd.concat([FP_InsRace, FPR_RIn.reindex(FP_InsRace.index)], axis=1)


        if (category_name1 == 'gender')  &  (category_name2 == 'race'):
            if i == 0:
                FPR_SR = pd.DataFrame(FPR_list, columns=["M"])
                FP_RaceSex = pd.concat([FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)

            if i == 1:
                FPR_SR = pd.DataFrame(FPR_list, columns=["F"])
                FP_RaceSex = pd.concat([FP_RaceSex, FPR_SR.reindex(FP_RaceSex.index)], axis=1)


        i = i + 1

    if (category_name1 == 'gender')  &  (category_name2 == 'insurance'):
        FP_InsSex.to_csv("./results/FP_InsSex.csv")

    if (category_name1 == 'gender')  &  (category_name2 == 'race'):
        FP_RaceSex.to_csv("./results/FP_RaceSex.csv")

    if (category_name1 == 'insurance')  &  (category_name2 == 'race'):
        FP_InsRace.to_csv("./results/FP_InsRace.csv")


    return FPR


def FPR_Underdiagnosis():
    

    #MIMIC data
    diseases_MIMIC = ['No Finding']
    age_decile_MIMIC = ['60-80', '40-60', '20-40', '80-', '0-20']

    gender_MIMIC = ['M', 'F']

    race_MIMIC = ['WHITE', 'BLACK/AFRICAN AMERICAN','HISPANIC/LATINO',
            'OTHER', 'ASIAN', 'AMERICAN INDIAN/ALASKA NATIVE']
    insurance_MIMIC = ['Medicare', 'Other', 'Medicaid']

    pred_MIMIC = pd.read_csv("./results/bipred.csv")
    factor_MIMIC = [gender_MIMIC, age_decile_MIMIC, race_MIMIC, insurance_MIMIC]
    factor_str_MIMIC = ['gender', 'age_decile', 'race', 'insurance']

    # #CXP data
    # diseases_CXP = ['No Finding']
    # Age_CXP = ['60-80', '40-60', '20-40', '80-', '0-20']
    # gender_CXP = ['M', 'F']
    # pred_CXP = pd.read_csv("./CXP/results/bipred.csv")
    # factor_CXP = [gender_CXP, Age_CXP]
    # factor_str_CXP = ['Sex', 'Age']

    # False positive rates are study on 'No Finding' label of MIMIC-CXR dataset only
    
    #Subgroup-specific Chronic Underdiagnosis
    FP_NF_MIMIC(pred_MIMIC, diseases_MIMIC, insurance_MIMIC, 'insurance')
    
    FP_NF_MIMIC(pred_MIMIC, diseases_MIMIC, age_decile_MIMIC, 'age_decile')
    
    FP_NF_MIMIC(pred_MIMIC, diseases_MIMIC, race_MIMIC, 'race')
    
    #Intersectional-specific Chronic Underdiagnosis
    
    FP_NF_MIMIC_Inter(pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender',race_MIMIC,'race')

    FP_NF_MIMIC_Inter(pred_MIMIC, diseases_MIMIC, gender_MIMIC, 'gender', insurance_MIMIC, 'insurance')

    FP_NF_MIMIC_Inter(pred_MIMIC, diseases_MIMIC, insurance_MIMIC,'insurance', race_MIMIC, 'race')




if __name__ == '__main__':
    FPR_Underdiagnosis()