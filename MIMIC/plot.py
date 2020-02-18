import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit

diseases_abbr = {'Cardiomegaly': 'Cd',
                 'Effusion': 'Ef',
                 'Enlarged Cardiomediastinum': 'EC',
                 'Lung Lesion': 'LL',
                 'Atelectasis': 'A',
                 'Pneumonia': 'Pa',
                 'Pneumothorax': 'Px',
                 'Consolidation': 'Co',
                 'Edema': 'Ed',
                 'Pleural Effusion': 'Ef',
                 'Pleural Other': 'PO',
                 'Fracture': 'Fr',
                 'Support Devices': 'SD',
                 'Airspace Opacity': 'AO',
                 'No Finding': 'NF'
                 }

ylabel = {'age_decile': 'AGE',
          'race': 'RACE',
          'marital_status': 'MARITAL STATUS',
          'insurance': 'INSURANCE TYPE',
          'gender': 'SEX',
          'F': 'FEMALE',
          'M': 'MALE'
          }


def plot_frequency(df, diseases, category, category_name):
    plt.rcParams.update({'font.size': 18})
    df = preprocess(df)
    freq = []
    for d in diseases:
        cate = []
        for c in category:
            cate.append(len(df.loc[(df[d] == 1) & (df[category_name] == c), :]))
        freq.append(cate)
    freq = np.array(freq)
    if category_name == 'age_decile':
        plt.figure(figsize=(18, 9))

        width = 0.075
    elif category_name == 'race':
        plt.figure(figsize=(18, 9))

        width = 0.1
    elif category_name == 'marital_status':
        plt.figure(figsize=(18, 9))

        width = 0.125
    elif category_name == 'insurance':
        plt.figure(figsize=(18, 9))

        width = 0.2
    elif category_name == 'gender':
        plt.figure(figsize=(18, 9))

        width = 0.35
    ind = np.arange((len(diseases)))
    for i in range(len(category)):
        if category_name == 'gender':
            plt.bar(ind + width * i, freq[:, i], width, label=ylabel[category[i]])
        else:
            plt.bar(ind + width * i, freq[:, i], width, label=category[i].upper())
    plt.ylabel(str(ylabel[category_name] + ' FREQUENCY IN CXR').upper())
    plt.xticks(ind + width * len(category) / 2, [diseases_abbr[k] for k in diseases], )
    plt.legend()
    plt.savefig("./results/Frequency_" + category_name + ".pdf")
    # plt.show()


def tpr(df, d, c, category_name):
    pred_disease = "bi_" + d
    gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
    pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
    if len(gt) != 0:
        TPR = len(pred) / len(gt)
        return TPR
    else:
        # print("Disease", d, "in category", c, "has zero division error")
        return -1


def func(x, m, b):
    return m * x + b


def distance_max_min(df, diseases, category, category_name):
    df = preprocess(df)
    map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
    df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
    GAP_total = np.zeros((len(category), len(diseases)))
    percentage_total = np.zeros((len(category), len(diseases)))
    cate = []
    for c in range(len(category)):
        for d in range(len(diseases)):
            pred_disease = "bi_" + diseases[d]
            gt = df.loc[(df[diseases[d]] == 1) & (df[category_name] == category[c]), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 1) & (df[category_name] == category[c]), :]
            n_gt = df.loc[(df[diseases[d]] == 1) & (df[category_name] != category[c]) & (df[category_name] != 0), :]
            n_pred = df.loc[(df[pred_disease] == 1) & (df[diseases[d]] == 1) & (df[category_name] != category[c]) & (
                        df[category_name] != 0), :]
            pi_gy = df.loc[(df[diseases[d]] == 1) & (df[category_name] == category[c]), :]
            pi_y = df.loc[(df[diseases[d]] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, diseases[d], c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()
                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                GAP_total[c, d] = GAP
                percentage_total[c, d] = percentage
            else:
                GAP_total[c, d] = 51
                percentage_total[c, d] = 0

        category[c] = category[c].replace(' ', '_', 3)
        category[c] = category[c].replace('/', '_', 3)
        cate.append(category[c])

    df = pd.DataFrame(index=['Max', 'Min', 'Distance'])
    for d in range(len(diseases)):
        mask = GAP_total[:, d] < 50
        minimum = np.min(GAP_total[:, d][mask])
        maximum = np.max(GAP_total[:, d][mask])
        distance = maximum - minimum
        df[diseases[d]] = [maximum, minimum, distance]
    df.to_csv("./results/distance_max_min_" + category_name + ".csv")


def count(list1, l, r):
    c = 0
    # traverse in the list1
    for x in list1:
        # condition check
        if x > l and x < r:
            c += 1
    return c


def plot_14(df, diseases, category, category_name):
    plt.rcParams.update({'font.size': 18})
    df = preprocess(df)
    map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
    df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
    GAP_total = []
    percentage_total = []
    cate = []

    print(diseases)

    if category_name == 'gender':
        Run1_sex = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'age_decile':
        Run1_age = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'race':
        Run1_race = pd.DataFrame(diseases, columns=["diseases"])

    if category_name == 'insurance':
        Run1_insurance = pd.DataFrame(diseases, columns=["diseases"])

    for c in category:
        GAP_y = []
        percentage_y = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt)
                percentage = len(pi_gy) / len(pi_y)
                if category_name != 'gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name) # return the TPR of the fubgroup
                        if ret != -1:  # if there is no patient in that subgroup return -1
                            temp.append(ret)
                    temp.sort()  # sort the TPR in subgroups per disease to estimate the median

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
                    else:
                        median = temp[(len(temp) // 2)]

                    # The TPR gap is the TPR (of the subgroup) - Median(all TPRs)
                    GAP = TPR - median
                else:
                    # for gender TPR gap is the TPR(sex(g)) - TPR(sex(not g))
                    GAP = TPR - n_TPR

                # append all gaps of 14 disease and percentages for plot
                GAP_y.append(GAP)
                percentage_y.append(percentage)
            else:
                GAP_y.append(51)
                percentage_y.append(0)

        # Gaps of all 14 diseases and categories
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        cate.append(c)

    GAP_total = np.array(GAP_total)
    x = np.arange(len(diseases))
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(111)
    for item in x:
        mask = GAP_total[:, item] < 50   # becouse we append 51 gap value for subgroup with no patients
        ann = ax.annotate('', xy=(item, np.max(GAP_total[:, item][mask])), xycoords='data',
                          xytext=(item, np.min(GAP_total[:, item][mask])), textcoords='data',
                          arrowprops=dict(arrowstyle="<->",
                                          connectionstyle="bar"))
    for i in range(len(GAP_total)):
        s = np.multiply(percentage_total[i], 1000)
        mask = GAP_total[i] < 50
        plt.scatter(x[mask], GAP_total[i][mask], s=s, marker='o', label=cate[i].upper())

        print("Perc", percentage_total[i])
        print("GAPt", GAP_total[i][mask])

        # Here for each attribiute we make a csv file which shows what is the gap and patient
        # percentage per  subgroup and disease


        if category_name == 'age_decile':

            if i == 0:
                Percent4 = pd.DataFrame(percentage_total[i], columns=["%60-80"])
                Run1_age = pd.concat([Run1_age, Percent4.reindex(Run1_age.index)], axis=1)

                Gap4 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_60-80"])
                Run1_age = pd.concat([Run1_age, Gap4.reindex(Run1_age.index)], axis=1)

            if i == 1:
                Percent6 = pd.DataFrame(percentage_total[i], columns=["%40-60"])
                Run1_age = pd.concat([Run1_age, Percent6.reindex(Run1_age.index)], axis=1)

                Gap6 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_40-60"])
                Run1_age = pd.concat([Run1_age, Gap6.reindex(Run1_age.index)], axis=1)

            if i == 2:
                Percent4 = pd.DataFrame(percentage_total[i], columns=["%20-40"])
                Run1_age = pd.concat([Run1_age, Percent4.reindex(Run1_age.index)], axis=1)

                Gap4 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_20-40"])
                Run1_age = pd.concat([Run1_age, Gap4.reindex(Run1_age.index)], axis=1)

            if i == 3:
                Percent8 = pd.DataFrame(percentage_total[i], columns=["%80-"])
                Run1_age = pd.concat([Run1_age, Percent8.reindex(Run1_age.index)], axis=1)

                Gap8 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_80-"])
                Run1_age = pd.concat([Run1_age, Gap8.reindex(Run1_age.index)], axis=1)

            if i == 4:
                Percent0 = pd.DataFrame(percentage_total[i], columns=["%0-20"])
                Run1_age = pd.concat([Run1_age, Percent0.reindex(Run1_age.index)], axis=1)

                Gap0 = pd.DataFrame(GAP_total[i][mask], columns=["Gap_0-20"])
                Run1_age = pd.concat([Run1_age, Gap0.reindex(Run1_age.index)], axis=1)

            Run1_age.to_csv("./results/Run1_Age.csv")

        if category_name == 'gender':

            if i == 0:
                MalePercent = pd.DataFrame(percentage_total[i], columns=["%M"])
                Run1_sex = pd.concat([Run1_sex, MalePercent.reindex(Run1_sex.index)], axis=1)

                MaleGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_M"])
                Run1_sex = pd.concat([Run1_sex, MaleGap.reindex(Run1_sex.index)], axis=1)

            else:
                FeMalePercent = pd.DataFrame(percentage_total[i], columns=["%F"])
                Run1_sex = pd.concat([Run1_sex, FeMalePercent.reindex(Run1_sex.index)], axis=1)

                FeMaleGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_F"])
                Run1_sex = pd.concat([Run1_sex, FeMaleGap.reindex(Run1_sex.index)], axis=1)

            Run1_sex.to_csv("./results/Run1_sex.csv")

        if category_name == 'race':
            if i == 0:
                WhPercent = pd.DataFrame(percentage_total[i], columns=["%White"])
                Run1_race = pd.concat([Run1_race, WhPercent.reindex(Run1_race.index)], axis=1)

                WhGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_White"])
                Run1_race = pd.concat([Run1_race, WhGap.reindex(Run1_race.index)], axis=1)

            if i == 1:
                BlPercent = pd.DataFrame(percentage_total[i], columns=["%Black"])
                Run1_race = pd.concat([Run1_race, BlPercent.reindex(Run1_race.index)], axis=1)

                BlGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_Black"])
                Run1_race = pd.concat([Run1_race, BlGap.reindex(Run1_race.index)], axis=1)

            if i == 2:
                BlPercent = pd.DataFrame(percentage_total[i], columns=["%Hisp"])
                Run1_race = pd.concat([Run1_race, BlPercent.reindex(Run1_race.index)], axis=1)

                BlGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_Hisp"])
                Run1_race = pd.concat([Run1_race, BlGap.reindex(Run1_race.index)], axis=1)

            if i == 3:
                BlPercent = pd.DataFrame(percentage_total[i], columns=["%Other"])
                Run1_race = pd.concat([Run1_race, BlPercent.reindex(Run1_race.index)], axis=1)

                BlGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_Other"])
                Run1_race = pd.concat([Run1_race, BlGap.reindex(Run1_race.index)], axis=1)

            if i == 4:
                AsPercent = pd.DataFrame(percentage_total[i], columns=["%Asian"])
                Run1_race = pd.concat([Run1_race, AsPercent.reindex(Run1_race.index)], axis=1)

                AsGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_Asian"])
                Run1_race = pd.concat([Run1_race, AsGap.reindex(Run1_race.index)], axis=1)

            if i == 5:
                AmPercent = pd.DataFrame(percentage_total[i], columns=["%American"])
                Run1_race = pd.concat([Run1_race, AmPercent.reindex(Run1_race.index)], axis=1)

                AmGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_American"])
                Run1_race = pd.concat([Run1_race, AmGap.reindex(Run1_race.index)], axis=1)

            Run1_race.to_csv("./results/Run1_race.csv")

        if category_name == 'insurance':
            if i == 0:
                CarePercent = pd.DataFrame(percentage_total[i], columns=["%Medicare"])
                Run1_insurance = pd.concat([Run1_insurance, CarePercent.reindex(Run1_insurance.index)], axis=1)

                CareGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_Medicare"])
                Run1_insurance = pd.concat([Run1_insurance, CareGap.reindex(Run1_insurance.index)], axis=1)

            if i == 1:
                OtherPercent = pd.DataFrame(percentage_total[i], columns=["%Other"])
                Run1_insurance = pd.concat([Run1_insurance, OtherPercent.reindex(Run1_insurance.index)], axis=1)

                OtherGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_Other"])
                Run1_insurance = pd.concat([Run1_insurance, OtherGap.reindex(Run1_insurance.index)], axis=1)

            if i == 2:
                AidPercent = pd.DataFrame(percentage_total[i], columns=["%Medicaid"])
                Run1_insurance = pd.concat([Run1_insurance, AidPercent.reindex(Run1_insurance.index)], axis=1)

                AidGap = pd.DataFrame(GAP_total[i][mask], columns=["Gap_Medicaid"])
                Run1_insurance = pd.concat([Run1_insurance, AidGap.reindex(Run1_insurance.index)], axis=1)

            Run1_insurance.to_csv("./results/Run1_insurance.csv")

    plt.xticks(x, [diseases_abbr[k] for k in diseases])
    plt.ylabel("TPR " + ylabel[category_name] + " DISPARITY")
    plt.legend()
    plt.savefig("./results/Median_Diseases_x_GAP_" + category_name + ".pdf")


def plot_sort_14(df, diseases, category, category_name):
    df_copy = df
    df = preprocess(df)
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
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]  # find ground truth of subgroup c has disease d
            # find pred_disease = 1 and gt =1  of subgroup c has disease d: TP
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]

            # Used for sex as disparity defined at TPR(group g) - TPR(not group d)
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :] # ground truth of not group g
            # Prediction of not group g
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]


            # to calculate percentage of patient per subgroup/disease
            pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]

            if len(gt) != 0 and len(n_gt) != 0 and len(pi_y) != 0:
                TPR = len(pred) / len(gt)
                n_TPR = len(n_pred) / len(n_gt) # TPR of group not g
                percentage = len(pi_gy) / len(pi_y)
                # for all groups except sex the median formulation of TPR gap is utilized
                if category_name != 'gender':
                    temp = []
                    for c1 in category:
                        ret = tpr(df, d, c1, category_name)
                        if ret != -1:
                            temp.append(ret)
                    temp.sort()

                    if len(temp) % 2 == 0:
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
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
                # we later remove the gap of disease with 0 percentages. 51 is just append to hold the space
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
    plot_14(df, diseases, category, category_name)


def plot_TPR_MIMIC(df, diseases, category, category_name):
    plt.rcParams.update({'font.size': 18})
    df = preprocess(df)
    map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
    df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
    final = {}
    for c in category:
        result = []
        for d in diseases:
            pred_disease = "bi_" + d
            gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
            pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
            n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
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
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
                    else:
                        median = temp[(len(temp) // 2)]
                    GAP = TPR - median
                else:
                    GAP = TPR - n_TPR
                result.append([percentage, GAP])
            else:
                result.append([50, 50])
        result = np.array(result)
        plt.figure(figsize=(10, 8))
        plt.subplots_adjust()
        mask = result[:, 1] < 50
        plt.scatter(result[:, 0][mask], result[:, 1][mask], label='TPR', color='green')

        # params, params_cov = curve_fit(func, result[:, 0][mask], result[:, 1][mask])
        # plt.plot(result[:, 0][mask], func(result[:, 0][mask], params[0], params[1]), color='green')
        diseases = np.array(diseases)
        for d, x, y in zip(diseases[mask], result[:, 0][mask], result[:, 1][mask]):
            plt.annotate(diseases_abbr[d], color='green', xy=(x, y), xytext=(-3, 3), textcoords='offset points',
                         ha='right', va='bottom')
        plt.xlabel("% " + c)
        plt.ylabel("TPR " + ylabel[category_name] + " DISPARITY")
        c = c.replace(' ', '_', 3)
        c = c.replace('/', '_', 3)
        c = c.replace('>=', '_', 3)
        plt.savefig("./results/Median_TPR_" + category_name + "_" + c + ".pdf")

        ans = {'result': result,
               'mask': mask}
        final[c] = ans
    return final


def preprocess(split):
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
         'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
         '>=90'],
        [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
         'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])
    return split


def random_split(map_path, total_subject_id, split_portion):
    df = pd.read_csv(map_path)
    subject_df = pd.read_csv(total_subject_id)
    subject_df['random_number'] = np.random.uniform(size=len(subject_df))

    train_id = subject_df[subject_df['random_number'] <= split_portion[0]]
    valid_id = subject_df[
        (subject_df['random_number'] > split_portion[0]) & (subject_df['random_number'] <= split_portion[1])]
    test_id = subject_df[subject_df['random_number'] > split_portion[1]]

    train_id = train_id.drop(columns=['random_number'])
    valid_id = valid_id.drop(columns=['random_number'])
    test_id = test_id.drop(columns=['random_number'])

    train_id.to_csv("train_id.csv", index=False)
    valid_id.to_csv("valid_id.csv", index=False)
    test_id.to_csv("test_id.csv", index=False)

    train_df = train_id.merge(df, left_on="subject_id", right_on="subject_id")
    valid_df = valid_id.merge(df, left_on="subject_id", right_on="subject_id")
    test_df = test_id.merge(df, left_on="subject_id", right_on="subject_id")

    print(len(train_df))
    print(len(valid_df))
    print(len(test_df))

    train_df.to_csv("new_train.csv", index=False)
    valid_df.to_csv("new_valid.csv", index=False)
    test_df.to_csv("new_test.csv", index=False)


def plot_14_MIMIC(df, diseases, category, category_name):
    df = preprocess(df)
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
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
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
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
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
        # print('Best Positive ' + c + ' ' + str(count(GAP_y, 0, 51)))
        # print('Worst Positive ' + c + ' ' + str(count(GAP_y, -50, 0)))
        # print('Zero ' + c + ' ' + str(GAP_y.count(0)))
        GAP_total.append(GAP_y)
        percentage_total.append(percentage_y)
        cate.append(c)

    return GAP_total, percentage_total, cate


def plot_sort_14_MIMIC(df, diseases, category, category_name):
    df_copy = df
    df = preprocess(df)
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
            n_pred = df.loc[
                     (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
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
                        median = (temp[(len(temp) // 2) - 1] + temp[(len(temp) // 2)]) / 2
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


def list_to_array(lst):
    ret = np.zeros((len(lst), len(lst[0])))
    for x in range(len(lst)):
        for y in range(len(lst[0])):
            ret[x, y] = lst[x][y]
    return ret


def plot():
    plt.rcParams.update({'font.size': 20})
    # MIMIC data
    diseases_MIMIC = ['Airspace Opacity', 'Edema',
                      'Lung Lesion', 'No Finding']

    race = ['WHITE', 'BLACK/AFRICAN AMERICAN',
            'OTHER', 'HISPANIC/LATINO', 'ASIAN',
            'AMERICAN INDIAN/ALASKA NATIVE']
    insurance = ['Medicare', 'Other', 'Medicaid']
    race_abbr = {'WHITE': 'WHITE',
                 'BLACK/AFRICAN AMERICAN': 'BLACK',
                 'OTHER': 'OTHER',
                 'HISPANIC/LATINO': 'HISPANIC',
                 'ASIAN': 'ASIAN',
                 'AMERICAN INDIAN/ALASKA NATIVE': 'NATIVE'
                 }

    pred_MIMIC = pd.read_csv("./results/bipred.csv")

    t = np.arange(4)
    fig, axs = plt.subplots(1, 2, sharey=False, figsize=(25, 10))
    fig.subplots_adjust(wspace=0.5)

    gap, percentage, cate = plot_14_MIMIC(pred_MIMIC, diseases_MIMIC, race, 'race')
    gap = list_to_array(gap)
    percentage = list_to_array(percentage)
    cate = np.array(cate)

    for g in range(gap.shape[0]):
        mask = gap[g, :] < 50
        s = np.multiply(percentage[g, :], 2000)
        axs[0].scatter(t[mask], gap[g, :][mask], s=s, marker='o', label=race_abbr[cate[g]].upper())
        axs[0].set_xticks(t)
        axs[0].set_xticklabels([diseases_abbr[k] for k in diseases_MIMIC])
        axs[0].set(xlabel='(a) RACE')
        axs[0].xaxis.set_label_position('top')
    axs[0].hlines([-0.2, 0, 0.2], xmin=0, xmax=3, linestyles='dotted', colors='red')
    axs[0].set(ylabel='TPR DISPARITY')
    axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
    axs[0].set_ylim([-0.25, 0.25])

    # for i in range(len(diseases_MIMIC)):
    #     for g in range(gap.shape[0]):
    #         s = np.multiply(percentage[g, i], 2000)
    #         if gap[g, i] < 50:
    #             axs[i].scatter(t[0], gap[g, i], s=s, marker='o', label=race_abbr[cate[g]])
    #         axs[i].set_xticks(t)
    #         axs[i].set_xticklabels(['RACE'], rotation=90)
    #         axs[i].set(xlabel=diseases_MIMIC[i])
    # axs[0].set(ylabel='TPR GAP')
    # axs[0].legend()

    gap, percentage, cate = plot_14_MIMIC(pred_MIMIC, diseases_MIMIC, insurance, 'insurance')
    gap = list_to_array(gap)
    percentage = list_to_array(percentage)
    cate = np.array(cate)

    for g in range(gap.shape[0]):
        mask = gap[g, :] < 50
        s = np.multiply(percentage[g, :], 2000)
        axs[1].scatter(t[mask], gap[g, :][mask], s=s, marker='o', label=cate[g].upper())
        axs[1].set_xticks(t)
        axs[1].set_xticklabels([diseases_abbr[k] for k in diseases_MIMIC])
        axs[1].set(xlabel='(b) INSURANCE')
        axs[1].xaxis.set_label_position('top')
    axs[1].hlines([-0.2, 0, 0.2], xmin=0, xmax=3, linestyles='dotted', colors='red')
    axs[1].set(ylabel='TPR DISPARITY')
    axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))
    axs[1].set_ylim([-0.25, 0.25])

    # for i in range(len(diseases_MIMIC)):
    #     for g in range(gap.shape[0]):
    #         s = np.multiply(percentage[g, i], 2000)
    #         if gap[g, i] < 50:
    #             axs[i+4].scatter(t[0], gap[g, i], s=s, marker='v', label=cate[g].upper())
    #         axs[i+4].set_xticks(t)
    #         axs[i+4].set_xticklabels(['INSURANCE'], rotation=90)
    #         axs[i+4].set(xlabel=diseases_MIMIC[i])
    # axs[7].legend()
    plt.savefig("./results/Combine_Spread_Protected_TPR.pdf")

# random_split("map.csv", "total_subject_id_with_gender.csv", [0.8, 0.9])

# def mean(df, diseases, category, category_name):
#     df = preprocess(df)
#     map_df = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/new_split/map.csv")
#     df = df.merge(map_df, left_on="subject_id", right_on="subject_id")
#     cate = []
#     mean_GAP = []
#     mean_abs_GAP = []
#     mean_percentage = []
#     for c in category:
#         try:
#             GAP_acc = 0
#             abs_GAP_acc = 0
#             percentage_acc = 0
#             for d in diseases:
#                 pred_disease = "bi_" + d
#                 gt = df.loc[(df[d] == 1) & (df[category_name] == c), :]
#                 pred = df.loc[(df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] == c), :]
#                 n_gt = df.loc[(df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0), :]
#                 n_pred = df.loc[
#                          (df[pred_disease] == 1) & (df[d] == 1) & (df[category_name] != c) & (df[category_name] != 0),
#                          :]
#                 TPR = len(pred) / len(gt)
#                 n_TPR = len(n_pred) / len(n_gt)
#                 GAP = TPR - n_TPR

#                 pi_gy = df.loc[(df[d] == 1) & (df[category_name] == c), :]
#                 pi_y = df.loc[(df[d] == 1) & (df[category_name] != 0), :]
#                 percentage = len(pi_gy) / len(pi_y)

#                 GAP_acc += GAP
#                 abs_GAP_acc += abs(GAP)
#                 percentage_acc += percentage
#             mean_GAP.append(GAP_acc / len(diseases))
#             mean_abs_GAP.append(abs_GAP_acc / len(diseases))
#             mean_percentage.append(percentage_acc / len(diseases))
#             cate.append(c)
#         except ZeroDivisionError:
#             pass
#     x = np.arange(len(cate))
#     plt.figure(figsize=(18, 9))
#     plt.plot(x, mean_GAP, marker='o', label='mean GAP')
#     plt.plot(x, mean_abs_GAP, marker='o', label='mean |GAP|')
#     plt.plot(x, mean_percentage, marker='o', label='mean %')
#     plt.xticks(x, cate)
#     plt.legend()
#     plt.savefig("./results/Mean_" + category_name + ".pdf")