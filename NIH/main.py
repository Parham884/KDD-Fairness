import torch
from train import *

from LearningCurve import *
from predictions import *
from nih import *
import pandas as pd

#---------------------- on q
path_image = "/scratch/gobi2/projects/ml4h/datasets/NIH/images/"

train_df_path ="/scratch/gobi2/projects/ml4h/datasets/NIH/split/July16/train.csv"
test_df_path = "/scratch/gobi2/projects/ml4h/datasets/NIH/split/July16/test.csv"
val_df_path = "/scratch/gobi2/projects/ml4h/datasets/NIH/split/July16/valid.csv"


diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
# age_decile = ['0-20', '20-40', '40-60', '60-80', '80-']
age_decile = ['40-60', '60-80', '20-40', '80-', '0-20']
gender = ['M', 'F']

def main():

    MODE = "plot"  # Select "train" or "test", "Resume", "plot", "Threshold"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_df_path)
    train_df_size = len(train_df)
    print("Train_df path", train_df_size)

    test_df = pd.read_csv(test_df_path)
    test_df_size = len(test_df)
    print("test_df path", test_df_size)

    val_df = pd.read_csv(val_df_path)
    val_df_size = len(val_df)
    print("val_df path", val_df_size)

    if MODE == "train":
        ModelType = "densenet"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, ModelType, CriterionType, device,LR)

        PlotLearnignCurve()


    if MODE =="test":
        val_df = pd.read_csv(val_df_path)
        test_df = pd.read_csv(test_df_path)

        CheckPointData = torch.load('results/checkpoint')
        model = CheckPointData['model']

        make_pred_multilabel(model, test_df, val_df, path_image, device)


    if MODE == "Resume":
        ModelType = "Resume"  # select 'ResNet50','densenet','ResNet34', 'ResNet18'
        CriterionType = 'BCELoss'
        LR = 0.5e-3

        model, best_epoch = ModelTrain(train_df_path, val_df_path, path_image, result_path, ModelType, CriterionType, device,LR)

        PlotLearnignCurve()

    if MODE == "plot":
        gt = pd.read_csv("./results/True.csv")
        pred = pd.read_csv("./results/bipred.csv")
        factor = [gender, age_decile]
        factor_str = ['Patient Gender', 'Patient Age']
        for i in range(len(factor)):
            # plot_frequency(gt, diseases, factor[i], factor_str[i])
            # plot_TPR_NIH(pred, diseases, factor[i], factor_str[i])
            plot_sort_median(pred, diseases, factor[i], factor_str[i])
            # distance_max_min(pred, diseases, factor[i], factor_str[i])


    # if MODE == "mean":
    #     pred = pd.read_csv("./results/bipred.csv")
    #     factor = [gender, age_decile]
    #     factor_str = ['Patient Gender', 'Patient Age']
    #     for i in range(len(factor)):
    #         mean(pred, diseases, factor[i], factor_str[i])

    # if MODE == "plot_14":
    #     pred = pd.read_csv("./results/bipred.csv")
    #     factor = [gender, age_decile]
    #     factor_str = ['Patient Gender', 'Patient Age']
    #     for i in range(len(factor)):
    #         #    plot_14(pred, diseases, factor[i], factor_str[i])
    #         plot_14_old(pred, diseases, factor[i], factor_str[i])
    #         plot_Median(pred, diseases, factor[i], factor_str[i])


if __name__ == "__main__":
    main()
