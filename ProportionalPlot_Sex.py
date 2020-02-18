import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
#----------------------

NIH_sex = pd.read_csv("./NIH/results/WantX.csv")
CXP_sex = pd.read_csv("./CXP/results/WantX.csv")
MIMIC_sex = pd.read_csv("./MIMIC/results/WantX.csv")

print("Average Distance:",round(NIH_sex["Distance"].mean(),3))
print("Count female with negative gap:",len(NIH_sex.loc[(NIH_sex.Gap_F_mean < 0)]))
print("Count male with negative gap  :",len(NIH_sex.loc[(NIH_sex.Gap_M_mean < 0)]))

print("Average Distance:",round(CXP_sex["Distance"].mean(),3))
print("Count female with negative gap:",len(CXP_sex.loc[(CXP_sex.Gap_F_mean < 0)]))
print("Count male with negative gap  :",len(CXP_sex.loc[(CXP_sex.Gap_M_mean < 0)])-1) # -1 is to exclude No Finding as it is not a disease

print("Average Distance:",round(MIMIC_sex["Distance"].mean(),5))
print("Count female with negative gap:",len(MIMIC_sex.loc[(MIMIC_sex.Gap_F_mean < 0)]))
print("Count male with negative gap  :",len(MIMIC_sex.loc[(MIMIC_sex.Gap_M_mean < 0)])-1) # -1 is to exclude No Finding as it is not a disease



def func(x, m, b):
    return m*x + b

#diseases_abbr_NIH = {'Atelectasis': 'At',
   #             'Cardiomegaly': 'Cd',
        #        'Effusion': 'Ef',
     #           'Infiltration': 'In',
     #           'Mass': 'M',
      #          'Nodule': 'N',
      #          'Pneumonia': 'Pa',
      #          'Pneumothorax': 'Px',
      #          'Consolidation': 'Co',
      #          'Edema': 'Ed',
       #         'Emphysema': 'Em',
       #         'Fibrosis': 'Fb',
        #        'Pleural_Thickening': 'PT',
         #       'Hernia': 'H'
         #       }


diseases_abbr_NIH = {'Atelectasis': 'Atelectasis',
                'Cardiomegaly': 'Cardiomegaly',
                'Effusion': 'Effusion',
                'Infiltration': 'Infiltration',
                'Mass': 'Mass',
                'Nodule': 'Nodule',
                'Pneumonia': 'Pneumonia',
                'Pneumothorax': 'Pneumothorax',
                'Consolidation': 'Consolidation',
                'Edema': 'Edema',
                'Emphysema': 'Emphysema',
                'Fibrosis': 'Fibrosis',
                'Pleural_Thickening': 'Pleural Thickening',
                'Hernia': 'Hernia'
                }

diseases_abbr_CXP = {'Lung Opacity': 'AO',
                     'Cardiomegaly': 'Cd',
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

diseases_abbr_CXR = {'Airspace Opacity': 'AO',
                     'Cardiomegaly': 'Cd',
                     'Pleural Effusion': 'Ef',
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
                     'No Finding': 'NF'
                     }

plt.rcParams.update({'font.size': 22})

plt.figure(figsize=(18, 14))

params_CXR, covar = curve_fit(func, MIMIC_sex['%F'], MIMIC_sex['Gap_F_mean'], sigma=MIMIC_sex['CI_F'],
                              absolute_sigma=True)
plt.plot(MIMIC_sex['%F'], func(MIMIC_sex['%F'], params_CXR[0], params_CXR[1]), color='red', label='CXR(slope=1.51)')
plt.legend()

params_CXP, covar = curve_fit(func, CXP_sex['%F'], CXP_sex['Gap_F_mean'], sigma=CXP_sex['CI_F'], absolute_sigma=True)
plt.plot(CXP_sex['%F'], func(CXP_sex['%F'], params_CXP[0], params_CXP[1]), color='blue', label='CXP(slope=0.29)')
plt.legend()

params_NIH, covar = curve_fit(func, NIH_sex['%F'], NIH_sex['Gap_F_mean'], sigma=NIH_sex['CI_F'], absolute_sigma=True)
plt.plot(NIH_sex['%F'], func(NIH_sex['%F'], params_NIH[0], params_NIH[1]), color='green', label='NIH(slope=0.73)')
plt.legend()

# ------------------- CXP
plt.scatter(CXP_sex['%F'], CXP_sex['Gap_F_mean'], marker='o', color='blue', label='CXP')
# plt.errorbar(CXP_sex['%F'],CXP_sex['Gap_F_mean'],yerr = CXP_sex['CI_F'],fmt='o',mfc='blue')

#for d, x, y in zip(CXP_sex['diseases'], CXP_sex['%F'], CXP_sex['Gap_F_mean']):
 #   plt.annotate(diseases_abbr_CXP[d], color='blue', xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='right',
 #                va='bottom')

diseases_CXP = np.array(len(diseases_abbr_CXP))
sigma_ab = np.sqrt(np.diagonal(covar))

# plotting the confidence intervals
hires_x = np.linspace(0.278, 0.447, 100)
bound_upper = func(hires_x, *(params_CXP + sigma_ab))
bound_lower = func(hires_x, *(params_CXP - sigma_ab))
plt.fill_between(hires_x, bound_lower, bound_upper, color='blue', alpha=0.15)
#

plt.xlabel("% FEMALE PER DISEASE")
plt.ylabel("TPR DISPARITY")

# --------------------------------------------- MIMIC
plt.scatter(MIMIC_sex['%F'], MIMIC_sex['Gap_F_mean'], marker='+', color='red', label='CXR')
# plt.errorbar(CXR_sex['%F'],CXR_sex['Gap_F_mean'],yerr = CXR_sex['CI_F'],fmt='o',mfc='green')

# for d, x, y in zip(MIMIC_sex['diseases'], MIMIC_sex['%F'], MIMIC_sex['Gap_F_mean']):
#      plt.annotate(diseases_abbr_CXR[d], color='red', xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='left', va='bottom')


diseases_CXR = np.array(len(diseases_abbr_CXR))
sigma_ab = np.sqrt(np.diagonal(covar))

# plotting the confidence intervals
hires_x = np.linspace(0.333, 0.536, 100)
bound_upper = func(hires_x, *(params_CXR + sigma_ab))
bound_lower = func(hires_x, *(params_CXR - sigma_ab))
plt.fill_between(hires_x, bound_lower, bound_upper, color='red', alpha=0.15)
#

plt.xlabel("% FEMALE PER DISEASE")
plt.ylabel("TPR DISPARITY")
# plt.legend('NIH (slope=' + str(round(params_NIH[0], 2)) + ')')


# --------------------------------------- 'NIH'
plt.scatter(NIH_sex['%F'], NIH_sex['Gap_F_mean'], marker='*', color='green', label='NIH')
# plt.errorbar(NIH_sex['%F'],NIH_sex['Gap_F_mean'],yerr = NIH_sex['CI_F'],fmt='*',mfc='red')

for d, x, y in zip(NIH_sex['diseases'], NIH_sex['%F'], NIH_sex['Gap_F_mean']):
    plt.annotate(diseases_abbr_NIH[d], color='green', xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='left',
                 va='bottom')

diseases_NIH = np.array(len(diseases_abbr_NIH))
sigma_ab = np.sqrt(np.diagonal(covar))

# plotting the confidence intervals
hires_x = np.linspace(0.36, 0.885, 100)
bound_upper = func(hires_x, *(params_NIH + sigma_ab))
bound_lower = func(hires_x, *(params_NIH - sigma_ab))
plt.fill_between(hires_x, bound_lower, bound_upper, color='green', alpha=0.15)
#

plt.xlabel("% FEMALE PER DISEASE")
plt.ylabel("TPR DISPARITY")
plt.grid()
plt.savefig('./TPRSEXProportional7.pdf')

print("CXR curvefit parameters", params_CXR)
print("CXP curvefit parameters", params_CXP)
print("NIH curvefit parameters", params_NIH)


#-------------

plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(18,18))

fig, ((ax1, ax2, ax3)) = plt.subplots(3,sharex=True, sharey=True, gridspec_kw={'hspace': 0})

fig.suptitle('TPR Disparities vs. Female Proportion per Disease')

params_CXR, covar = curve_fit(func, MIMIC_sex['%F'], MIMIC_sex['Gap_F_mean'], sigma=MIMIC_sex['CI_F'], absolute_sigma=True)
ax1.plot(MIMIC_sex['%F'], func(MIMIC_sex['%F'], params_CXR[0], params_CXR[1]), color='red', label='CXR(slope=1.514)' )
ax1.legend()   

params_CXP, covar = curve_fit(func, CXP_sex['%F'], CXP_sex['Gap_F_mean'], sigma=CXP_sex['CI_F'], absolute_sigma=True)
ax2.plot(CXP_sex['%F'], func(CXP_sex['%F'], params_CXP[0], params_CXP[1]), color='blue', label='CXP(slope=0.293)' )
ax2.legend()   

params_NIH, covar = curve_fit(func, NIH_sex['%F'],NIH_sex['Gap_F_mean'], sigma=NIH_sex['CI_F'], absolute_sigma=True)
ax3.plot(NIH_sex['%F'], func(NIH_sex['%F'], params_NIH[0], params_NIH[1]), color='green', label='NIH(slope=0.726)' )
ax3.legend()   

#------------------- CXP
ax2.scatter(CXP_sex['%F'],CXP_sex['Gap_F_mean'], marker='o',color='blue', label='CXP')
#plt.errorbar(CXP_sex['%F'],CXP_sex['Gap_F_mean'],yerr = CXP_sex['CI_F'],fmt='o',mfc='blue')

for d, x, y in zip(CXP_sex['diseases'], CXP_sex['%F'], CXP_sex['Gap_F_mean']):
    ax2.annotate(diseases_abbr_CXP[d], color='blue', xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='right', va='bottom')

diseases_CXP = np.array(len(diseases_abbr_CXP))
sigma_ab = np.sqrt(np.diagonal(covar))   

# plotting the confidence intervals
hires_x = np.linspace(0.278, 0.447, 100)    
bound_upper = func(hires_x, *(params_CXP + sigma_ab))
bound_lower = func(hires_x, *(params_CXP - sigma_ab))
ax2.fill_between(hires_x, bound_lower, bound_upper, color = 'blue', alpha = 0.15)
#
    
# ax2.xlabel("% FEMALE PER DISEASE")
# ax2.ylabel("TPR DISPARITY")

#--------------------------------------------- MIMIC 
ax1.scatter(MIMIC_sex['%F'],MIMIC_sex['Gap_F_mean'], marker='s',color='red', label='CXR')
#plt.errorbar(CXR_sex['%F'],CXR_sex['Gap_F_mean'],yerr = CXR_sex['CI_F'],fmt='o',mfc='green')

for d, x, y in zip(MIMIC_sex['diseases'], MIMIC_sex['%F'], MIMIC_sex['Gap_F_mean']):
      ax1.annotate(diseases_abbr_CXR[d], color='red', xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='left', va='bottom')
    


diseases_CXR = np.array(len(diseases_abbr_CXR))
sigma_ab = np.sqrt(np.diagonal(covar))   

# plotting the confidence intervals
hires_x = np.linspace(0.333, 0.536, 100)    
bound_upper = func(hires_x, *(params_CXR + sigma_ab))
bound_lower = func(hires_x, *(params_CXR - sigma_ab))
ax1.fill_between(hires_x, bound_lower, bound_upper, color = 'red', alpha = 0.15)
#
    
# ax1.xlabel("% FEMALE PER DISEASE")
# ax1.ylabel("TPR DISPARITY")
#plt.legend('NIH (slope=' + str(round(params_NIH[0], 2)) + ')')   




#--------------------------------------- 'NIH'
ax3.scatter(NIH_sex['%F'],NIH_sex['Gap_F_mean'], marker='*',color='green', label='NIH')
#plt.errorbar(NIH_sex['%F'],NIH_sex['Gap_F_mean'],yerr = NIH_sex['CI_F'],fmt='*',mfc='red')

for d, x, y in zip( NIH_sex['diseases'], NIH_sex['%F'], NIH_sex['Gap_F_mean']):
    ax3.annotate(diseases_abbr_NIH[d], color='green', xy=(x, y), xytext=(-3, 3), textcoords='offset points', ha='left', va='bottom')


diseases_NIH = np.array(len(diseases_abbr_NIH))
sigma_ab = np.sqrt(np.diagonal(covar))   

# plotting the confidence intervals
hires_x = np.linspace(0.36, 0.885, 100)    
bound_upper = func(hires_x, *(params_NIH + sigma_ab))
bound_lower = func(hires_x, *(params_NIH - sigma_ab))
ax3.fill_between(hires_x, bound_lower, bound_upper, color = 'green', alpha = 0.15)
#
    
# ax3.xlabel("% FEMALE PER DISEASE")
# ax3.ylabel("TPR DISPARITY")

fig.savefig('./TPRSEXProportional5.pdf')

print("CXR curvefit parameters",params_CXR)
print("CXP curvefit parameters",params_CXP)
print("NIH curvefit parameters",params_NIH)
