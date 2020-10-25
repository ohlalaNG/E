import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation

data = pd.read_csv('file:///E:\Python%20projects\Datasets\SC_P.csv')
# print(data.shape)
# print(data.columns)
extracted_data = data[['SC', 'RC', 'CC', 'AGE', 'EXP', 'SEX', 'EDU', 'EMP', 'INCOME', 'COG',
                       'Info_Pre', 'Info_Aft', 'R_Per', 'R_Taking', 'Scenario', 'DEV1', 'DEV2']]
#print(extracted_data.head().to_string())

#visualisation
def correlationmatrix():
    corr_mat = extracted_data.corr()
    print(round(corr_mat,2).to_string())
    ax = sb.heatmap(corr_mat,
                    vmin = -1, vmax =1, center = 0,
                    cmap = sb.diverging_palette(20,200,n = 30),
                    square = True)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 45,
        horizontalalignment = 'right')
    plt.show()



def reg_coeff(x,y):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    return model.coef_

# reg_coeff(extracted_data[['SC','RC','CC']], extracted_data['Info_Aft'])
# reg_coeff(extracted_data[['Info_aft']], extracted_data['DEV1'])
# reg_coeff(extracted_data[['Info_aft']], extracted_data['DEV2'])

#Scenario as moderator
def scenarioasmoderator():

    fig = plt.figure(figsize=(17,9))

    fig.add_subplot(221)
    s0=extracted_data.loc[extracted_data['Scenario'] == 0]
    sb.regplot(x=s0['RC'],y=s0['Info_Aft'], fit_reg=True,)
    plt.xlabel("Relational capital")
    plt.ylabel("Perceived Social Capital")

    fig.add_subplot(222)
    s0=extracted_data.loc[extracted_data['Scenario'] == 1]
    sb.regplot(x=s0['RC'],y=s0['Info_Aft'], fit_reg=True,)
    plt.xlabel("Relational capital")
    plt.ylabel("Perceived Social Capital")

    fig.add_subplot(223)
    s0=extracted_data.loc[extracted_data['Scenario'] == 2]
    sb.regplot(x=s0['RC'],y=s0['Info_Aft'], fit_reg=True,)
    plt.xlabel("Relational capital")
    plt.ylabel("Perceived Social Capital")

    fig.tight_layout()
    plt.show()

def pre_aft_per():
    outcome_model = sm.OLS.from_formula('DEV1 ~ Info_Pre + Info_Aft', data = extracted_data)
    mediator_model = sm.OLS.from_formula('Info_Aft ~ Info_Pre', data = extracted_data)
    med = Mediation(outcome_model, mediator_model, 'Info_Pre', 'Info_Aft')
    med_result = med.fit(n_rep=500)
    print(med_result.summary())

correlationmatrix()
