import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV
#%%
#load data
file_path = "/datasets/corona/data_1.csv"
data = pd.read_csv(file_path)
#rename data columns
data = data.rename(columns={"date_onset_symptoms": "onset", "date_admission_hospital": "admission", "date_confirmation" : "confirmation"
                     , "date_death_or_discharge" : "death_discharge"})
# indexing rows with no missing values
index_onset = ~data['onset'].isna()
index_admission = ~data['admission'].isna()
index_confirmation = ~data['confirmation'].isna()
index_death_discharge = ~data['death_discharge'].isna()

'''a = index_onset | index_admission  | index_confirmation | index_death_discharge
#data = data[a]'''
# removing rows where these features are missing simultaneusly
a = index_onset | index_admission  | index_death_discharge
data = data[a]
data = data.reset_index()
data =  data.drop(['index'], axis=1)

#unifying the names of outcomes
for i in range(np.shape(data)[0]):
    if data['outcome'][i] == 'discharged':
        data['outcome'][i] = 'discharge'
    elif data['outcome'][i] == 'Discharged':
        data['outcome'][i] = 'discharge'
    elif data['outcome'][i] == 'Death':
        data['outcome'][i] = 'death'
    elif data['outcome'][i] == 'died':
        data['outcome'][i] = 'death'
    elif data['outcome'][i] == 'dead':
        data['outcome'][i] = 'death'
    elif data['outcome'][i] == 'released from quarantine':
        data['outcome'][i] = 'discharge'
    elif data['outcome'][i] == 'stable':
        data['outcome'][i] = 'discharge'
    elif data['outcome'][i] == 'recovered':
        data['outcome'][i] = 'discharge'
    elif data['outcome'][i] == 'severe':
        data['outcome'][i] = 'death'
    elif data['outcome'][i] == 'severe':
        data['outcome'][i] = 'death'

#removing unknown outcomes
data = data.drop([79,614])
#data['outcome'] = data['outcome'].astype('category')

#we only analyse dischaarges, so we remove deaths
a = data['outcome'] == 'death'
a = ~a
data = data[a]
data = data.reset_index()
data =  data.drop(['index'], axis=1)

#indexing dates
index_onset = ~data['onset'].isna()
index_admission = ~data['admission'].isna()
index_confirmation = ~data['confirmation'].isna()
index_death_discharge = ~data['death_discharge'].isna()
#assumption making about dates
for i in range(np.shape(data)[0]):
    if index_onset[i] == False and index_admission[i] == True:
        data['onset'][i] = data['admission'][i]
    elif index_onset[i] == False and index_admission[i] == False and index_confirmation[i] == True:
        data['onset'][i] = data['confirmation'][i]


#reformatting dates
data['onset'] = data['onset'].apply(pd.to_datetime, dayfirst=True)
data['confirmation'] = data['confirmation'].apply(pd.to_datetime, dayfirst=True)
data['death_discharge'] = data['death_discharge'].apply(pd.to_datetime, dayfirst=True)

# cesnsorship status
index_censoring = ~data['death_discharge'].isna()
#calculating event and censoring days
event_censoring_days = []
for i in range(np.shape(data)[0]):
    if index_censoring[i] == False:
        event_censoring_days = np.append(event_censoring_days,data['confirmation'].dt.dayofyear[i] - data['onset'].dt.dayofyear[i] )
    elif index_censoring[i] == True:
        event_censoring_days = np.append(event_censoring_days, data['death_discharge'].dt.dayofyear[i] - data['onset'].dt.dayofyear[i])
# becasuse of some inconsistencies, some event-censoring days are negative, se we remove them and adjust the data
a = event_censoring_days > 0
event_censoring_days = event_censoring_days[a]
data = data[a]
data = data.reset_index()
data =  data.drop(['index'], axis=1)
index_censoring = index_censoring[a]

# creating structured label
frm = { 'event_censoring_days' : event_censoring_days, 'status': index_censoring}
data_y_unstructured = pd.DataFrame(data=frm)
data_y_unstructured = data_y_unstructured[['status', 'event_censoring_days']]
s = data_y_unstructured.dtypes
data_y = np.array([tuple(x) for x in data_y_unstructured.values], dtype=list(zip(s.index, s)))

gender = data['sex']
gender = gender.astype('category')

#%%
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

for value in gender.unique():
    mask = gender == value
    if value == 'male':
        time_cell, survival_prob_cell = kaplan_meier_estimator(data_y["status"][mask],data_y["event_censoring_days"][mask])
        plt.step(time_cell, 1 - survival_prob_cell, where="post", color = 'r',label="%s" % (value))
    else:
        time_cell, survival_prob_cell = kaplan_meier_estimator(data_y["status"][mask],data_y["event_censoring_days"][mask])
        plt.step(time_cell, 1 - survival_prob_cell, where="post",color = 'b',label="%s" % (value))
plt.ylabel("est. probability of discharge $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
limit = plt.gca()
limit.set_xlim([0, 42])
limit.set_ylim([0, 1])
plt.savefig('corona_gender_onset.pdf')