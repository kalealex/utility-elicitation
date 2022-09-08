import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.special import logit

# utils:
# override sort order within label encoder
class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self

def code_string(df, colname):
    # make y numeric
    le = MyLabelEncoder()
    # arr = df[[colname]].values.flatten() # if this was  arr = df[colname].values, would we still need to flatten and reshape?
    # le = le.fit(arr)
    # result = le.transform(arr)
    # result = result.reshape(len(arr), 1)
    # df[[colname]] = result
    arr = df[colname].values
    le = le.fit(arr)
    df[colname] = le.transform(arr)

    return (df, le)

RECODE_SET = ['disposition', 'dep_name', 'gender', 'ethnicity', 'race', 'lang', 'religion', 'maritalstatus', 'employstatus', 'insurance_status', 'arrivalmode', 'previousdispo']


# load data
filename = 'hospital-admissions.csv'
df = pd.read_csv(filename)

# prepare dataset:
# eliminate timeseries variables
for var in ['arrivalmonth', 'arrivalday', 'arrivalhour_bin']:
    df.pop(var)
# TODO: normalize X values?
# make sure discharge maps to 0 when we use MyLabelEncoder
df.sort_values(by = 'disposition', ascending = True) 
# call label encoder to code strings as numbers for modeling
label_encoders = {}
for var in RECODE_SET:
    df, label_encoders[var] = code_string(df, var)
# separate features and target
y = df.pop('disposition').values
X = df

# train/test/waiting room split
seed = 42
train_prop = 0.7
waiting_room = 200
train_X, holdout_X, train_y, holdout_y = train_test_split(
        X,
        y,
        test_size = 1.0 - train_prop,
        random_state = seed,
        stratify = y
    )
test_X, waiting_X, test_y, waiting_y = train_test_split(
        holdout_X,
        holdout_y,
        test_size = waiting_room, # our 'test set' in this split is our waiting room
        random_state = seed,
        stratify = holdout_y
    )

# TODO: remove columns for protected variables from train and test sets?

# train ML model
model = XGBClassifier()
model.fit(train_X, train_y)

# test accuracy
pred_y = model.predict(test_X)
pred_y_bi = [round(prob) for prob in pred_y]
accuracy = accuracy_score(test_y, pred_y_bi)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# generate risk scores for waiting room:
# get probabilities from model
prob_admit = model.predict(waiting_X)
# nudge 0 and 1 probabilities
adjust_p = []
for p in prob_admit:
    if p == 1.0:
        adjust_p.append(p - 0.01)
    elif p == 0.0:
        adjust_p.append(p + 0.01)
    else:
        adjust_p.append(p)
adjust_p = np.array(adjust_p)
# convert to logit risk scores
risk_score = logit(adjust_p)

# format output dataframe
output = waiting_X[['race', 'gender', 'age', 'insurance_status', 'arrivalmode', 'cc_chestpain', 'cc_fever', 'cc_fatigue', 'cc_hypertension', 'cc_seizures', 'sicklecell', 'pulmhartdx']]
output['prob_admit'] = prob_admit
output['risk_score'] = risk_score
# backtransform coded strings
for var in RECODE_SET:
    if var in output.columns:
        output[var] = label_encoders[var].inverse_transform(output[var])

print(output.head(10))

# save out csv for 
output.to_csv('ml-output.csv')