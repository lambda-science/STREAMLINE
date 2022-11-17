from sklearn import preprocessing
import pandas as pd
import numpy as np
import sys

raw_df = pd.read_csv("EHRoes/text_reports.csv")

# Modify Onto features values: on modifie les valeur: 0:-1, puis on modifie les N/A en 0
# Get list of ontology columns
onto_col = []
for column in raw_df.columns.to_list():
    if column[0:4] == "MHO:":
        onto_col.append(column)

#  Simplify quantity
raw_df[onto_col] = raw_df[onto_col].replace({0: 0, 0.25: 1, 0.5: 1, 0.75: 1})

# One-hot Encode Cat Variable
# Needed if gene_diag included in analysis
# raw_df = pd.get_dummies(raw_df, columns=["gene_diag"])

# Label Encode Conclusion
raw_df["conclusion"].replace(
    {
        "COM_CCD": "COM",
        "UNCLEAR": np.NaN,
        "NM_CAP": "NM",
        "CFTD": np.NaN,
        "COM_MMM": "COM",
        "NON_CM": np.NaN,
        "CM": np.NaN,
    },
    inplace=True,
)
le = preprocessing.LabelEncoder()
fit_by = pd.Series([i for i in raw_df["conclusion"].unique() if type(i) == str])
le.fit(fit_by)
raw_df["conclusion"] = raw_df["conclusion"].map(
    lambda x: le.transform([x])[0] if type(x) == str else x
)
print("Class Label: ")
for index, value in enumerate(le.classes_):
    print("Class Value: ", index, " Class Name: ", value)

# Delete columns with all missing values
raw_df[onto_col].dropna(how="all", axis=1, inplace=True)

raw_df[onto_col] = raw_df[onto_col].fillna(0)

raw_df.to_csv(sys.argv[1] + "/input.csv", sep=",", index=False)
