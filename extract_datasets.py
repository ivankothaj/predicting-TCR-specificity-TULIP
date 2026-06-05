import pandas as pd

# creates subset from BATCAVE database which we use for general evaluation of TULIP
def extract_BATCAVE():

    # reads database
    file = pd.read_excel("data/BATCAVE.xlsx")

    # takes only necessary columns from the dataset
    file = file[["Unnamed: 0", "cdr3b", "cdr3a", "peptide", "mhc", "activation", "peptide_activity"]]

    # renames column names to match CSV input format of predict.py
    file = file.rename(columns={
        "Unnamed: 0" : "index",
        "cdr3b" : "CDR3b",
        "cdr3a" : "CDR3a",
        "mhc" : "MHC",
        "activation" : "binder",
        "peptide_activity" : "peptide activity",
    })

    # filters out rows containing either mouse data or rows missing TCRs
    file = pd.concat([
        file.iloc[5730:8157],
        file.iloc[13359:15069],
        file.iloc[15624:16480],
        file.iloc[16653:21336],
        file.iloc[21719:22482]
    ])

    # converts ternary binder labels to binary binder labels for AUC-ROC computation
    file["binder"] = (file["binder"] >= 1).astype(int)

    # replaces or MHCs with uniforma placeholder string
    # used for experiment where we checked whether MHC affects prediction
    # file["MHC"] = "placeholder"

    # creates CSV file containing input data for predict.py
    file[["index", "CDR3b", "CDR3a", "peptide", "MHC", "binder", "peptide activity"]
            ].to_csv("data/BATCAVE_subset.csv", index=False)
    

# creates subset from data sample with index peptides and their mutants
def extract_mutant_subset():

    # reads dataset
    file = pd.read_csv("data/index_peptides_uniqueTCRs.csv")

    # renames column names to match CSV input format of predict.py
    file = file.rename(columns={
        "cdr3b" : "CDR3b",
        "cdr3a" : "CDR3a",
        "mhc" : "MHC",
        "index_peptide" : "index peptide",
        "activation" : "binder",
        "peptide_activity" : "peptide activity",
    })

    # takes only necessary columns from the dataset
    file = file[["CDR3b", "CDR3a", "peptide", "MHC", "binder", "index peptide", "peptide activity"]]

    # filters out rows missing TCRs
    file = pd.concat([
        file.iloc[:1023],
        file.iloc[1369:]
    ])

    # converts ternary binder labels to binary binder labels for AUC-ROC computation
    file["binder"] = (file["binder"] >= 1).astype(int)

    # creates CSV file containing input data for predict.py
    file[["CDR3b", "CDR3a", "peptide", "MHC", "index peptide", "binder", "peptide activity"]
            ].to_csv("data/mutant_subset.csv", index=True, index_label="index")

extract_BATCAVE()
extract_mutant_subset()
