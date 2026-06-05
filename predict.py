
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig
from sklearn.metrics import roc_auc_score

import torch

from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from src.multiTrans import TulipPetal, TCRDataset, BertLastPooler, unsupervised_auc, train_unsupervised, eval_unsupervised, MyMasking, Tulip, get_logscore

import argparse

import os
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


torch.manual_seed(0)

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--test_dir",
        default=None,
        type=str,
        required=True,
        help="The test data dir. Should contain the .fasta files (or other data files) for the task.",
    )
    parser.add_argument(
        "--modelconfig",
        default="configs/shallow.config.json",
        type=str,
        help="path to json including the config of the model" ,
    )
    parser.add_argument(
        "--load",
        default="model_weights/pytorch_model.bin",
        type=str,
        help="path to the model pretrained to load" ,
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="path to save results" ,
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="batch_size" ,
    )

    args = parser.parse_args()

    with open(args.modelconfig, "r") as read_file:
        print("loading hyperparameter")
        modelconfig = json.load(read_file)



    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    test_path = args.test_dir


    tokenizer = AutoTokenizer.from_pretrained("aatok/")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<MIS>'})
        
    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '<CLS>'})

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<EOS>'})

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<MASK>'})

    from tokenizers.processors import TemplateProcessing
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <EOS>",
        pair="<CLS> $A <MIS> $B:1 <EOS>:1",
        special_tokens=[
            ("<EOS>", 2),
            ("<CLS>", 3),
            ("<MIS>", 4),
        ],
    )

    mhctok = AutoTokenizer.from_pretrained("mhctok/")
    vocabsize = len(tokenizer._tokenizer.get_vocab())
    mhcvocabsize = len(mhctok._tokenizer.get_vocab())
    print(mhcvocabsize)
    print("Loading models ..")
    # vocabsize = encparams["vocab_size"]

    max_length = 50
    encoder_config = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = modelconfig["num_attn_heads"],
                        num_hidden_layers = modelconfig["num_hidden_layers"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        pad_token_id =  tokenizer.pad_token_id)

    encoder_config.mhc_vocab_size  =mhcvocabsize

    encoderA = BertModel(config=encoder_config)
    encoderB = BertModel(config=encoder_config)
    encoderE = BertModel(config=encoder_config)

    max_length = 100
    max_length = 50
    decoder_config = BertConfig(vocab_size = vocabsize,
                        max_position_embeddings = max_length, # this shuold be some large value
                        num_attention_heads = modelconfig["num_attn_heads"],
                        num_hidden_layers = modelconfig["num_hidden_layers"],
                        hidden_size = modelconfig["hidden_size"],
                        type_vocab_size = 1,
                        is_decoder=True,
                        pad_token_id =  tokenizer.pad_token_id)    # Very Important

    decoder_config.add_cross_attention=True

    decoderA = TulipPetal(config=decoder_config) #BertForMaskedLM
    decoderA.pooler = BertLastPooler(config=decoder_config)
    decoderB = TulipPetal(config=decoder_config) #BertForMaskedLM
    decoderB.pooler = BertLastPooler(config=decoder_config)
    decoderE = TulipPetal(config=decoder_config) #BertForMaskedLM
    decoderE.pooler = BertLastPooler(config=decoder_config)


    # Define encoder decoder model
    

    model = Tulip(encoderA=encoderA,encoderB=encoderB,encoderE=encoderE, decoderA=decoderA, decoderB=decoderB, decoderE=decoderE)
    if torch.cuda.is_available():
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint)
        
    else:
        checkpoint = torch.load(args.load, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
    model.to(device)

######################################## MODIFIED CODE BELOW ########################################

    if test_path == "data/BATCAVE_subset.csv":
        # evaluates TULIP on the majority of the BATCAVE database
        predict_BATCAVE(test_path, model, tokenizer, device, mhctok)

    elif test_path == "data/mutant_subset.csv":
        # evaluates TULIP on the small dataset with index peptides and their mutants
        predict_mutants(test_path, model, tokenizer, device, mhctok)

    else:
        print("Incorrect path!")


def predict_BATCAVE(test_path, model, tokenizer, device, mhctok):

    # threshould used for binomial distribution
    boundary_threshold = -20
   
    # reads dataset
    dataset = pd.read_csv(test_path)
    
    # filters unique target peptides used for grouping
    target_peptides = pd.read_csv(test_path)["peptide"].unique()

    # output file name
    output_file = "data/predict_results.csv"

    # removes old output file
    if os.path.exists(output_file):
        os.remove(output_file)

    # set used to collect data for overall evaluation
    final_results = []

    # loops through the whole dataset making groups of same peptides with different TCRs
    for i, target_peptide in enumerate(target_peptides):
        results = pd.DataFrame(columns=["CDR3a", "CDR3b", "peptide", "MHC" "score"])
        modified_dataset = TCRDataset(test_path, tokenizer, device, target_peptide=target_peptide, mhctok=mhctok)

        # raw log-likelihoods (multiplied with -1): closer to 0 = higher log_probability => stronger binder
        scores = -1*np.array(get_logscore(modified_dataset, model, ignore_index =  tokenizer.pad_token_id))
        
        results["CDR3a"] = modified_dataset.alpha
        results["CDR3b"] = modified_dataset.beta
        results["peptide"] = target_peptide
        results["MHC"] = modified_dataset.MHC
        results["score"] = scores

        # creates evaluation subset for current target_peptide
        eval_sorted = dataset[dataset["peptide"] == target_peptide].copy()

        # merges prediction results with evaluation data for the current target_peptide
        merged_output = results.merge(
            eval_sorted[["index", "CDR3a", "CDR3b", "peptide", "MHC", "binder", "peptide activity"]], 
            on=["CDR3a", "CDR3b", "peptide", "MHC"])
        merged_output = merged_output[["index", "CDR3a", "CDR3b", "peptide", "MHC", "binder", "peptide activity", "score"]]

        # data transformation by binomial distribution that splits data into two binder classes
        # used to compute additional AUC for an experiment describe in the result section of the thesis
        merged_output["predicted_binder"] = (merged_output["score"] > boundary_threshold).astype(int)

        # checks whether there are at least two same peptides with different TCRs, otherwise writes np.nan
        if len(merged_output) >= 2:
            score_var = merged_output["score"].nunique()
            peptide_activity_var = merged_output["peptide activity"].nunique()
            
            # checks whether variance of score and peptide activity is greater than zero, if not Pearson cannot be computed
            if score_var > 0 and peptide_activity_var > 0:
                # computes Pearson correlation between score and peptide activity and adds it to the merged output
                merged_output["pearson_corr"] = stats.pearsonr(merged_output["peptide activity"], merged_output["score"])[0]

            else:
                merged_output["pearson_corr"] = np.nan

            # computes AUC and adds it to the merged output
            # AUC is computed only if the target peptide contains both labels, otherwise writes np.nan
            if len(np.unique(merged_output["binder"])) == 2:
                merged_output["AUC"] = roc_auc_score(merged_output["binder"], merged_output["score"])
            else:
                merged_output["AUC"] = np.nan
            
        else:
            merged_output["pearson_corr"] = np.nan
            merged_output["AUC"] = np.nan

        # writes results into output file and puts an empty line between each target peptide group
        merged_output.to_csv(output_file, mode='a', header=(i==0), index=False)
        with open(output_file, 'a') as file:
            file.write("\n")

        # adds results of the target_peptide to set of all results for overall computations
        final_results.append(merged_output)

    # concatenates all results together
    final_results = pd.concat(final_results, ignore_index=True)
    
    # creates scatterplot with data distribution and computes Pearson correlation
    plot_pearson(final_results["peptide activity"], final_results["score"], "BATCAVE_data_distribution")

    # plots AUC between binder and predicted binder
    plot_AUC(final_results["binder"], final_results["predicted_binder"], "AUC", "AUC_binder_vs_predicted_binder")

    # plots AUC between binder and predicted score
    plot_AUC(final_results["binder"], final_results["score"], "AUC", "AUC_binder_vs_score")

    # plots AUC_0.1 between binder and predicted score
    plot_AUC(final_results["binder"], final_results["score"], "AUC_0.1", "AUC_0.1_binder_vs_score")


def predict_mutants(test_path, model, tokenizer, device, mhctok):

    # reads dataset
    dataset = pd.read_csv(test_path)

    # filters unique index peptides used for grouping
    index_peptides = dataset["index peptide"].unique()  

    # set used to collect data for overall evaluation
    final_results = []

    # loops throught the whole dataset making groups of same index peptides with different mutants
    for i, index_peptide in enumerate(index_peptides):
        mutant_peptides = dataset[dataset["index peptide"] == index_peptide]["peptide"].unique()

        # computes scores for mutant peptides
        for j, mutant_peptide in enumerate(mutant_peptides):
            results = pd.DataFrame(columns=["CDR3a", "CDR3b", "peptide", "MHC", "score"])
            modified_dataset = TCRDataset(test_path, tokenizer, device, target_peptide=mutant_peptide, mhctok=mhctok)

            # raw log-likelihoods (multiplied with -1): closer to 0 = higher log_probability => stronger binder
            scores = -1*np.array(get_logscore(modified_dataset, model, ignore_index =  tokenizer.pad_token_id))
            
            results["CDR3a"] = modified_dataset.alpha
            results["CDR3b"] = modified_dataset.beta
            results["peptide"] = mutant_peptide
            results["MHC"] = modified_dataset.MHC
            results["score"] = scores

            # creates evaluation subset for current target_peptide
            eval_sorted = dataset[dataset["peptide"] == mutant_peptide].copy()

            # merges prediction results with evaluation data for the current target_peptide
            merged_output = results.merge(
                eval_sorted[["index", "CDR3a", "CDR3b", "peptide", "MHC", "binder", "peptide activity"]], 
                on=["CDR3a", "CDR3b", "peptide", "MHC"])
            merged_output = merged_output[["index", "CDR3a", "CDR3b", "peptide", "MHC", "binder", "peptide activity", "score"]]

            # adds results of the target peptide to set of all results for overall computations
            final_results.append(merged_output)

    # concatenates dataframes together
    final_results = pd.concat(final_results, ignore_index=True)

    # creates scatterplot with data distribution and computes Pearson correlation
    plot_pearson(final_results["peptide activity"], final_results["score"], "mutant_data_distribution")

    # plots AUC between binder and predicted score that is directly comparable with results from BATMAN paper
    plot_AUC(final_results["binder"], final_results["score"], "AUC", "AUC_mutant")


# creates AUC figure
def plot_AUC(x, y, AUC_type, filename):
    # creates and populates figure with data
    plt.figure(figsize=(15, 10))
    fpr, tpr, _ = roc_curve(x, y)
    plt.plot(fpr, tpr, label=AUC_type)
    plt.plot([0, 1], [0, 1], "r--", label="random chance")

    # computes AUC or AUC_0.1 based on the AUC_type
    if AUC_type == "AUC":
        auce = roc_auc_score(x, y)

        plt.xlim([0, 1])
        plt.ylim([0, 1])

    elif AUC_type == "AUC_0.1":
        auce = roc_auc_score(x, y, max_fpr=0.1)

        plt.xlim([0, 0.1])
        plt.ylim([0, 0.2])

        plt.yticks(np.arange(0.0, 0.21, 0.05))

    else:
        print("Incorrect type of AUC!")
        auce = np.nan
    
    # sets titles for axis and increases font sizes
    plt.xlabel("False Positive Rate", fontsize=40)
    plt.ylabel("True Positive Rate", fontsize=40)
    plt.title(f"{AUC_type} = {auce:.3f}", fontsize=50)
    plt.tick_params(axis="both", labelsize=30)
    plt.legend(fontsize=30)
    plt.margins(x=0, y=0)

    # saves the figure
    plt.savefig(f'data/{filename}.png')

# creates scatterplot with data distribution and computes Pearson correlation on it
def plot_pearson(x, y, filename):
    # creates and populates scatterplot with data
    plt.figure(figsize=(15, 10))
    plt.scatter(x, y, alpha=0.3, s=10)
    z = np.polyfit(x, y, 1)
    plt.plot([0,1], np.poly1d(z)([0,1]), "r-", label="Pearson")
    plt.legend()

    # computes overall Pearson correlation
    r, p = stats.pearsonr(x, y)
    
    # sets titles for axis and increases font sizes
    plt.xlabel("peptide activity", fontsize=40)
    plt.ylabel("score", fontsize=40)
    plt.title(f"Pearson Correlation r={r:.3f}, p={p:.7f}", fontsize=40)
    plt.tick_params(axis="both", labelsize=30)
    plt.legend(fontsize=30)
    plt.margins(x=0)

    # saves and displays the figure
    plt.savefig(f'data/{filename}.png')


if __name__ == "__main__":
    main()
