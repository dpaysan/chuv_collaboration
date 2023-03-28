# -*- coding: utf-8 -*-
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm


def find_markers(data, labels, test="welch"):
    results = []
    i = 0
    for label in tqdm(np.unique(labels), desc="Run marker screen"):
        label_results = {
            "label": [],
            "marker": [],
            "fc": [],
            "abs_delta_fc": [],
            "pval": [],
        }
        for c in data.columns:
            i += 1
            x = np.array(data.loc[labels == label, c])
            y = np.array(data.loc[labels != label, c])
            x = np.array(x[x != np.nan]).astype(float)
            y = np.array(y[y != np.nan]).astype(float)

            if test == "welch":
                pval = stats.ttest_ind(x, y, equal_var=False)[1]
            elif test == "ttest":
                pval = stats.ttest_ind(x, y, equal_var=False)[1]
            elif test == "wilcoxon":
                pval = ranksums(x, y)[1]
            else:
                raise NotImplementedError("Unknown test type: {}".format(test))
            fc = (np.mean(x) + 1e-15) / (np.mean(y) + 1e-15)
            label_results["label"].append(label)
            label_results["marker"].append(c)
            label_results["fc"].append(fc)
            label_results["abs_delta_fc"].append(abs(fc - 1))
            label_results["pval"].append(pval)
        label_result = pd.DataFrame(label_results)
        label_result.pval = label_result.pval.astype(float)
        label_result = label_result.sort_values("pval")
        results.append(label_result)
    result = pd.concat(results)
    result["adjusted_pval"] = fdrcorrection(np.array(result.loc[:, "pval"]))[1]
    return result.sort_values("adjusted_pval")
