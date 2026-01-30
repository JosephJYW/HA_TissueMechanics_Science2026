# %% [markdown]
# # Regen - NonRegen Fibroblast scVI integration
# 

# %%
import os
import scanpy as sc
import pandas as pd
import torch
import scvi 
import anndata as ad 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
scvi.settings.seed = 0
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

# %%
plt.rcParams['figure.figsize']=(8,8) #rescale figures
# sc.logging.print_versions()

# %%
import os
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.2'
import logging
import rpy2.rinterface_lib.callbacks

# %%
import logging
from rpy2.robjects import pandas2ri
import anndata2ri

# %%
from matplotlib import colors
#Define a nice colour map for gene expression
colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
colors3 = plt.cm.Greys_r(np.linspace(0.7,0.8,40))
colorsComb = np.vstack([colors3, colors2])
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colorsComb)

# %% [markdown]
# ## Load the preprocessed/integrated adata_fibroblast and model created from scVI

# %%
# load the adata_fibroblast
adata_fibroblast = sc.read_h5ad("./anndata/R_NR_14DPA_fibroblast.h5ad")

# %%
# load the scvi model 
model_fib = scvi.model.SCVI.load("./Regen-NonRegen_analysis_results/scvi_model_fib_25032024.model", adata=adata_fibroblast)

# %%
sc.tl.leiden(adata_fibroblast, key_added="scvi_fib_leiden07", resolution=0.7)

# %%
sc.pl.embedding(
    adata_fibroblast,
    basis="umap",
    color=["batch", "scvi_leiden10","annotation_pre3", "scvi_fib_leiden07", "Amp_location"],
    frameon=False,
    ncols=1,
)

# %% [markdown]
# ## Quick glimpse of the cellular composition for each condition

# %%
fib_df = pd.DataFrame(adata_fibroblast.obs.groupby(['batch','scvi_fib_leiden07']).size(), columns = ['count'])
# add an extra column of percentage
fib_df['percentage'] = fib_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
fib_df
# show the RJS 
fib_df.xs('NRB1')

# %%
# plot a bar chart of the percentage of cells in each cluster
ax = fib_df['percentage'].unstack().plot(kind='barh', stacked=True, figsize=(10,5),
                      title='Percentage of cells in each cluster of Pdfgra+ cells', 
                      ylabel='Batch', xlabel='Percentage', legend=False,
                      )
labels = ['Fib.0','Fib.1','Fib.2','Fib.3','Fib.4','Fib.5','Fib.6','Fib.7','Fib.8','Fib.9','Fib.10','Fib.11','Fib.12','Fib.13']
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('figures/fib_percentage_batch.pdf', bbox_inches='tight')

# %%
# calculate the average percentage for each level
fib_df.loc[pd.IndexSlice[['RB1','RB2', 'RJS'],:], 'percentage'].groupby("scvi_fib_leiden07").mean()

# %%
Rclus_avg = fib_df.loc[pd.IndexSlice[['RB1','RB2', 'RJS'],:], 'percentage'].groupby("scvi_fib_leiden07").mean()
NRclus_avg = fib_df.loc[pd.IndexSlice[['NRB1','NRB2', 'NRB3'],:], 'percentage'].groupby("scvi_fib_leiden07").mean()

# %%
# create a dataframe of the average percentages 
d = {'Regen': Rclus_avg, 'NonRegen': NRclus_avg}
clus_avg = pd.DataFrame(data=d)
clus_avg

# %%
# plot a horizontal bar plot for clus_avg
ax = clus_avg.T.plot(kind='barh', stacked=True, figsize=(10,5),
                     title='Percentage of cells in each cluster of PDGFRa+ cells', 
                      ylabel='Batch', xlabel='Percentage', legend=False,
                      )
labels = adata_fibroblast.obs['scvi_fib_leiden07'].cat.categories
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# %%
# calculate the probability of the cells in each cluster from each subcluster. 
# calculate the sum across the rows 
clus_sum = clus_avg.sum(axis = 1)

# %%
clus_avg['Reg_Perct'] = (clus_avg['Regen'] / clus_sum ) * 100
clus_avg['NonReg_Perct'] = (clus_avg['NonRegen'] / clus_sum) * 100
clus_avg

# %%
sc.pl.correlation_matrix(adata_fibroblast, 'batch', figsize=(10,10))

# %% [markdown]
# ## Proportional analysis with Propellor 

# %%
# prepare information to feed into propeller 
clust = adata_fibroblast.obs['scvi_fib_leiden07'].values
biorep = adata_fibroblast.obs['batch'].values
grp = adata_fibroblast.obs['Amp_location'].values

# %%
%%R -i grp -i clust -i biorep -o propeller_res
library(speckle)
library(ggplot2)

propeller_res = propeller(clusters = clust, sample = biorep, group = grp,
  robust = FALSE, trend = FALSE, transform="asin")

# %%
# create a new row inidicating whether FDR < 0.05
propeller_res["FDR_TF"] = ["True" if x < 0.05 else "False" for x in propeller_res["FDR"]]

# %%
# take the last 3 column
propeller_res[["P.Value", "FDR", "FDR_TF"]]

# %%
# plot the percentage bar plot again
ax = clus_avg.T.plot(kind='barh', stacked=True, figsize=(10,5),
                     title='Percentage of cells in each cluster of Pdgfra+ cells', 
                      ylabel='Amputation condition', xlabel='Percentage', legend=False,
                      )
labels = adata_fibroblast.obs['scvi_fib_leiden07'].cat.categories
labels = [f'{cat} #' if propeller_res["FDR_TF"][cat]=="True" else cat for cat in labels]
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('figures/fib_percentage_AmpLoc.pdf', bbox_inches='tight')


