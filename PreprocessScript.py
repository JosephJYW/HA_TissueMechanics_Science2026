# %% [markdown]
# # Preprocess Pipeline from 10x
# Sample script for preprocessing 10x h5 datasets

# %%
import os
import scanpy as sc
import pandas as pd
import anndata as ad 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import seaborn as sb
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")

# %%
import os
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.3'

import rpy2.rinterface_lib.callbacks
import logging
from rpy2.robjects import pandas2ri
import anndata2ri

# %%
# Ignore R warning messages
#Note: this can be commented out to get more verbose R output
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

# Automatically convert rpy2 outputs to pandas dataframes
pandas2ri.activate()
anndata2ri.activate()
%load_ext rpy2.ipython

# %%
results_file = './anndata/HA_4MU_Ctrl_CellBend_2.h5ad'

# %%
plt.rcParams['figure.figsize']=(8,8) #rescale figures

# %%
# read the 10x output
path = "./SeqFeb2024/HA_4MU_Ctrl_2/cellbender_output/"
adata = sc.read_10x_h5(path + "cellbender_output_filtered.h5")
adata.uns["name"] = "4MU_Ctrl_2"

# %%
adata.obs.index.names = ['barcode']
adata.obs.head()

# %%
all(adata.var.index.notnull())

# %%
adata.var_names_make_unique()

# %%
adata.shape

# %%
adata.to_df().head()

# %% [markdown]
# ## 1. Preprocessing and visualisation
# Quality control 

# %%
# Quality control - calculate QC covariates
adata.obs['n_counts'] = adata.X.sum(1)
adata.obs['log_counts'] = np.log(adata.obs['n_counts'])
adata.obs['n_genes'] = (adata.X > 0).sum(1)

# %%
mt_gene_mask = [gene.startswith('mt-') for gene in adata.var_names]
adata.obs['mt_frac'] = adata.X[:, mt_gene_mask].sum(1)/adata.obs['n_counts'].values.reshape(adata.shape[0], 1)

# %%
print(adata.uns["name"])
sc.pl.violin(
    adata,
    ["n_genes", "n_counts", "mt_frac"],
    jitter=0.4,
    multi_panel=True,
)

# %%
sc.pl.scatter(adata, x='n_counts', y='mt_frac')
sc.pl.scatter(adata, x='n_counts', y='n_genes', color='mt_frac')

# %%
#Thresholding decision: counts
p3 = sb.displot(adata.obs['n_counts'], kde=False)
plt.show()

p4 = sb.displot(adata.obs['n_counts'][adata.obs['n_counts']<10000], kde=False, bins=120)
plt.show()

p5 = sb.displot(adata.obs['n_counts'][adata.obs['n_counts']>20000], kde=False, bins=60)
plt.show()

# %%
# Filter cells according to identified QC thresholds:
print('Total number of cells: {:d}'.format(adata.n_obs))

sc.pp.filter_cells(adata, min_counts = 4000)
print('Number of cells after count filter: {:d}'.format(adata.n_obs))

sc.pp.filter_cells(adata, max_counts = 20000)
print('Number of cells after max count filter: {:d}'.format(adata.n_obs))

adata = adata[adata.obs['mt_frac'] < 0.1]
print('Number of cells after MT filter: {:d}'.format(adata.n_obs))

# %%
#Thresholding decision: genes
p6 = sb.displot(adata.obs['n_genes'], kde=False, bins=60)
plt.show()

p7 = sb.displot(adata.obs['n_genes'][adata.obs['n_genes']< 1000], kde=False, bins=100)
plt.show()

p8 = sb.displot(adata.obs['n_genes'][adata.obs['n_genes']> 3000], kde=False, bins=60)
plt.show()

# %%
# Thresholding on number of genes
print('Total number of cells: {:d}'.format(adata.n_obs))

sc.pp.filter_cells(adata, min_genes = 600)
print('Number of cells after gene filter: {:d}'.format(adata.n_obs))

sc.pp.filter_cells(adata, max_genes = 6000)
print('Number of cells after gene filter: {:d}'.format(adata.n_obs))

# %%
#Filter genes:
print('Total number of genes: {:d}'.format(adata.n_vars))

# Min 20 cells - filters out 0 count genes
sc.pp.filter_genes(adata, min_cells=3)
print('Number of genes after cell filter: {:d}'.format(adata.n_vars))

# %%
# Remove doublets with scrublet 
# input a raw (unnormalized) UMI counts matrix with genes as columns and rows as cells
import scrublet as scr 
scrub = scr.Scrublet(adata.X, expected_doublet_rate=0.06)

# run the doublet detection
doublet_scores, predicted_doublets = scrub.scrub_doublets(min_counts=2,
                                                        min_cells=3,
                                                        min_gene_variability_pctl=85,
                                                        n_prin_comps=50)
print(scrub.plot_histogram())

# %%
# define the theshold for doublet detection (manually based on the first graph)
threshold_doublet = 0.20
scrub.call_doublets(threshold=threshold_doublet)
print(scrub.plot_histogram())
# redefine predicted_doublets based on the threshold
predicted_doublets = doublet_scores > threshold_doublet


adata.obs['scrublet_scores'] = doublet_scores
adata.obs['predicted_doublets'] = predicted_doublets
# remove the doublets
print("Removing doublets from", adata.uns["name"], "from shape:", adata.shape)
adata = adata[adata.obs['predicted_doublets'] == False, :]
print(adata.uns["name"], ": data to shape: ", adata.shape)

# %% [markdown]
# ## Normalization

# %%
# save the raw counts data
adata.layers["counts"] = adata.X.copy()

# %%
#Perform a clustering for scran normalization in clusters
adata_pp = adata.copy()
sc.pp.normalize_total(adata_pp)
sc.pp.log1p(adata_pp)

# %%
sc.pp.pca(adata_pp, n_comps=50, svd_solver='arpack')
sc.pp.neighbors(adata_pp)
sc.tl.leiden(adata_pp, key_added='groups', resolution=1)

# %%
# Check if the minimum number of cells per cluster is < 10:in that case, sizes will be also passed as input to the normalization
adata_pp.obs['groups'].value_counts()

# %%
from scipy.sparse import csr_matrix
#Preprocess variables for scran normalization
input_groups = adata_pp.obs['groups']
data_mat = csr_matrix(adata.X.T, dtype=np.float32)

# %%
%%R -i data_mat -i input_groups -o size_factors
library(scran)
library(Matrix)
library(MatrixExtra)
# Convert to sparse matrix from csr to csc format
data_mat = as.csc.matrix(data_mat)

# Compute size factors
size_factors = sizeFactors(computeSumFactors(
  SingleCellExperiment::SingleCellExperiment(list(counts=data_mat)), 
  clusters=input_groups, min.mean=0.1
  ))


# %%
adata.obs['size_factors'] = size_factors

sc.pl.scatter(adata, 'size_factors', 'n_counts')
sc.pl.scatter(adata, 'size_factors', 'n_genes')

sb.displot(size_factors, bins=50, kde=False)
plt.show()

# %%
# Normalize adata with the size factors calcuated
adata.X /= adata.obs['size_factors'].values[:,None]
sc.pp.log1p(adata)

# %%
# store the state at raw
adata.raw = adata

# %%
del adata_pp
# make a new copy of the adata for cell cycle scoring
# without scaling is superior compared to with scaling
adata_pp = adata.copy() 

# %%
# Compute the cell cycle effect
s_genes = pd.read_csv("./cc_genes_s.csv", header=0, index_col=0)
s_genes = s_genes.values.tolist()
s_genes = [item for sublist in s_genes for item in sublist]
s_genes2 = []
for x in s_genes:
    if x in adata_pp.var_names:
        s_genes2.append(x)
s_genes = s_genes2
# read the s phase and g2m phase genes from csv files 
g2m_genes = pd.read_csv("./cc_genes_g2m.csv", header=0, index_col=0)
g2m_genes = g2m_genes.values.tolist()
g2m_genes = [item for sublist in g2m_genes for item in sublist]
g2m_genes2 = []
for x in g2m_genes:
    if x in adata_pp.var_names:
        g2m_genes2.append(x)
g2m_genes = g2m_genes2
# combine the s phase and g2m phase genes that appears in the dataset
cc_genes = s_genes + g2m_genes
# perform the cell cycle scoring
sc.tl.score_genes_cell_cycle(adata_pp, s_genes=s_genes, g2m_genes=g2m_genes)
adata_pp.obs["phase"] = adata_pp.obs["phase"].astype("category")

# %%
# plot the cc genes
adata_cc = adata_pp[:, cc_genes]
sc.tl.pca(adata_cc)
sc.pl.pca_scatter(adata_cc, color="phase")

# %%
# save the s_score, g2m_score and phase_pred
s_score = adata_pp.obs["S_score"].values
g2m_score = adata_pp.obs["G2M_score"].values
phase_pred = adata_pp.obs["phase"].values

# pass the information to the adata_pp object
adata.obs["S_score"] = s_score
adata.obs["G2M_score"] = g2m_score
adata.obs["phase"] = phase_pred

# %%
# remove adata extras 
del adata_pp
del adata_cc

# %% [markdown]
# ## HVG

# %%
# compute the highly variable genes after normalization/log1p but without scaling the data
sc.pp.highly_variable_genes(adata,  flavor='cell_ranger', n_top_genes=4000)
print('\n','Number of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))

# %%
# plot the highly variable genes
sc.pl.highly_variable_genes(adata, log=False)

# %% [markdown]
# # 2. Visualisation

# %%
# Calculate the visualizations
sc.pp.pca(adata, n_comps=50, use_highly_variable=True, svd_solver='arpack')
# create the elbow plot
sc.pl.pca_variance_ratio(adata, log=True, n_pcs = 50)

# %%
# determine the number of PCs to use
nPC = 30
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=nPC)

sc.tl.umap(adata, min_dist=0.3)

# %%
sc.pl.pca_scatter(adata, color='n_counts')
sc.pl.pca_scatter(adata, color='phase')
sc.pl.umap(adata, color='n_counts')
sc.pl.umap(adata, color='phase')

# %%
adata.write(results_file)

# %% [markdown]
# # 3. Clustering

# %%
# Perform leiden clustering - using highly variable genes
sc.tl.leiden(adata, key_added='leiden_r1', resolution=1)

# %%
adata.obs['leiden_r1'].value_counts()

# %%
# remove clusters high in hbb and with less that 50 cells per cluster 
adata = adata[~adata.obs['leiden_r1'].isin(['23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45'])]

# %%
#Visualize the clustering and how this is reflected by different technical covariates
sc.pl.umap(adata, color=['leiden_r1'], legend_loc='on data', legend_fontsize=12)
sc.pl.umap(adata, color=['log_counts', 'mt_frac'])
# umap for cell cycle
sc.pl.umap(adata, color=['phase'], legend_loc='on data', legend_fontsize=12)

# %%
# save the results
adata.write(results_file)

# %% [markdown]
# # 4. Markers and cluster annotation

# %%
del adata.raw
adata.raw = adata

# %%
# reinstate the raw data
adata.X = adata.raw.X

#Calculate marker genes r1
sc.tl.rank_genes_groups(adata, groupby='leiden_r1', method= 'wilcoxon', key_added='rank_genes_r1')

# %%
# show the top 20 genes per cluster
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, key='rank_genes_r1')

# %%
#Define a nice colour map for gene expression
colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
colors3 = plt.cm.Greys_r(np.linspace(0.7,0.8,20))
colorsComb = np.vstack([colors3, colors2])
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colorsComb)

# %% [markdown]
# ### Pericytes / Smooth muscle cells

# %%
# Pericytes / Smooth muscle cells
sc.pl.umap(adata, color=['Rgs5'], use_raw=False, color_map=mymap, size = 40)
sc.pl.violin(adata, 'Rgs5', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %% [markdown]
# ### Mesenchymal cells

# %%
# Mesenchymal cells
sc.pl.umap(adata, color=['Pdgfra','Prrx1'], use_raw=False, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Pdgfra', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %% [markdown]
# ### Lymphatic Endothelial cells

# %%
# Lympathic cells
sc.pl.umap(adata, color=['Prox1'], use_raw=False, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Prox1', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %% [markdown]
# ### Keratinocyte

# %%
# keratinocytes
sc.pl.umap(adata, color=['Krt5', 'Krt14'], use_raw=False, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Krt5', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %% [markdown]
# ### Endothelial cells

# %%
# Endothelial cells
sc.pl.umap(adata, color=['Cdh5','Pecam1'], use_raw=False, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Cdh5', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %% [markdown]
# ### Schwann cells

# %%
# Schwann cells
sc.pl.umap(adata, color=['Scn7a','Plp1'], use_raw=False, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Plp1', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %% [markdown]
# ### Erythrocytes

# %%
# Erythrocytes
sc.pl.umap(adata, color=['Hba-a1'], use_raw=False, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Hba-a1', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %%
# remove clusters high in hbb
# adata = adata[~adata.obs['leiden_r1'].isin(['18'])]

# %% [markdown]
# ### Monocyte/Macrophage cells

# %%
# Immune cells
sc.pl.umap(adata, color=['Lyz2'], use_raw=False, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Lyz2', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %% [markdown]
# ### Bone cells

# %%
# Bone cells
sc.pl.umap(adata, color=['Bglap'], use_raw=False, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Bglap', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %% [markdown]
# ### Basal epidermal cells

# %%
# Basal epidermal cells
sc.pl.umap(adata, color=['Krt90', 'Col17a1'], use_raw=False, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Krt90', groupby='leiden_r1', use_raw=False, rotation=90, stripplot=True, log=False)

# %% [markdown]
# ### Lymphocyte

# %%
# Cd45
sc.pl.umap(adata, color=['Ptprc'], use_raw=True, color_map=mymap, size= 40)
sc.pl.violin(adata, 'Ptprc', groupby='leiden_r1', use_raw=True, rotation=90, stripplot=True, log=False)

# %%
# # calculate the degs for each cluster
# sc.tl.rank_genes_groups(adata, groupby='louvain_r1', method='wilcoxon', key_added='rank_genes_r1')
# show the top 10 marker genes for cluster 1
sc.pl.rank_genes_groups(adata, key='rank_genes_r1', groups=['10'], show=False)

# %%
# write the results to a h5
adata.write(results_file)

# %% [markdown]
# # Annotate the cells 

# %%
# Categories to rename
adata.obs['annotation_pre'] = adata.obs['leiden_r1']
adata.rename_categories('annotation_pre', ['Fibroblast.1','Fibroblast.2','Fibroblast.3','Fibroblast.4','Fibroblast.5','Fibroblast.6','Immune.1','Fibroblast.7','Immune.2','Fibroblast.8','Endothelial.1','Pericyte/SMC','Fibroblast.9','Immune.3','Fibroblast.10','Immune.4','Immune.5','Keratinocyte','Endothelial.3','Fibroblast.11','Pericyte/SMC.2'])

# %%
tmp = adata.obs['annotation_pre']
tmp = ['Fibroblast' if item.startswith('Fibroblast') else item for item in tmp]
tmp = ['Endothelial' if item.startswith('Endothelial') else item for item in tmp]
tmp = ['Schwann' if item.startswith('Schwann') else item for item in tmp]
tmp = ['Monocyte/Macrophage' if item.startswith('Monocyte/Macrophage') else item for item in tmp]
tmp = ['Pericyte/SMC' if item.startswith('Pericyte/SMC') else item for item in tmp]
tmp = ['Keratinocyte' if item.startswith('Keratinocyte') else item for item in tmp]
tmp = ['Epithelial' if item.startswith('Basal epidermal') else item for item in tmp]
tmp = ['T cell' if item.startswith('T_cell') else item for item in tmp]
tmp = ['Remove' if item.startswith('Remove') else item for item in tmp]
tmp = ['Erythrocyte' if item.startswith('Erythrocyte') else item for item in tmp]
tmp = ['Chondrocyte' if item.startswith('Chondrocyte') else item for item in tmp]
tmp = ['Lymphocyte' if item.startswith('Lymphocyte') else item for item in tmp]
tmp = ['Lymphatic Endothelial' if item.startswith('Lymphatic Endothelial') else item for item in tmp]
tmp = ['Immune' if item.startswith('Immune') else item for item in tmp]
tmp = ['Osteoclast' if item.startswith('Osteoclast') else item for item in tmp]
adata.obs['annotation_pre2'] = tmp

# %%
sc.pl.umap(adata, color=['annotation_pre2'], legend_loc='on data', legend_fontsize=10)

# %%
adata.write(results_file)

# %%
del adata.raw
adata.raw = adata


