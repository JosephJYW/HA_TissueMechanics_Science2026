# %% [markdown]
# # Integrating Uninjured P2 P3 and Regen Non Regen 14DPA

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
from matplotlib import colors
#Define a nice colour map for gene expression
colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
colors3 = plt.cm.Greys_r(np.linspace(0.7,0.8,40))
colorsComb = np.vstack([colors3, colors2])
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colorsComb)

# %%
plt.rcParams['figure.figsize']=(8,8) #rescale figures
# sc.logging.print_versions()

# %%
# read the full counts of the Uninjured dataset 
adata_Uninj_all = sc.read_h5ad("./anndata/Uninj_P2P3_fullcounts_25032024.h5ad")

# %%
adata_Uninj_all

# %%
# read the full counts of 14DPA dataset
adata_RNR_all = sc.read_h5ad("./anndata/R_NR_14DPA_fullCounts_25032024.h5ad")

# %%
adata_RNR_all

# %%
# confirm that the symbols were new symbols
adata_Uninj_all[:,['Ecrg4']].to_df()
# adata_Uninj_all[:,['1500015O10Rik']].to_df()

# %%
# concaterate the datsets
adata_all = ad.concat([adata_Uninj_all, adata_RNR_all], 
                      join='outer', 
                      label='Condition',
                      keys=['Uninjured', 'RNR'],
                      index_unique='-')
# change all NaN to 0 (since pooled data are not sparse matrix)
adata_all.X = np.nan_to_num(adata_all.X) 
adata_all.to_df()

# %%
del adata_Uninj_all, adata_RNR_all

# %%
adata_all.to_df(layer = "counts")

# %%
adata_all.obs['annotation_pre2'].value_counts()

# %%
# clean up the annotations 
# update the cell type names 
tmp = adata_all.obs['annotation_pre2']
tmp = ['Immune' if item.startswith('Lymphocyte') else item for item in tmp]
tmp = ['Immune' if item.startswith('Monocyte/Macrophage') else item for item in tmp]
tmp = ['Keratinocyte' if item.startswith('Epithelial') else item for item in tmp]
tmp = ['Remove' if item.startswith('Remove') else item for item in tmp]
adata_all.obs['annotation_pre3'] = tmp

# %%
# remove the cells that are marked remove
adata_all = adata_all[adata_all.obs['annotation_pre3'] != 'Remove']

# %%
adata_all.obs['annotation_pre3'].value_counts()

# %%
# save all the cells to counts matrix
adata_all.layers['counts'] = adata_all.X.copy()

# %%
# save this full counts anndata object
adata_all.write_h5ad("./anndata/Uninj_RNR_fullcounts_25052024.h5ad")

# %%
# create a copy of adata_all 
adata_all_copy = adata_all.copy()
# normalise and transform the data
sc.pp.normalize_total(adata_all_copy, target_sum=1e4)
sc.pp.log1p(adata_all_copy)

# %%
# remove X_diffmap if it exists
if "X_diffmap" in adata_all_copy.obsm.keys():
    del adata_all_copy.obsm["X_diffmap"]

# %%
sc.pp.pca(adata_all_copy, n_comps=50, svd_solver='arpack')
sc.pp.neighbors(adata_all_copy)
sc.tl.leiden(adata_all_copy, key_added='groups', resolution=0.8)

# %%
# Check if the minimum number of cells per cluster is < 21:in that case, sizes will be also passed as input to the normalization
adata_all_copy.obs['groups'].value_counts()

# %%
from scipy.sparse import csr_matrix

#Preprocess variables for scran normalization
input_groups = adata_all_copy.obs['groups']
data_mat = csr_matrix(adata_all.X.T, dtype=np.float32)

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
adata_all.obs['size_factors'] = size_factors

sc.pl.scatter(adata_all, 'size_factors', 'n_counts')
sc.pl.scatter(adata_all, 'size_factors', 'n_genes')

sb.displot(size_factors, bins=50, kde=False)
plt.show()

# %%
# Normalize adata with the size factors calcuated
adata_all.X /= adata_all.obs['size_factors'].values[:,None]
sc.pp.log1p(adata_all)
# store the state at raw
adata_all.raw = adata_all
# delete adata_all_copy
del adata_all_copy

# %%
sc.pp.pca(adata_all, n_comps=50, svd_solver='arpack')
# plot the elbow plot
sc.pl.pca_variance_ratio(adata_all, log=True, n_pcs=50)

# %%
# compute neighbours and leiden clustering and umap
nPC = 40

if "X_diffmap" in adata_all.obsm.keys():
    del adata_all.obsm["X_diffmap"]
sc.pp.neighbors(adata_all, n_neighbors=10, n_pcs=nPC)
sc.tl.umap(adata_all)
sc.tl.leiden(adata_all, key_added='groups', resolution=0.8)

# %%
# save the current pca and umpa coordinates
adata_all.obsm['X_pca_org'] = adata_all.obsm['X_pca']
adata_all.obsm['X_umap_org'] = adata_all.obsm['X_umap']

# %%
# plot the umap
# make sure it is the original umap without corrections
adata_all.obsm['X_umap'] = adata_all.obsm['X_umap_org']
sc.pl.umap(adata_all, color=['groups', 'batch', 'annotation_pre3'], ncols=1)

# %%
# trial
sc.pl.umap(adata_all, color=['Ptprc', 'Cd74', 'Cd44'],  color_map=mymap, size= 40)

# %%
# update the adata_all by saving 
adata_all.write_h5ad("./anndata/Uninj_RNR_fullcounts_25052024.h5ad")

# %%
# load the adata_all
adata_all = sc.read_h5ad("./anndata/Uninj_RNR_fullcounts_25052024.h5ad")

# %% [markdown]
# # Prepare and perform scvi integration

# %%
# compute highly variable genes 
sc.pp.highly_variable_genes(adata_all, n_top_genes=4000, 
                            batch_key='batch', subset=False, 
                            layer='counts', flavor='seurat_v3', # counts layer is expected when using seurat_v3
                            inplace=True)

# %%
# create a subset of the adata_all
adata_all_subset = adata_all[:, adata_all.var.highly_variable]
adata_all_subset

# %%
# save the subset to h5ad
adata_all_subset.write_h5ad("./anndata/Uninj_RNR_HvgSubset_25052024.h5ad")

# %%
# load the subset
adata_all_subset = sc.read_h5ad("./anndata/Uninj_RNR_HvgSubset_25052024.h5ad")

# %%
# delete the adata_all if necessary to save space 
del adata_all

# %%
scvi.model.SCVI.setup_anndata(
    adata_all_subset,
    layer = "counts",
    batch_key="batch"
    # categorical_covariate_keys=["batch"]
    # continuous_covariate_keys=["size_factors"],
    # batch_correction=True,
    # latent_distribution="normal"
)

# %%
model = scvi.model.SCVI(adata_all_subset, n_latent=30, n_layers=2, gene_likelihood="nb")

# %%
model.train(max_epochs=800, 
            early_stopping=True, 
            check_val_every_n_epoch=5, 
            early_stopping_patience=20, 
            early_stopping_monitor='elbo_validation')

# %%
# save the model 
model.save("./scvi_model_Uninj_RNR_30052024.model", overwrite=True)

# %%
# load the model
model = scvi.model.SCVI.load("./scvi_model_Uninj_RNR_30052024.model", adata=adata_all_subset)

# %% [markdown]
# ## read the embeddings

# %%
SCVI_LATENT_KEY = "X_scVI"
adata_all_subset.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

# %%
if "X_diffmap" in adata_all_subset.obsm.keys():
    del adata_all_subset.obsm["X_diffmap"]
sc.pp.neighbors(adata_all_subset, use_rep=SCVI_LATENT_KEY,
                n_neighbors=30)
sc.tl.leiden(adata_all_subset, key_added="scvi_leiden05", resolution=0.5)
sc.tl.leiden(adata_all_subset, key_added="scvi_leiden08", resolution=0.8)
sc.tl.leiden(adata_all_subset, key_added="scvi_leiden10", resolution=1.0)

# %%
SCVI_MDE_KEY = "X_scVI_MDE"
adata_all_subset.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(adata_all_subset.obsm[SCVI_LATENT_KEY])

# %%
SCVI_NORMALIZED_KEY = "X_scVI_normalizedCounts"
adata_all_subset.layers[SCVI_NORMALIZED_KEY] = model.get_normalized_expression(adata_all_subset, library_size=10e4)


# %%
# define the current X_pca and X_umap
adata_all_subset.obsm['X_pca'] = adata_all_subset.obsm['X_scVI']
adata_all_subset.obsm['X_umap'] = adata_all_subset.obsm['X_scVI_MDE']

# %%
# save the adata_all_subset
adata_all_subset.write_h5ad("./anndata/Uninj_RNR_HvgSubset_30052024.h5ad")

# %% [markdown]
# ## Annotate the clusters 

# %%
# load the adata_all_subset
adata_all_subset = sc.read_h5ad("./anndata/Uninj_RNR_HvgSubset_30052024.h5ad")
adata_all_subset

# %%
# use wilcoxon rank sum test to find the differentially expressed genes
sc.tl.rank_genes_groups(adata_all_subset, 
                        groupby='scvi_leiden05', 
                        method='wilcoxon',
                        use_raw=True,
                        corr_method='benjamini-hochberg')
sc.pl.rank_genes_groups(adata_all_subset, n_genes=25, sharey=False)

# %%
sc.pl.umap(adata_all_subset, color=['scvi_leiden05'], legend_loc = "on data", use_raw=True, size= 10)

# %%
# check
key = 'Col2a1'
sc.pl.umap(adata_all_subset, color=[key], use_raw=True, color_map=mymap, size= 10)
sc.pl.violin(adata_all_subset, key, groupby='scvi_leiden05', use_raw=True, rotation=90, stripplot=True, log=False)

# %%
# create new annotation based on clusters 
# Categories to rename
adata_all_subset.obs['annotation_int'] = adata_all_subset.obs['scvi_leiden05']
adata_all_subset.rename_categories('annotation_int', ['Fibroblast.1','Fibroblast.2','Immune.1','Fibroblast.3','Fibroblast.4','Endothelial.1','Keratinocyte.1','Pericyte/SMC.1','Keratinocyte.2','Fibroblast.5','Chondrocyte','Fibroblast.6','Pericyte/SMC.2','Schwann','Fibroblast.7','Immune.2','Lymphatic Endothelial','Immune.3','Immune.4','Fibroblast.8','Fibroblast.9','Fibroblast.10','Immune.5','Immune.6']) # remove 23?

# %%
tmp = adata_all_subset.obs['annotation_int']
tmp = ['Fibroblast' if item.startswith('Fibroblast') else item for item in tmp]
tmp = ['Endothelial' if item.startswith('Endothelial') else item for item in tmp]
tmp = ['Schwann' if item.startswith('Schwann') else item for item in tmp]
tmp = ['Pericyte/SMC' if item.startswith('Pericyte/SMC') else item for item in tmp]
tmp = ['Keratinocyte' if item.startswith('Keratinocyte') else item for item in tmp]
tmp = ['Immune' if item.startswith('Immune') else item for item in tmp]

adata_all_subset.obs['annotation_int2'] = tmp

# %%
# remove the Erythrocyte from the clusters 
adata_all_subset = adata_all_subset[adata_all_subset.obs['annotation_int2']!='Erythrocyte' , :]

# %%
adata_all_subset.obs['annotation_int2'] = adata_all_subset.obs['annotation_int2'].astype('category')
adata_all_subset.obs['annotation_int2'] = adata_all_subset.obs['annotation_int2'].cat.reorder_categories(['Fibroblast','Chondrocyte','Schwann','Immune','Pericyte/SMC','Keratinocyte','Endothelial','Lymphatic Endothelial'])

# %%
sc.pl.umap(adata_all_subset, color=['annotation_int2', 'annotation_pre3'], legend_loc='on data', legend_fontsize=10)

# %%
# Annotation
sc.pl.umap(adata_all_subset, color=["Amp_location","batch","annotation_int2", "Condition"], use_raw=True, color_map=mymap, size= 10)

# %%
# save the adata_all_subset
adata_all_subset.write_h5ad("./anndata/Uninj_RNR_HvgSubset_30052024.h5ad")


