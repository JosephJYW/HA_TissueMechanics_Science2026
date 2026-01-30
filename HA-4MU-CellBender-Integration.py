# %% [markdown]
# # Integrating HA 4MU Treated and untreated

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

# %% [markdown]
# # read the datasets 
# datasets has gone through filtering with SoupX manual filtering 

# %%
# read the full counts of the Uninjured dataset 
adata_Ctrl1 = sc.read_h5ad("./anndata/HA_4MU_Ctrl_CellBend_1.h5ad")
adata_Ctrl1

# %%
adata_Ctrl2 = sc.read_h5ad("./anndata/HA_4MU_Ctrl_CellBend_2.h5ad")
adata_Ctrl2

# %%
# read the full counts of 14DPA dataset
adata_Treat_1 = sc.read_h5ad("./anndata/HA_4MU_Treat_CellBend_1-2.h5ad")
adata_Treat_1

# %%
# read the full counts of 14DPA dataset
adata_Treat_2 = sc.read_h5ad("./anndata/HA_4MU_Treat_CellBend_2.h5ad")
adata_Treat_2

# %%
# confirm that the symbols were new symbols
adata_Ctrl1[:,['Ecrg4']].to_df()
# adata_Uninj_all[:,['1500015O10Rik']].to_df()

# %%
# concaterate the datsets
adata_all = ad.concat([adata_Ctrl1, adata_Ctrl2, adata_Treat_1, adata_Treat_2], 
                      join='outer', 
                      label='batch',
                      keys=['Control_1', 'Control_2','4MU_1', '4MU_2'],
                      index_unique='-')
# change all NaN to 0 (since pooled data are not sparse matrix)
adata_all.X = np.nan_to_num(adata_all.X) 
adata_all.to_df()

# %%
del adata_Ctrl1, adata_Ctrl2, adata_Treat_1, adata_Treat_2

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
# tmp = ['Fibroblast' if item.startswith('Bone') else item for item in tmp]
# tmp = ['Fibroblast' if item.startswith('Chondrocyte') else item for item in tmp]
tmp = ['Remove' if item.startswith('Remove') else item for item in tmp]
# tmp = ['Fibroblast' if item.startswith('Fibroblast') else item for item in tmp]
adata_all.obs['annotation_pre3'] = tmp

# %%
# remove the cells that are marked remove
adata_all = adata_all[adata_all.obs['annotation_pre3'] != 'Remove']

# %%
adata_all.obs['annotation_pre3'].value_counts()

# %%
# save all the cells to counts matrix
adata_all.layers['counts'] = adata_all.X

# %%
# save this full counts anndata object
adata_all.write_h5ad("./anndata/HA_4MU_fullcounts_30052024.h5ad")

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
# update the adata_all by saving 
adata_all.write_h5ad("./anndata/HA_4MU_fullcounts_30052024.h5ad")

# %%
# load the adata_all
adata_all = sc.read_h5ad("./anndata/HA_4MU_fullcounts_30052024.h5ad")

# %% [markdown]
# # Prepare and perform scvi integration

# %%
# compute highly variable genes 
sc.pp.highly_variable_genes(adata_all, n_top_genes=2000, 
                            batch_key='batch', subset=False, 
                            layer='counts', flavor='seurat_v3', # counts layer is expected when using seurat_v3
                            inplace=True)

# %%
# create a subset of the adata_all
adata_all_subset = adata_all[:, adata_all.var.highly_variable]
adata_all_subset

# %%
# save the subset to h5ad
adata_all_subset.write_h5ad("./anndata/HA_4MU_HvgSubset_30052024.h5ad")

# %%
# load the subset
adata_all_subset = sc.read_h5ad("./anndata/HA_4MU_HvgSubset_30052024.h5ad")

# %%
adata_all_subset = adata_all_subset.copy()

# %%
# delete the adata_all if necessary to save space 
del adata_all

# %%
scvi.model.SCVI.setup_anndata(
    adata_all_subset,
    layer = "counts",
    batch_key="batch"
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
model.save("./scvi_model_HA_4MU_30052024.model", overwrite=True)

# %%
# load the model
model = scvi.model.SCVI.load("./scvi_model_HA_4MU_30052024.model", adata=adata_all_subset)

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
adata_all_subset.write_h5ad("./anndata/HA_4MU_HvgSubset_30052024.h5ad")

# %% [markdown]
# ## Annotate the clusters 

# %%
# load the adata_all_subset
adata_all_subset = sc.read_h5ad("./anndata/HA_4MU_HvgSubset_30052024.h5ad")
adata_all_subset

# %%
# use wilcoxon rank sum test to find the differentially expressed genes
sc.tl.rank_genes_groups(adata_all_subset, 
                        groupby='scvi_leiden10', 
                        method='wilcoxon',
                        use_raw=True,
                        corr_method='benjamini-hochberg')
sc.pl.rank_genes_groups(adata_all_subset, n_genes=25, sharey=False)

# %%
# create new annotation based on clusters 
# Categories to rename
adata_all_subset.obs['annotation_int'] = adata_all_subset.obs['scvi_leiden10']
adata_all_subset.rename_categories('annotation_int', ['Fibroblast.1','Fibroblast.2','Fibroblast.3','Immune.1','Osteoclast','Fibroblast.4','Fibroblast.5','Endothelial.1','Immune.2','Pericyte/SMC','Immune.3','Fibroblast.7','Keratinocyte','Immune.4','Fibroblast.8','Fibroblast.9','Chondrocyte','Pericyte/SMC.2','Immune.5','Schwann','Immune.6','Fibroblast.11'])
# cluster 17 is an interesting group with Rgs5 Notch3 and Myh11 (Doublet or really intesting?)

# %%
tmp = adata_all_subset.obs['annotation_int']
tmp = ['Fibroblast' if item.startswith('Fibroblast') else item for item in tmp]
tmp = ['Endothelial' if item.startswith('Endothelial') else item for item in tmp]
tmp = ['Schwann' if item.startswith('Schwann') else item for item in tmp]
tmp = ['Pericyte/SMC' if item.startswith('Pericyte/SMC') else item for item in tmp]
tmp = ['Keratinocyte' if item.startswith('Keratinocyte') else item for item in tmp]
tmp = ['Immune' if item.startswith('Immune') else item for item in tmp]
tmp = ['Immune' if item.startswith('HSC') else item for item in tmp] # Hematopoietic stem cell
# tmp = ['Immune' if item.startswith('Osteoclast') else item for item in tmp]
tmp = ['Immune' if item.startswith('T_cell') else item for item in tmp]
adata_all_subset.obs['annotation_int2'] = tmp

# %%
sc.pl.umap(adata_all_subset, color=['annotation_int2', 'annotation_pre3'], legend_loc='on data', legend_fontsize=10)

# %%
# remove the Erythrocyte from the clusters 
adata_all_subset = adata_all_subset[adata_all_subset.obs['annotation_int2']!='Erythrocyte' , :]

# %%
adata_all_subset.obs['annotation_int2'] = adata_all_subset.obs['annotation_int2'].astype('category')
adata_all_subset.obs['annotation_int2'] = adata_all_subset.obs['annotation_int2'].cat.reorder_categories(['Fibroblast','Chondrocyte','Schwann','Immune','Osteoclast','Pericyte/SMC','Keratinocyte','Endothelial'])

# %%
sc.pl.umap(adata_all_subset, color=['annotation_int','annotation_int2', 'annotation_pre3'], legend_loc='on data', legend_fontsize=10)

# %%
# add a new grouping based on batch -> condition 
adata_all_subset.obs['condition'] = adata_all_subset.obs['batch']
adata_all_subset.obs['condition'] = adata_all_subset.obs['condition'].replace('Control_1', 'Control')
adata_all_subset.obs['condition'] = adata_all_subset.obs['condition'].replace('Control_2', 'Control')
adata_all_subset.obs['condition'] = adata_all_subset.obs['condition'].replace('4MU_1', '4MU')
adata_all_subset.obs['condition'] = adata_all_subset.obs['condition'].replace('4MU_2', '4MU')

# %%
# Annotation
sc.pl.umap(adata_all_subset, color=["batch","annotation_int2","condition"], use_raw=True, color_map=mymap, size= 10)

# %%
# save the adata_all_subset
adata_all_subset.write_h5ad("./anndata/HA_4MU_HvgSubset_03072024.h5ad")


