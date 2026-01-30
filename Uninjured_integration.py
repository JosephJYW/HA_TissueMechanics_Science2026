# %% [markdown]
# # Uninjured scVi integration

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

# %% [markdown]
# load datasets

# %%
# read the h5ad file
adata_js = sc.read_h5ad("anndata/Uninj_JS_p3.h5ad")
adata_js.obs['batch'] = 'JS_P3_1'
adata_js.obs['Amp_location'] = 'P3'
adata_js

# %%
# read the h5ad file
adata_ms1 = sc.read_h5ad("anndata/Uninj_B1_p3.h5ad")
adata_ms1.obs['batch'] = 'MS_P3_1'
adata_ms1.obs['Amp_location'] = 'P3'
adata_ms1

# %%
# read the h5ad file
adata_ms2 = sc.read_h5ad("anndata/Uninj_B2_p3.h5ad")
adata_ms2.obs['batch'] = 'MS_P3_2'
adata_ms2.obs['Amp_location'] = 'P3'
adata_ms2

# %%
# read the h5ad file
adata_b21 = sc.read_h5ad("anndata/Uninj_B3_p2.h5ad")
adata_b21.obs['batch'] = 'B_P2_1'
adata_b21.obs['Amp_location'] = 'P2'
adata_b21

# %% [markdown]
# new datasets from Feb 2024 sequencing

# %%
# read the h5ad file 
adata_b22 = sc.read_h5ad("anndata/Uninj_P2_Feb24_SoupX_1.h5ad")
adata_b22.obs['batch'] = 'B_P2X_2'
adata_b22.obs['Amp_location'] = 'P2'
adata_b22

# %%
# read the h5ad file 
adata_b23 = sc.read_h5ad("anndata/Uninj_P2_Feb24_SoupX_2.h5ad")
adata_b23.obs['batch'] = 'B_P2X_2'
adata_b23.obs['Amp_location'] = 'P2'
adata_b23

# %%
# read the h5ad file 
adata_b31 = sc.read_h5ad("anndata/Uninj_P3_Feb24_SoupX_1.h5ad")
adata_b31.obs['batch'] = 'B_P3X_1'
adata_b31.obs['Amp_location'] = 'P3'
adata_b31

# %%
# read the h5ad file 
adata_b32 = sc.read_h5ad("anndata/Uninj_P3_Feb24_SoupX_2.h5ad")
adata_b32.obs['batch'] = 'B_P3X_2'
adata_b32.obs['Amp_location'] = 'P3'
adata_b32

# %%
del adata_js.raw
del adata_ms1.raw
del adata_ms2.raw
del adata_b21.raw
del adata_b22.raw
del adata_b23.raw
del adata_b31.raw
del adata_b32.raw

# %%
def flatten(xss):
    return [x for xs in xss for x in xs]

# %%
new_ver = set(flatten([adata_b32.var.index, adata_b31.var.index, adata_b23.var.index, adata_b22.var.index]))

# %%
old_ver = set(flatten([adata_js.var.index, adata_ms1.var.index, adata_ms2.var.index, adata_b21.var.index]))

# %%
unique_to_new = [gene for gene in new_ver if gene not in old_ver]
unique_to_old = [gene for gene in old_ver if gene not in new_ver]

# %%
df = pd.DataFrame(unique_to_new, columns=['Gene'])
df['Gene'] = df['Gene'].str.replace('\..*', '', regex=True)
# Export the DataFrame to a CSV file
df.to_csv('unique_to_new_uninj.csv', index=False)

# %%
df = pd.DataFrame(unique_to_old, columns=['Gene'])
df['Gene'] = df['Gene'].str.replace('\..*', '', regex=True)
# Export the DataFrame to a CSV file
df.to_csv('unique_to_old_uninj.csv', index=False)

# %% [markdown]
# generate a new dataset symbol from MGI website https://www.informatics.jax.org/batch

# %%
# load the generated conversion from MGI 
MGI_uniqueToOld = pd.read_excel("MGIBatchReport_20240409_Uninj_old.xlsx", header=0)
# load the generated conversion from MGI
MGI_uniqueToNew = pd.read_excel("MGIBatchReport_20240409_Uninj_new.xlsx", header=0)

# %%
# if the Symbol is NaN, remove the row 
MGI_convertOld = MGI_uniqueToOld[(MGI_uniqueToOld['Input Type'].notna()) & (MGI_uniqueToOld['Input Type'] == 'old symbol')]
# make the input the index 
MGI_convertOld.index = MGI_convertOld['Input'].values

# %%
# if the Symbol is NaN, remove the row 
MGI_convertNew = MGI_uniqueToNew[(MGI_uniqueToNew['Input Type'].notna()) & (MGI_uniqueToNew['Input Type'] == 'old symbol')]
# make the input the index 
MGI_convertNew.index = MGI_convertNew['Input'].values

# %%
MGI_convertOld

# %%
# create a dictionary from the MGI_convertOld['Symbol']
Symbol_dic_old = MGI_convertOld['Symbol'].to_dict()
# create a dictionary from the MGI_convertNew['Symbol']
Symbol_dic_new = MGI_convertNew['Symbol'].to_dict()

# %%
adata_js.X = adata_js.layers["counts"]
adata_ms1.X = adata_ms1.layers["counts"]
adata_ms2.X = adata_ms2.layers["counts"]
adata_b21.X = adata_b21.layers["counts"]
adata_b22.X = adata_b22.layers["counts"]
adata_b23.X = adata_b23.layers["counts"]
adata_b31.X = adata_b31.layers["counts"]
adata_b32.X = adata_b32.layers["counts"]

# %%
del adata_js.layers["counts"]
del adata_ms1.layers["counts"]
del adata_ms2.layers["counts"]
del adata_b21.layers["counts"]
del adata_b22.layers["counts"]
del adata_b23.layers["counts"]
del adata_b31.layers["counts"]
del adata_b32.layers["counts"]

# %%
# update the 4 previous files with old dictionary
adata_js.var_names = [Symbol_dic_old.get(str(gene),str(gene)) for gene in adata_js.var_names]
adata_ms1.var_names = [Symbol_dic_old.get(str(gene),str(gene)) for gene in adata_ms1.var_names]
adata_ms2.var_names = [Symbol_dic_old.get(str(gene),str(gene)) for gene in adata_ms2.var_names]
adata_b21.var_names = [Symbol_dic_old.get(str(gene),str(gene)) for gene in adata_b21.var_names]
# update the 2 new ones with new dictionary
adata_b22.var_names = [Symbol_dic_new.get(str(gene),str(gene)) for gene in adata_b22.var_names]
adata_b23.var_names = [Symbol_dic_new.get(str(gene),str(gene)) for gene in adata_b23.var_names]
adata_b31.var_names = [Symbol_dic_new.get(str(gene),str(gene)) for gene in adata_b31.var_names]
adata_b32.var_names = [Symbol_dic_new.get(str(gene),str(gene)) for gene in adata_b32.var_names]

# %%
adata_js.var_names_make_unique()
adata_ms1.var_names_make_unique()
adata_ms2.var_names_make_unique()
adata_b21.var_names_make_unique()
adata_b22.var_names_make_unique()
adata_b23.var_names_make_unique()
adata_b31.var_names_make_unique()
adata_b32.var_names_make_unique()

# %%
# save the counts layers
adata_js.layers["counts"] = adata_js.X.copy()
adata_ms1.layers["counts"] = adata_ms1.X.copy()
adata_ms2.layers["counts"] = adata_ms2.X.copy()
adata_b21.layers["counts"] = adata_b21.X.copy()
adata_b22.layers["counts"] = adata_b22.X.copy()
adata_b23.layers["counts"] = adata_b23.X.copy()
adata_b31.layers["counts"] = adata_b31.X.copy()
adata_b32.layers["counts"] = adata_b32.X.copy()

# %%
# concatenate the datasets
adata_all = ad.concat([adata_b21, adata_b22, adata_b23, adata_js, adata_ms1, adata_ms2, adata_b31, adata_b32], 
                      join="outer", # https://anndata.readthedocs.io/en/latest/generated/anndata.concat.html
                      label="batch", keys=["B_P2_1", "B_P2X_2", "B_P2X_3", "JS_P3_1","MS_P3_1", "MS_P3_2", "B_P3X_1", "B_P3X_2"],
                      index_unique="-")
# change all NaN to 0 (since pooled data are not sparse matrix)
adata_all.X = np.nan_to_num(adata_all.X) 
adata_all.to_df()

# %%
del adata_js, adata_ms1, adata_ms2, adata_b21, adata_b22, adata_b23, adata_b31, adata_b32

# %%
adata_all.to_df(layer = "counts")

# %%
adata_all.obs['annotation_pre2'].value_counts()

# %%
# update the cell type names 
tmp = adata_all.obs['annotation_pre2']
tmp = ['Immune' if item.startswith('Lymphocyte') else item for item in tmp]
tmp = ['Immune' if item.startswith('Monocyte/Macrophage') else item for item in tmp]
tmp = ['Keratinocyte' if item.startswith('Epithelial') else item for item in tmp]
tmp = ['Fibroblast' if item.startswith('Bone') else item for item in tmp]
tmp = ['Fibroblast' if item.startswith('Chondrocyte') else item for item in tmp]
tmp = ['Remove' if item.startswith('Remove') else item for item in tmp]
tmp = ['Remove' if item.startswith('Erythrocyte') else item for item in tmp]

# tmp = ['Fibroblast' if item.startswith('Fibroblast') else item for item in tmp]
adata_all.obs['annotation_pre3'] = tmp

# %%
# remove the cells that are marked remove
adata_all = adata_all[adata_all.obs['annotation_pre3'] != 'Remove']

# %%
adata_all.obs['annotation_pre3'].value_counts()

# %%
# save the counts to .counts
adata_all.counts = adata_all

# %%
# save an anndata 
adata_all.write_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/Uninj_P2P3_counts_25032024.h5ad")

# %% [markdown]
# ## Preprocess the concaterated dataset

# %%
# load the anndata 
adata_all = sc.read_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/Uninj_P2P3_counts_25032024.h5ad")

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
#Preprocess variables for scran normalization
input_groups = adata_all_copy.obs['groups']
data_mat = adata_all.X.T

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
# save the previous pca and umpa coordinates
adata_all.obsm['X_pca_org'] = adata_all.obsm['X_pca']
adata_all.obsm['X_umap_org'] = adata_all.obsm['X_umap']

# %%
# plot the umap
# make sure it is the original umap without corrections
adata_all.obsm['X_umap'] = adata_all.obsm['X_umap_org']
sc.pl.umap(adata_all, color=['groups', 'batch', 'annotation_pre2'], ncols=1)

# %%
# compute highly variable genes 
sc.pp.highly_variable_genes(adata_all, n_top_genes=4000, 
                            batch_key='batch', subset=False, 
                            layer='counts', flavor='seurat_v3', # counts layer is expected when using seurat_v3
                            inplace=True)

# %%
subset = adata_all[:,adata_all.var['highly_variable']].copy()

# %%
scvi.model.SCVI.setup_anndata(
    subset,
    layer = "counts",
    batch_key="batch"
    # categorical_covariate_keys=["batch"]
    # continuous_covariate_keys=["size_factors"],
    # batch_correction=True,
    # latent_distribution="normal"
)

# %%
model = scvi.model.SCVI(subset, n_latent=30, n_layers=2, gene_likelihood="nb")

# %%
model.train(max_epochs=600, 
            early_stopping=True, 
            check_val_every_n_epoch=5, 
            early_stopping_patience=20, 
            early_stopping_monitor='elbo_validation')

# %%
SCVI_LATENT_KEY = "X_scVI"
adata_all.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

# %%
SCVI_LATENT_KEY = "X_scVI"

# %%
if "X_diffmap" in adata_all.obsm.keys():
    del adata_all.obsm["X_diffmap"]
sc.pp.neighbors(adata_all, use_rep=SCVI_LATENT_KEY,
                n_neighbors=30)
sc.tl.leiden(adata_all, key_added="scvi_leiden05", resolution=0.5)
sc.tl.leiden(adata_all, key_added="scvi_leiden08", resolution=0.8)
sc.tl.leiden(adata_all, key_added="scvi_leiden10", resolution=1.0)

# %%
SCVI_MDE_KEY = "X_scVI_MDE"

# %%
adata_all.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(adata_all.obsm[SCVI_LATENT_KEY])

# %%
SCVI_NORMALIZED_KEY = "X_scVI_normalized"
subset.layers[SCVI_NORMALIZED_KEY] = model.get_normalized_expression(subset, library_size=10e4)


# %%
subset.write_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/Uninj_P2P3_counts_25032024_2_subset.h5ad")

# %%
# load subset
subset = sc.read_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/Uninj_P2P3_counts_25032024_2_subset.h5ad")

# %%
# save the file as a h5ad file
adata_all.write_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/Uninj_P2P3_counts_25032024_2.h5ad")


# %%
# load the file 
adata_all = sc.read_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/Uninj_P2P3_counts_25032024_2.h5ad")

# %%
# save the model 
model.save("./Uninjured_analysis_results/scvi_model_25032024_2.model", overwrite=True)

# %%
# load the model
model = scvi.model.SCVI.load("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/Uninjured_analysis_results/scvi_model_25032024.model", adata=adata_all)

# %%
# load the model 2 
model = scvi.model.SCVI.load("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/Uninjured_analysis_results/scvi_model_25032024_2.model", adata=subset)

# %%
# look at the hvgnbatches
n_batches = subset.var["highly_variable_nbatches"].value_counts()
ax = n_batches.plot(kind="bar")
n_batches

# %% [markdown]
# ## plot of the gene expression level

# %%
# name the subset as adata_all hereafter
adata_all = subset
adata_all

# %%
# define the active pca and umap 
adata_all.obsm['X_pca'] = adata_all.obsm['X_scVI']
adata_all.obsm['X_umap'] = adata_all.obsm['X_scVI_MDE']

# %%
from matplotlib import colors
#Define a nice colour map for gene expression
colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
colors3 = plt.cm.Greys_r(np.linspace(0.7,0.8,40))
colorsComb = np.vstack([colors3, colors2])
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colorsComb)

# %%
# Trial
key = "Tnxb"
sc.pl.umap(adata_all, color=[key], use_raw=True, color_map=mymap, size= 20)
sc.pl.violin(adata_all, [key], groupby='Amp_location', rotation=90, stripplot=True, multi_panel=True, use_raw=True)


# %%
# plot the umap of the adata_fibroblast
sc.tl.leiden(adata_all, key_added="scvi_leiden14", resolution=1.4)
sc.pl.umap(adata_all, color=['scvi_leiden14'], legend_loc='on data', )
sc.pl.umap(adata_all, color=['batch'])

# %%
sc.pl.umap(adata_all, color=['scvi_leiden14'], legend_loc='on data', )

# %%
# show the cell counts per cluster
adata_all.obs['scvi_leiden14'].value_counts()

# %%
# Calculate the cell counts per cluster
cell_counts = adata_all.obs['scvi_leiden14'].value_counts()

# Calculate the percentage of cells in each cluster
cell_percentages = cell_counts / cell_counts.sum() * 100

# Get the clusters that are too small
small_clusters = cell_percentages[cell_percentages < 0.1].index

# Filter out the small clusters
adata_all = adata_all[~adata_all.obs['scvi_leiden14'].isin(small_clusters), :]

# %%
# plot again after removal of small groups
sc.pl.umap(adata_all, color=['scvi_leiden14'], legend_loc='on data', )

# %% [markdown]
# ## Find Markers / Label cell types

# %% [markdown]
# Plot techniques from scanpy for presentation purposes
# https://scanpy-tutorials.readthedocs.io/en/latest/plotting/core.html

# %% [markdown]
# ### scanpy find marker genes

# %%
sc.tl.rank_genes_groups(adata_all, 
                        use_raw=True,
                        groupby='scvi_leiden14', 
                        method='wilcoxon', 
                        corr_method='benjamini-hochberg')

# %%
results = adata_all.uns['rank_genes_groups']
results

# %%
groups = results['names'].dtype.names
out =  np.array([[0,0,0,0,0]])
for group in groups:
    out = np.vstack((out, np.vstack((results['names'][group],  
                                     np.float64(results['pvals_adj'][group]), 
                                     np.float64(results['logfoldchanges'][group]), 
                                     np.float64(results['scores'][group]), 
                                     np.array([group]*len(results['names'][group])).astype('object'))).T))

# %%
Markers = pd.DataFrame(out[1:], columns=['Gene_symb', 'pvals_adj', 'lfc', 'scores', 'group'])
Markers.groupby('group').head(10)

# %%
Markers['pvals_adj'] = Markers['pvals_adj'].astype(float)
Markers['lfc'] = Markers['lfc'].astype(float)
Markers['scores'] = Markers['scores'].astype(float)

# %%
Markers_filter = Markers[(Markers['pvals_adj'] < 0.05) & (abs(Markers['lfc']) > 2)]
Markers_filter

# %%
sc.pl.rank_genes_groups(adata_all, n_genes=25, sharey=False)

# %%
sc.pl.rank_genes_groups(adata_all, n_genes=25, sharey=False, groups=['12'],)

# %%
sc.pl.umap(adata_all, color=['scvi_leiden14'], legend_loc = 'on data')

# %%
# check
# key = 'Sdc1'
key = 'Pecam1'
sc.pl.umap(adata_all, color=[key], use_raw=True, color_map=mymap, size= 10)
sc.pl.violin(adata_all, key, groupby='scvi_leiden14', use_raw=True, rotation=90, stripplot=True, log=False)

# %%
# Categories to rename
adata_all.obs['annotation_int'] = adata_all.obs['scvi_leiden14']
adata_all.rename_categories('annotation_int', ['Fibroblast.1','Fibroblast.2','Fibroblast.3','Keratinocyte.1','Pericyte/SMC.1','Fibroblast.4','Pericyte/SMC.2','Immune.1','Fibroblast.5','Endothelial.1','Keratinocyte.2','Fibroblast.6','Fibroblast.7','Fibroblast.8','Fibroblast.9','Pericyte/SMC.3','Keratinocyte.3','Fibroblast.10','Schwann','Keratinocyte.4','Fibroblast.11','Endothelial.2','Fibroblast.12','Keratinocyte.5','Immune.2','Immune.3','Keratinocyte.6','Immune.4','Fibroblast.13','Endothelial.3','Fibroblast.14','Lymphatic Endothelial','Immune.5','Immune.6','Endothelial.4','Fibroblast.15','Keratinocyte.7','Fibroblast.16','Erythrocyte','Fibroblast.17','Keratinocyte.8'])

# %%
tmp = adata_all.obs['annotation_int']
tmp = ['Fibroblast' if item.startswith('Fibroblast') else item for item in tmp]
tmp = ['Endothelial' if item.startswith('Endothelial') else item for item in tmp]
tmp = ['Schwann' if item.startswith('Schwann') else item for item in tmp]
tmp = ['Pericyte/SMC' if item.startswith('Pericyte/SMC') else item for item in tmp]
tmp = ['Keratinocyte' if item.startswith('Keratinocyte') else item for item in tmp]
tmp = ['Immune' if item.startswith('Immune') else item for item in tmp]

adata_all.obs['annotation_int2'] = tmp

# %%
# remove the Erythrocyte from the clusters 
adata_all = adata_all[adata_all.obs['annotation_int2']!='Erythrocyte' , :]

# %%
adata_all.obs['annotation_int2'] = adata_all.obs['annotation_int2'].astype('category')

# %%
adata_all.obs['annotation_int2'] = adata_all.obs['annotation_int2'].cat.reorder_categories(['Fibroblast','Schwann','Immune','Pericyte/SMC','Keratinocyte','Endothelial','Lymphatic Endothelial'])

# %%
sc.pl.umap(adata_all, color=['annotation_int2', 'annotation_pre3'], legend_loc='on data', legend_fontsize=10)

# %%
sc.pl.umap(adata_all, color=['annotation_int2'], save=f'_Uninj_all_annotation.pdf')

# %%
sc.pl.umap(adata_all, color=['Amp_location', 'batch'])

# %%
# create a new copy of the adata 
adata_all_copy = adata_all.raw.to_adata().copy()
# scale and store results in layer
adata_all_copy.layers["scaled"] = sc.pp.scale(adata_all_copy, copy=True, ).X

# %%
adata_all_copy.obs['annotation_int2'] = adata_all_copy.obs['annotation_int2'].astype("category")

# %%
adata_all_copy.obs['annotation_int2'] = adata_all_copy.obs['annotation_int2'].cat.reorder_categories(['Fibroblast','Schwann','Immune','Pericyte/SMC','Keratinocyte','Endothelial','Lymphatic Endothelial'])


# %%
# Fibroblast, Pericyte/SMC, Schwann, Immune, Keratinocyte, Endothelial, Lymphatic Endothelial
marker_genes_plHeat = [ 'Pdgfra','Col1a1','Lum',  # Fibroblast 
                        'Sox10','Plp1','Cdh19', # Schwann
                        'Ptprc','Lyz2','Cd53', # Immune
                        'Rgs5','Acta2','Myh11', # Pericyte/SMC
                        'Krt5','Krt14','Cdh1', # Keratinocyte
                        'Cdh5','Pecam1','Eng', # Endothelial
                        'Prox1','Flt4','Lyve1', # Lymphatic Endothelial
]
adata_all_copy.obs['annotation_int2'] = adata_all_copy.obs['annotation_int2'].cat.reorder_categories(['Fibroblast','Schwann','Immune','Pericyte/SMC','Keratinocyte','Endothelial','Lymphatic Endothelial'])
sc.pl.heatmap(adata_all_copy, marker_genes_plHeat, groupby='annotation_int2', 
                use_raw = True, swap_axes=False,
                layer = "scaled",
                vmin=-2,
                vmax=2,
                cmap="coolwarm", 
                save = f"_Uninj_all_annotation.pdf"
                )

# %%
del adata_all_copy

# %%
# save the file as a h5ad file
adata_all.write_h5ad("./anndata/Uninj_P2P3_counts_25032024_2_subset.h5ad")


