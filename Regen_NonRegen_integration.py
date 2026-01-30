# %% [markdown]
# # Regen - NonRegen scVi integration 

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
from matplotlib import colors
#Define a nice colour map for gene expression
colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
colors3 = plt.cm.Greys_r(np.linspace(0.7,0.8,40))
colorsComb = np.vstack([colors3, colors2])
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colorsComb)

# %%
import os
# os.environ['R_HOME'] = '/Users/jjyw2/AppData/Local/miniforge3/envs/scvi-env/Lib/R'
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.2'

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
# read the h5ad file
adata_regen1 = sc.read_h5ad("anndata/Regen_14DPA_1_B.h5ad")
adata_regen1.obs['batch'] = 'RB1'
adata_regen1.obs['Amp_location'] = 'P3'
adata_regen1

# %%
# read the h5ad file
adata_regen2 = sc.read_h5ad("anndata/Regen_14DPA_2_B.h5ad")
adata_regen2.obs['batch'] = 'RB2'
adata_regen2.obs['Amp_location'] = 'P3'
adata_regen2

# %%
# read the h5ad file
adata_regen3 = sc.read_h5ad("anndata/Regen_14DPA_JS.h5ad")
adata_regen3.obs['batch'] = 'RJS'
adata_regen3.obs['Amp_location'] = 'P3'
adata_regen3

# %%
# read the h5ad file
adata_NR1 = sc.read_h5ad("anndata/NonRegen_14DPA_1.h5ad")
adata_NR1.obs['batch'] = 'NRB1'
adata_NR1.obs['Amp_location'] = 'P2'
adata_NR1

# %%
# read the h5ad file 
adata_NR2 = sc.read_h5ad("anndata/NR_14DPA_Feb24_SoupX2_1.h5ad")
adata_NR2.obs['batch'] = 'NRB2'
adata_NR2.obs['Amp_location'] = 'P2'
adata_NR2

# %%
# read the h5ad file 
adata_NR3 = sc.read_h5ad("anndata/NR_14DPA_Feb24_soupX_2.h5ad")
adata_NR3.obs['batch'] = 'NRB3'
adata_NR3.obs['Amp_location'] = 'P2'
adata_NR3

# %%
del adata_NR3.raw
del adata_NR2.raw
del adata_NR1.raw
del adata_regen1.raw
del adata_regen2.raw
del adata_regen3.raw

# %%
adata_NR3[:,['Pakap','Pakap.1']].to_df()

# %%
def flatten(xss):
    return [x for xs in xss for x in xs]

# %%
new_ver = set(flatten([adata_NR3.var.index, adata_NR2.var.index]))

# %%
old_ver = set(flatten([adata_NR1.var.index, adata_regen3.var.index, adata_regen2.var.index, adata_regen1.var.index]))

# %%
unique_to_new = [gene for gene in new_ver if gene not in old_ver]
unique_to_old = [gene for gene in old_ver if gene not in new_ver]

# %%
df = pd.DataFrame(unique_to_new, columns=['Gene'])
df['Gene'] = df['Gene'].str.replace('\..*', '', regex=True)
# Export the DataFrame to a CSV file
df.to_csv('unique_to_new.csv', index=False)

# %%
# load the generated conversion from MGI 
MGI_uniqueToOld = pd.read_excel("MGIBatchReport_uniqueToOld.xlsx", header=0)
# load the generated conversion from MGI
MGI_uniqueToNew = pd.read_excel("MGIBatchReport_uniqueToNew.xlsx", header=0)

# %%
# if the Symbol is NaN, remove the row 
MGI_convertOld = MGI_uniqueToOld[(MGI_uniqueToOld['Input Type'].notna()) & (MGI_uniqueToOld['Input Type'] == 'old symbol')]
# make the input the index 
MGI_convertOld.index = MGI_convertOld['Input'].values

# %%
MGI_convertOld.loc['1500015O10Rik']


# %%
# if the Symbol is NaN, remove the row 
MGI_convertNew = MGI_uniqueToNew[(MGI_uniqueToNew['Input Type'].notna()) & (MGI_uniqueToNew['Input Type'] == 'old symbol')]
# make the input the index 
MGI_convertNew.index = MGI_convertNew['Input'].values

# %%
MGI_convertNew

# %%
# create a dictionary from the MGI_convertOld['Symbol']
Symbol_dic_old = MGI_convertOld['Symbol'].to_dict()
# create a dictionary from the MGI_convertNew['Symbol']
Symbol_dic_new = MGI_convertNew['Symbol'].to_dict()

# %%
adata_regen1.X = adata_regen1.layers["counts"]
adata_regen2.X = adata_regen2.layers["counts"]
adata_regen3.X = adata_regen3.layers["counts"]
adata_NR1.X = adata_NR1.layers["counts"]
adata_NR2.X = adata_NR2.layers["counts"]
adata_NR3.X = adata_NR3.layers["counts"]

# %%
del adata_regen1.layers["counts"]
del adata_regen2.layers["counts"]
del adata_regen3.layers["counts"]
del adata_NR1.layers["counts"]
del adata_NR2.layers["counts"]
del adata_NR3.layers["counts"]

# %%
# update the 4 previous files with old dictionary
adata_NR1.var_names = [Symbol_dic_old.get(str(gene),str(gene)) for gene in adata_NR1.var_names]
adata_regen1.var_names = [Symbol_dic_old.get(str(gene),str(gene)) for gene in adata_regen1.var_names]
adata_regen2.var_names = [Symbol_dic_old.get(str(gene),str(gene)) for gene in adata_regen2.var_names]
adata_regen3.var_names = [Symbol_dic_old.get(str(gene),str(gene)) for gene in adata_regen3.var_names]
# update the 2 new ones with new dictionary
adata_NR2.var_names = [Symbol_dic_new.get(str(gene),str(gene)) for gene in adata_NR2.var_names]
adata_NR3.var_names = [Symbol_dic_new.get(str(gene),str(gene)) for gene in adata_NR3.var_names]

# %%
adata_NR1.var_names_make_unique()
adata_NR2.var_names_make_unique()
adata_NR3.var_names_make_unique()
adata_regen1.var_names_make_unique()
adata_regen2.var_names_make_unique()
adata_regen3.var_names_make_unique()

# %%
# save the counts layers
adata_NR1.layers["counts"] = adata_NR1.X.copy()
adata_NR2.layers["counts"] = adata_NR2.X.copy()
adata_NR3.layers["counts"] = adata_NR3.X.copy()
adata_regen1.layers["counts"] = adata_regen1.X.copy()
adata_regen2.layers["counts"] = adata_regen2.X.copy()
adata_regen3.layers["counts"] = adata_regen3.X.copy()

# %% [markdown]
# # SCVI

# %%
# concatenate the datasets
adata_all = ad.concat([adata_regen1, adata_regen2, adata_regen3, adata_NR1, adata_NR2, adata_NR3], 
                      join="outer", # https://anndata.readthedocs.io/en/latest/generated/anndata.concat.html
                      label="batch", keys=["RB1", "RB2", "RJS", "NRB1", "NRB2", "NRB3"],
                      index_unique="-")
# change all NaN to 0 (since pooled data are not sparse matrix)
adata_all.X = np.nan_to_num(adata_all.X) 
adata_all.to_df()

# %%
del adata_regen1, adata_regen2, adata_regen3, adata_NR1, adata_NR2, adata_NR3

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
adata_all.obs['annotation_pre3'] = tmp

# %%
# remove the cells that are marked remove
adata_all = adata_all[adata_all.obs['annotation_pre3'] != 'Remove']

# %%
adata_all.obs['annotation_pre3'].value_counts()

# %% [markdown]
# ## Preprocess the concaterated dataset

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
# prove scvi integration is needed
adata_all.obsm['X_umap'] = adata_all.obsm['X_umap_org']
sc.pl.umap(adata_all, color=['groups', 'batch', 'annotation_pre2'], ncols=1)

# %%
# compute highly variable genes 
sc.pp.highly_variable_genes(adata_all, n_top_genes=2000, 
                            batch_key='batch', subset=True, 
                            layer='counts', flavor='seurat_v3', # counts layer is expected when using seurat_v3
                            inplace=True)

# %%
scvi.model.SCVI.setup_anndata(
    adata_all,
    layer = "counts",
    batch_key="batch"
)

# %%
model = scvi.model.SCVI(adata_all, n_latent=30, n_layers=2, gene_likelihood="nb")

# %%
model.train(max_epochs=400, 
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
sc.pl.embedding(
    adata_all,
    basis=SCVI_MDE_KEY,
    color=["batch", "scvi_leiden10", "annotation_pre2","annotation_pre3", "Amp_location"],
    frameon=False,
    ncols=1,
)

# %%
SCVI_NORMALIZED_KEY = "X_scVI_normalized"
adata_all.layers[SCVI_NORMALIZED_KEY] = model.get_normalized_expression(adata_all, library_size=10e4)


# %%
# save the file as a h5ad file
adata_all.write_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/R_NR_14DPA_25032024.h5ad")


# %%
# load the file 
adata_all = sc.read_h5ad("./anndata/R_NR_14DPA_25032024.h5ad")

# %%
# save the model 
model.save("./Regen-NonRegen_analysis_results/scvi_model_25032024.model", overwrite=True)

# %%
# load the model
model = scvi.model.SCVI.load("./Regen-NonRegen_analysis_results/scvi_model_25032024.model", adata=adata_all)

# %% [markdown]
# ## plot of the gene expression level

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
key = "Prox1"
sc.pl.umap(adata_all, color=[key], use_raw=True, color_map=mymap, size= 40, save= f"_all_{key}.pdf")
sc.pl.violin(adata_all, [key], groupby='scvi_leiden10', rotation=90, stripplot=True, multi_panel=True, save = f"_all_{key}.pdf")

# %%
sc.pl.umap(adata_all, color=['scvi_leiden10'], legend_loc = "on data", use_raw=True, color_map=mymap, size= 10)

# %%
# check
# key = 'Sdc1'
key = 'Plp1'
sc.pl.umap(adata_all, color=[key], use_raw=True, color_map=mymap, size= 10)
sc.pl.violin(adata_all, key, groupby='scvi_leiden10', use_raw=True, rotation=90, stripplot=True, log=False)

# %%
# create new annotation based on clusters 
# Categories to rename
adata_all.obs['annotation_int'] = adata_all.obs['scvi_leiden10']
adata_all.rename_categories('annotation_int', ['Immune.1','Fibroblast.1','Fibroblast.2','Fibroblast.3','Fibroblast.4','Fibroblast.5','Fibroblast.6','Endothelial.1','Fibroblast.7','Fibroblast.8','Immune.2','Keratinocyte.1','Pericyte/SMC','Keratinocyte.2','Fibroblast.9','Fibroblast.10','Immune.3','Fibroblast.11','Immune.4','Lymphatic Endothelial','Schwann','Fibroblast.12','Fibroblast.13','Fibroblast.14','Endothelial.2','Erythrocyte'])

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
# Annotation
sc.pl.umap(adata_all, color="annotation_int2", use_raw=True, color_map=mymap, size= 10, save= f"_all_annotation.pdf")

# %%
# Annotation
sc.pl.umap(adata_all, color="Amp_location", use_raw=True, color_map=mymap, size= 10, save= f"_all_ampLocation.pdf")

# %%
# Annotation
sc.pl.umap(adata_all, color="batch", use_raw=True, color_map=mymap, size= 10, save= f"_all_batch.pdf")

# %% [markdown]
# ## Find Markers / Label cell types

# %% [markdown]
# Plot techniques from scanpy for presentation purposes
# https://scanpy-tutorials.readthedocs.io/en/latest/plotting/core.html

# %% [markdown]
# ### scanpy find marker genes

# %%
adata_all.to_df()

# %%
sc.tl.rank_genes_groups(adata_all, 
                        use_raw=True,
                        groupby='annotation_pre3', 
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
# create a new copy of the adata 
adata_all_copy = adata_all.raw.to_adata().copy()
# scale and store results in layer
adata_all_copy.layers["scaled"] = sc.pp.scale(adata_all_copy, copy=True, ).X

# %%
# Fibroblast, Pericyte/SMC, Schwann, Immune, Keratinocyte, Endothelial, Lymphatic Endothelial
marker_genes_plHeat = ['Pdgfra','Col1a1','Lum',  # Fibroblast 
                        'Sox10','Plp1','Cdh19', # Schwann
                        'Ptprc','Lyz2','Cd53', # Immune
                        'Rgs5','Acta2','Myh11', # Pericyte/SMC
                        'Krt5','Krt14','Cdh1', # Keratinocyte
                        'Cdh5','Pecam1','Eng', # Endothelial
                        'Prox1','Flt4','Lyve1', # Lymphatic Endothelial
]
adata_all.obs['annotation_int2'] = adata_all.obs['annotation_int2'].cat.reorder_categories(['Fibroblast','Schwann','Immune','Pericyte/SMC','Keratinocyte','Endothelial','Lymphatic Endothelial'])
sc.pl.heatmap(adata_all_copy, marker_genes_plHeat, groupby='annotation_int2', 
                use_raw = True, swap_axes=False,
                layer = "scaled",
                vmin=-2,
                vmax=2,
                cmap="coolwarm", 
                save = f"_all_annotation.pdf"
                )

# %%
sc.tl.rank_genes_groups(adata_all, groupby='scvi_leiden10', method='wilcoxon')
sc.pl.rank_genes_groups(adata_all, n_genes=25, sharey=False)

# %%
sc.pl.rank_genes_groups_dotplot(adata_all, n_genes=5, groupby='annotation_pre3', dendrogram=True, standard_scale='var')

# %%
sc.tl.dendrogram(adata_all, 'annotation_pre3')

# %%
sc.pl.correlation_matrix(adata_all, 'annotation_pre3', figsize=(10,10))

# %%
# save the marker genes dataframe 
Markers.to_csv("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/Regen-NonRegen_analysis_results/R_NR_14DPA_25032024_markers.csv")

# %% [markdown]
# ## GO analysis
# perform with goatools
# reference and adapted from https://www.youtube.com/watch?v=ONiWugVEf2s

# %%
# load the marker genes dataframe if not loaded
Markers = pd.read_csv("./Regen-NonRegen_analysis_results/R_NR_14DPA_25032024_markers.csv", index_col=0)

# %%
Markers

# %% [markdown]
# consider using https://pypi.org/project/r-functions/

# %%
%%R -i Markers 
# Perform pathway analysis on the top 20 genes 
library("clusterProfiler")
library("org.Mm.eg.db")
library("dplyr")
library("ggplot2")

# take the top 20 genes from each group
Markers_top20 <- Markers %>% group_by(group) %>% top_n(n = 20, wt = scores) # use scores or lfc or pvals_adj??
# print(Markers_top20)
# add extra column of entrez id
Markers_top20$entrez <- mapIds(org.Mm.eg.db, Markers_top20$Gene_symb, "ENTREZID", "SYMBOL", multiVals = "first")
# remove the lines that could not be mapped
Markers_top20 <- Markers_top20[!is.na(Markers_top20$entrez),]
# create a list of genes for each cluster
Markers_top20 <- split(Markers_top20$entrez, Markers_top20$group)
# print(Markers_top20)

# compareCluster
# One of "groupGO", "enrichGO", "enrichKEGG", "enrichDO" or "enrichPathway"
clustersummary <- compareCluster(geneClusters = Markers_top20, fun = "enrichGO", OrgDb = org.Mm.eg.db, ont = "BP", pvalueCutoff = 0.05, pAdjustMethod = "BH", qvalueCutoff = 0.05, readable = TRUE)
print(clustersummary)
pdf("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/temp.pdf", 
    width = 15, height = 30)
title <- "GO analysis of the top 20 upregulated genes in each group"
p1 <- dotplot(clustersummary, showCategory = 5) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        plot.margin = margin(1, 1, 1, 10),
        axis.title.x = element_blank(),
        axis.text.y = element_text(size = 8),
        legend.text = element_text(size = 8),
        legend.title = element_text(size = 8)) +
    scale_color_gradient("p.adjust", high = "#deebf7", low = "#3182bd")
print(p1 + ggtitle(title))
dev.off()
print(p1)

# %% [markdown]
# ### subsetting the data for marker gene analysis 

# %%
# load the adata_all
adata_all = sc.read_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/R_NR_14DPA_25032024.h5ad")

# %%
# define the active pca and umap
adata_all.obsm['X_pca'] = adata_all.obsm['X_scVI']
adata_all.obsm['X_umap'] = adata_all.obsm['X_scVI_MDE']

# %%
# plot the scvi leiden clusters 1.0 resolution with legends on top 
sc.pl.umap(adata_all, color=['scvi_leiden10'], 
           legend_loc='on data', 
           frameon=False, 
           ncols=1)
# also plot the clusters with annotation_pre3
sc.pl.umap(adata_all, color=['annotation_pre3'], 
           legend_loc='on data', 
           frameon=False, 
           ncols=1)
# also plot the clusters with annotation_pre3
sc.pl.umap(adata_all, color=['Amp_location'], 
           
           frameon=False, 
           ncols=1)

# %%
sc.pl.umap(adata_all, color=['Mest', 'Tnxb', 'Comp'], use_raw=True, color_map=mymap, size= 40)

# %%
sc.pl.umap(adata_all, color=['Arsi', 'Fbn2', 'Lrrc17', 'Sall4'], use_raw=True, color_map=mymap, size= 40)

# %%
# SMC marker genes 
sc.pl.umap(adata_all, color=['Tagln', 'Acta2', 'Cd44'], use_raw=True, color_map=mymap, size= 40)

# %%
# trial
sc.pl.umap(adata_all, color=['Ptprc', 'Cd74', 'Cd44'], use_raw=True, color_map=mymap, size= 40)

# %%
# create a violin plot 
sc.pl.violin(adata_all, ['Hapln1'], groupby='annotation_pre3', use_raw=True, rotation=90, stripplot=True, multi_panel=True)

# %%
all_df = pd.DataFrame(adata_all.obs.groupby(['batch','annotation_pre3']).size(), columns = ['count'])
# add an extra column of percentage
all_df['percentage'] = all_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
all_df

# %% [markdown]
# Analysis with full dataset may be misinformation as the preparation procedures between different batches are very different, affecting the overall cellular proportions. Reporting this should be dealt with extreme caution!

# %%
# plot a bar chart of the percentage of cells in each cluster
all_ax = all_df['percentage'].unstack().plot(kind='bar', stacked=True, figsize=(10,5),
                      title='Cell type percentage in each batch', 
                      ylabel='Percentage', xlabel='Batch', legend=False)
all_labels = ['Endothelial', 'Fibroblast', 'Immune','Keratinocyte','Lymphatic Endothelial','Pericyte/SMC','Schwann']
all_ax.legend(all_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# %%
# plot the umap of the scvi_leiden10
sc.pl.umap(adata_all, color=['scvi_leiden10'], ncols=1, frameon=False, legend_loc='on data')
sc.pl.umap(adata_all, color=['Pdgfra'], ncols=1, frameon=False, legend_loc='on data', use_raw=True, color_map=mymap, size= 40)

# %%
# remove all the cells that are not in the groups 
adata_fibroblast = adata_all[adata_all.obs['scvi_leiden10'].isin(['1','2','3','4','5','6','8','9','14','15','17','21','22','23']), :]

# %%
adata_fibroblast = adata_fibroblast.copy()
# use scvi to group the cells again 
scvi.model.SCVI.setup_anndata(
    adata_fibroblast,
    layer = "counts",
    batch_key="batch"
    # categorical_covariate_keys=["batch"]
    # continuous_covariate_keys=["size_factors"],
    # batch_correction=True,
    # latent_distribution="normal"
)

# %%
adata_fibroblast.obs['scvi_leiden10'].value_counts()

# %%
fib_df = pd.DataFrame(adata_fibroblast.obs.groupby(['batch','scvi_leiden10']).size(), columns = ['count'])
# add an extra column of percentage
fib_df['percentage'] = fib_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
fib_df

# %%
# plot a bar chart of the percentage of cells in each cluster
ax = fib_df['percentage'].unstack().plot(kind='bar', stacked=True, figsize=(10,5),
                      title='Percentage of cells in each cluster of PDGFRa+ cells', 
                      ylabel='Percentage', xlabel='Batch', legend=False)
labels = ['Fib.1','Fib.2','Fib.3','Fib.4','Fib.5','Fib.6','Fib.8','Fib.9','Fib.14','Fib.15','Fib.17','Fib.21','Fib.22','Fib.23']
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# %%
# rank the genes in each cluster
sc.tl.rank_genes_groups(adata_fibroblast, groupby='scvi_leiden10', method='wilcoxon', use_raw=True)

# %%
sc.pl.rank_genes_groups(adata_fibroblast, n_genes=25, sharey=False)

# %%
# plot a violin plot 
sc.pl.violin(adata_fibroblast, ['Mest', 'Tagln', 'Mif'], groupby='Amp_location', use_raw=True, rotation=90, stripplot=True, multi_panel=True)

# %%
# create a gene of interest list 
gene_list = ['Has1','Has2','Has3','Hapln1','Cd44', 'Cd74','Hmmr','Tnfaip6','Id1','Id2','Id3','Col1a1','Hdac1','Sp9','Comp'] # Fgf8

# %%
# plot the violin pots for the genes of interest
sc.pl.violin(adata_fibroblast, gene_list, 
             groupby='Amp_location', 
             use_raw=True, 
             rotation=90, 
             stripplot=True, 
             ncols = 4)

# %%
adata_fibroblast_fil = adata_fibroblast[adata_fibroblast.obs['annotation_pre3'].isin(['Fibroblast'])]
sc.tl.rank_genes_groups(adata_fibroblast_fil, groupby = 'scvi_leiden10', method='wilcoxon', use_raw=True)

# %%
# plot the violin pots for the genes of interest
sc.pl.violin(adata_fibroblast_fil, gene_list, groupby='Amp_location', use_raw=True)

# %%
# recreate the umap for the fibroblast clusters with scvi leiden 10
adata_fibroblast = adata_fibroblast.copy()
scvi.model.SCVI.setup_anndata(
    adata_fibroblast,
    layer = "counts",
    batch_key="batch"
)


# %%
model_fib = scvi.model.SCVI(adata_fibroblast, n_latent=30, n_layers=2, gene_likelihood="nb")

# %%
model_fib.train(max_epochs=600, 
            early_stopping=True, 
            check_val_every_n_epoch=5, 
            early_stopping_patience=20, 
            early_stopping_monitor='elbo_validation')

# %%
SCVI_LATENT_KEY_FIB = "X_fib_scVI"
adata_fibroblast.obsm[SCVI_LATENT_KEY_FIB] = model_fib.get_latent_representation()
sc.pp.neighbors(adata_fibroblast, use_rep=SCVI_LATENT_KEY_FIB,
                n_neighbors=30)
sc.tl.leiden(adata_fibroblast, key_added="scvi_fib_leiden06", resolution=0.6)

# %%
sc.tl.leiden(adata_fibroblast, key_added="scvi_fib_leiden05", resolution=0.5)
sc.tl.leiden(adata_fibroblast, key_added="scvi_fib_leiden04", resolution=0.4)
sc.tl.leiden(adata_fibroblast, key_added="scvi_fib_leiden07", resolution=0.7)

# %%
SCVI_MDE_KEY_FIB = "X_fib_scVI_MDE"
adata_fibroblast.obsm[SCVI_MDE_KEY_FIB] = scvi.model.utils.mde(adata_fibroblast.obsm[SCVI_LATENT_KEY_FIB])


# %%
sc.pl.embedding(
    adata_fibroblast,
    basis=SCVI_MDE_KEY_FIB,
    color=["batch", "scvi_leiden10","scvi_fib_leiden04", "scvi_fib_leiden05","scvi_fib_leiden06", "scvi_fib_leiden07", "Amp_location"],
    frameon=False,
    ncols=1,
)

# %%
# redefine the current umap
adata_fibroblast.obsm['X_umap'] = adata_fibroblast.obsm['X_fib_scVI_MDE']   

sc.pl.umap(adata_fibroblast, color=['scvi_fib_leiden07','scvi_fib_leiden06','scvi_fib_leiden05','scvi_fib_leiden04'], ncols=1, frameon=False, legend_loc='on data')

# %%
# remove the clusters 13/14 from the clustering leiden07
adata_fibroblast = adata_fibroblast[~adata_fibroblast.obs['scvi_fib_leiden07'].isin(['13','14'])]

# %%
sc.tl.leiden(adata_fibroblast, key_added="scvi_fib_leiden045", resolution=0.45)

# %%
# redefine the current umap
adata_fibroblast.obsm['X_umap'] = adata_fibroblast.obsm['X_fib_scVI_MDE']   
# plot the umap
sc.pl.umap(adata_fibroblast, color=['scvi_fib_leiden045', 'Amp_location','batch'], ncols=1, frameon=True, save = f"_fib_clus-batch-ampLocat.pdf")

# %%
# save adata_fibroblast
sc.write("./anndata/R_NR_14DPA_fibroblast.h5ad", adata_fibroblast)

# %%
# save the scvi model
model_fib.save("./Regen-NonRegen_analysis_results/scvi_model_fib_25032024.model", overwrite=True)


