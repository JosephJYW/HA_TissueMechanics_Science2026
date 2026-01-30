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
sc.settings._vector_friendly = False

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
# # take the 4MU control group 

# %%
# load full counts anndata object
adata_4MU = sc.read_h5ad("./anndata/HA_4MU_fullcounts_30052024.h5ad")
adata_4MU

# %%
adata_4MU_subset = sc.read_h5ad("./anndata/HA_4MU_HvgSubset_03072024.h5ad")
adata_4MU_subset

# %%
# transfer information from subset to full counts
adata_4MU.obs = adata_4MU_subset.obs.copy()
adata_4MU.uns = adata_4MU_subset.uns.copy()
adata_4MU.obsm = adata_4MU_subset.obsm.copy()

# %%
# load the adata_fibroblast
adata_fibroblast = sc.read_h5ad("./anndata/HA_4MU_Fibroblast_subset_03072024.h5ad")
adata_fibroblast

# %%
# copy the adata_fibroblast to adata_4MU_fibroblast
adata_4MU_fibroblast = adata_4MU[adata_fibroblast.obs_names].copy()
adata_4MU_fibroblast

# %%
# # remove cells in the ("./anndata/HA_4MU_fib_manualCluster3_remove_27062024.csv")
# remove = pd.read_csv("./anndata/HA_4MU_fib_manualCluster3_remove_27062024.csv", header=0)
# remove = remove["barcode"].tolist()
# # split the string by '-' and remove the last element
# remove = [x.split("-")[:-1] for x in remove]
# # join the string by '-'
# remove = ["-".join(x) for x in remove]
# remove

# %%
# remove cluster 2 from scvi_leiden10
# adata_4MU = adata_4MU[~adata_4MU.obs['scvi_leiden10'].isin(['2'])]
# adata_fibroblast = adata_fibroblast[~adata_fibroblast.obs['scvi_leiden10'].isin(['2'])]

# %%
# # remove the cells from adata_4MU
# adata_4MU = adata_4MU[~adata_4MU.obs_names.isin(remove)].copy()
# adata_4MU

# # remove the cells from adata_fibroblast
# adata_fibroblast = adata_fibroblast[~adata_fibroblast.obs_names.isin(remove)].copy()
# adata_fibroblast

# %%
# plot the 4MU umap 
sc.pl.umap(adata_4MU, color=["scvi_leiden08_2"], ncols=1, wspace=0.3)

# %%
del adata_4MU, adata_fibroblast

# %% [markdown]
# # load the regennonregen-fibroblast

# %%
# load the adata_fibroblast
adata_RNR_fibroblast = sc.read_h5ad("./anndata/R_NR_14DPA_fib_fullcounts_08052024_2.h5ad")
adata_RNR_fibroblast

# %%
adata_RNR_fibroblast.obs['Amp_location'].value_counts()

# %%
# take all cells that are in the regeneration fibroblast subset (Amp_location == 'P3')
adata_Regen = adata_RNR_fibroblast[adata_RNR_fibroblast.obs['Amp_location'] == 'P3'].copy()
adata_Regen

# %%
adata_Regen.obs['batch']

# %%
del adata_RNR_fibroblast

# %% [markdown]
# # Integrate the anndata 

# %%
# concaterate the datsets
adata_all = ad.concat([adata_4MU_fibroblast, adata_Regen], 
                      join='outer', 
                      label='Condition',
                      keys=['4MUControl', 'RegenP3'],
                      index_unique='-')
# change all NaN to 0 (since pooled data are not sparse matrix)
adata_all.X = np.nan_to_num(adata_all.X) 
adata_all.to_df()

# %%
adata_all.to_df(layer = "counts")

# %%
adata_all.obs['annotation_int2'].value_counts()

# %%
# save this adata_all
adata_all.write_h5ad("./anndata/HA_4MU_RegenP3_fullcounts_03072024.h5ad")

# %%
# update the X to counts layer
adata_all.X = adata_all.layers['counts']

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
sc.pl.umap(adata_all, color=['groups', 'batch', 'annotation_int2'], ncols=1)

# %%
# save the adata_all
adata_all.write_h5ad("./anndata/HA_4MU_RegenP3_fullcounts_03072024.h5ad")

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
# save the subset 
adata_all_subset.write_h5ad("./anndata/HA_4MU_RegenP3_subset_03072024.h5ad")

# %%
adata_all_subset = adata_all_subset.copy()

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
model.save("models/HA_4MU_RegenP3_subset_03072024", overwrite = True)

# %%
# load the model
model = scvi.model.SCVI.load("models/HA_4MU_RegenP3_subset_03072024", adata = adata_all_subset)

# %%
# read the embeddings and save to the adata_all
SCVI_LATENT_KEY = "X_scVI"
adata_all_subset.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

# %%
if "X_diffmap" in adata_all_subset.obsm.keys():
    del adata_all_subset.obsm["X_diffmap"]
sc.pp.neighbors(adata_all_subset, use_rep=SCVI_LATENT_KEY,
                n_neighbors=30)
sc.tl.leiden(adata_all_subset, key_added="scvi_HAReg_leiden05", resolution=0.5)
sc.tl.leiden(adata_all_subset, key_added="scvi_HAReg_leiden04", resolution=0.4)
sc.tl.leiden(adata_all_subset, key_added="scvi_HAReg_leiden03", resolution=0.3)
sc.tl.leiden(adata_all_subset, key_added="scvi_HAReg_leiden02", resolution=0.2)

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
# create a new obs column based on batches 
adata_all_subset.obs['condition'] = adata_all_subset.obs['batch'].copy()
adata_all_subset.obs['condition'] = adata_all_subset.obs['condition'].replace({'4MU_1': '4MU_Treated', '4MU_2': '4MU_Treated', 'Control_1': '4MU_Control', 'Control_2': '4MU_Control', 'RB1': 'Regen', 'RB2': 'Regen', 'RJS': 'Regen'})

# %%
# update the adata_all_subset
adata_all_subset.write_h5ad("./anndata/HA_4MU_RegenP3_subset_03072024.h5ad")

# %%
# load the adata_all_subset
adata_all_subset = sc.read_h5ad("./anndata/HA_4MU_RegenP3_subset_03072024.h5ad")

# %% [markdown]
# # plot the clusters

# %%
sc.pl.umap(adata_all_subset, color=['batch','scvi_fib_leiden045', 'condition'], ncols=4, size=50)

# %%
# plot the umap 
sc.pl.umap(adata_all_subset, color=['scvi_HAReg_leiden05', 'scvi_HAReg_leiden04', 'scvi_HAReg_leiden03', 'scvi_HAReg_leiden02'], ncols=4)

# %%
# based on scvi_HAReg_leiden04, merge group 6, 1 and 2; merge 0 and 4
adata_all_subset.obs['scvi_HAReg_manual'] = adata_all_subset.obs['scvi_HAReg_leiden02'].replace({'3': '0'})

# %%
# plot the umap
sc.pl.umap(adata_all_subset, color=['scvi_HAReg_manual','scvi_fib_leiden045', 'condition','batch'], ncols=2, size=40) #save="_HA_4MU_RegenP3_subset_27062024.pdf")

# %%
adata_all_subset.uns['condition_colors']

# %%
GeneList = ['Arsi', 'Ltbp2', 'Msx2', 'Dlx5', 'Dlx6', 'Bmp5', 'Lhx9', 'Mest','Fbn2','Lrrc17'] # (From Mekayla Dev Cell) Lgr6 excluded as suggested by Mekayla
GeneList2 = ['Arsi', 'Ltbp2', 'Msx1', 'Msx2', 'Mest', 'Dlx5', 'Dlx6', 'Bmp5', 'Lhx9','Lgr6'] # new list from Byron 5th Nov 2025
# creage gene scoring from the GeneList
sc.tl.score_genes(adata_all_subset, GeneList, score_name='gene_score', use_raw=True)
# plot the gene score
# sc.pl.umap(adata_all_subset, color=['gene_score'], cmap=mymap, vmin=0, vmax=2, size=40)
# swap the order of the categories so that 4MU_Control comes first in condition
cond_order = ['4MU_Control', '4MU_Treated', 'Regen']
adata_all_subset.obs['condition'] = pd.Categorical(
    adata_all_subset.obs['condition'],
    categories=cond_order,
    ordered=True
)
# Reverse the colors to match the new order
adata_all_subset.uns['condition_colors'] = [ '#cc9900','#0a437a', '#c7c7c7']

# plot the gene score by condition
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
             'gene_score', groupby='condition', rotation=90, save="_HA_4MU_BlastemaMarker_subset_20250926.pdf")
# create a new .obs by combining the condition and scvi_HAReg_manual
adata_all_subset.obs['condition_manual'] = adata_all_subset.obs['condition'].astype(str) + '_'+ adata_all_subset.obs['scvi_HAReg_manual'].astype(str)
# make it categorical
adata_all_subset.obs['condition_manual'] = adata_all_subset.obs['condition_manual'].astype('category')


# plot the umap with the new condition_manual
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
             'gene_score', groupby='condition_manual', rotation=90)
# sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
             # GeneList, groupby='condition_manual', rotation=90) #, save="_HA_4MU_BlastemaMarker_conditionManual_subset_20250926.pdf")
sc.pl.dotplot(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
              GeneList, 
              groupby='condition_manual', 
              standard_scale='var', 
              cmap='bwr', dendrogram=False, swap_axes=True,)

# %%
# plot the violin plot for the genes
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
             GeneList, groupby='condition_manual', rotation=90, 
             size=1, stripplot=True, jitter=0.4, 
             )

# %%
GeneList2 = ['Arsi', 'Ltbp2', 'Msx1', 'Msx2', 'Mest', 'Dlx5', 'Dlx6', 'Bmp5', 'Lhx9','Lgr6'] # new list from Byron 5th Nov 2025
# plot the violin plot 
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
             GeneList2, groupby='condition_manual', rotation=90, save="_HA_4MU_BlastemaMarker_conditionManual_subset_20251105.pdf")
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
             GeneList2, groupby='condition', rotation=90, save="_HA_4MU_BlastemaMarker_condition_subset_20251105.pdf")

# %%
# plot the violin plot for the genes
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
             GeneList, groupby='condition', rotation=90, 
             size=1, stripplot=True, jitter=0.4, 
             )

# %%
# check the gene expressions of tenocyte markers
tenocyte_markers = ['Scx', 'Tnmd', 'Tnc', 'Col1a1', 'Col3a1', 'Thbs4', 'Dcn', 'Fmod', 'Egr1', 'Mkx']
# check if the genes are in the adata_all_subset.raw
tenocyte_markers_in = [gene for gene in tenocyte_markers if gene in adata_all_subset.raw.var_names]
tenocyte_markers_in

# %%
# use the gene scores to plot the tenocyte markers
sc.tl.score_genes(adata_all_subset, tenocyte_markers_in, score_name='tenocyte_score', use_raw=True)
# plot the tenocyte score
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], keys=['tenocyte_score'], groupby='condition', save="_HA_4MU_TenocyteScore_subset_20250926.pdf")
# plot the gene expression of the tenocyte markers
sc.pl.umap(adata_all_subset, color=tenocyte_markers_in, cmap=mymap, size=40, ncols=5,frameon = False, save="_HA_4MU_TenocyteMarkers_subset_20250926.pdf")

# use the dotplot to plot the tenocyte markers
sc.pl.dotplot(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
              tenocyte_markers,
                groupby='condition_manual',
                standard_scale='var',
                cmap='bwr', dendrogram=False, swap_axes=True,

)

# %%
Wnt_list = ['Axin2', 'Lgr5', 'Myc', 'Ccnd1', 'Tbx1','Tbx3', 'Cdx1', 'Mmp7', 'Wisp1', 'Wnt1', 'Wnt2', 'Wnt3', 'Wnt3a', 'Wnt8a']
# use the gene scores to plot the Wnt markers
sc.tl.score_genes(adata_all_subset, Wnt_list, score_name='Wnt_score', use_raw=True)
# plot the Wnt score
sc.pl.umap(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], color=['Wnt_score'], cmap=mymap, size=40)
# plot violin of the Wnt score by condition
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
             'Wnt_score', groupby='condition', rotation=90, save="_HA_4MU_WntScore_subset_20250926.pdf")
# plot the wnt score by condition_manual on umap
sc.pl.umap(adata_all_subset[adata_all_subset.obs['condition'] == '4MU_Treated'], color=['Wnt_score'], vmax=1, cmap=mymap, size=40)
sc.pl.umap(adata_all_subset[adata_all_subset.obs['condition'] == '4MU_Control'], color=['Wnt_score'], vmax=1, cmap=mymap, size=40)

# plot the fibro3 cluster
adata_all_subset_fibro3 = adata_all_subset[adata_all_subset.obs['scvi_HAReg_manual'] == '0'].copy()
sc.pl.violin(adata_all_subset_fibro3[adata_all_subset_fibro3.obs['condition'] != 'Regen'], 
             'Wnt_score', groupby='condition', rotation=90)

# %%
Canonical_wnt = ['Wnt1', 'Wnt2', 'Wnt3', 'Wnt3a', 'Wnt8a']
# calculate the gene scores for the canonical Wnt genes
sc.tl.score_genes(adata_all_subset, Canonical_wnt, score_name='Canonical_Wnt_score', use_raw=True)
# plot the canonical Wnt score
sc.pl.umap(adata_all_subset, color=['Canonical_Wnt_score'], cmap=mymap, size=40)
# plot the violin
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 
             'Canonical_Wnt_score', groupby='condition', rotation=90)
# plot individual genes
for gene in Canonical_wnt:
    sc.pl.umap(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], color=[gene], cmap=mymap, size=40)

# %%
nonCanonical_wnt = ['Wnt5a', 'Wnt5b', 'Wnt11', 'Wnt16']
# calculate the gene scores for the non canonical wnt genes
sc.tl.score_genes(adata_all_subset, nonCanonical_wnt, score_name='nonCanonical_Wnt_score', use_raw=True)
# plot the non canonical Wnt score
sc.pl.umap(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], color=['nonCanonical_Wnt_score'], cmap=mymap, size=40)
# plot the violin plot
sc.pl.violin(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], 'nonCanonical_Wnt_score', groupby='condition', rotation=90, save="_HA_4MU_nonCanonicalWnt_subset_20250926.pdf")
# plot individual genes
for gene in nonCanonical_wnt:
    sc.pl.umap(adata_all_subset[adata_all_subset.obs['condition'] != 'Regen'], color=[gene], cmap=mymap, size=40)


# %% [markdown]
# ### wilcoxon + correlation matrix for leiden0.2-0.4

# %%
# do the wilcoxon
sc.tl.rank_genes_groups(adata_all_subset, groupby='scvi_HAReg_manual', method='wilcoxon', n_genes=2000, key_added='wilcoxon_HAReg_manual')

# %%
# plot the wilcoxon
sc.pl.rank_genes_groups(adata_all_subset, key='wilcoxon_HAReg_manual', n_genes=25)

# %%
# create a subset for the 4MU
adata_4MU_subset = adata_all_subset[adata_all_subset.obs['condition'].isin(['4MU_Treated', '4MU_Control'])].copy()
adata_4MU_subset

# %%
import matplotlib.colors as mcolors
# take the tab10 colors
tab10 = plt.cm.tab10.colors
mytab10 = mcolors.LinearSegmentedColormap.from_list('my_colormap', tab10)
# Convert colors to hex format
hex10 = [mcolors.to_hex(color) for color in tab10]
hex10

# %%
# create color scheme for the plots 
UMAPcolor_manualCluster = ['#279e48', '#aa40fc','#8c564b']
UMAPcolor_3color = ['#cc9900', '#0a437a',  '#c7c7c7']
UMAPcolor_2color = ['#0a437a', '#cc9900']

# %%
# create a copy of the adata_all_subset
adata_all_subset_copy = adata_all_subset.copy()

# %%
# create a copy of the adata_all_subset
adata_all_subset = adata_all_subset_copy.copy()

# %%
adata_all_subset_copy.obs

# %%
# reorganise the regen samples to the front of the index 
Regen_index = adata_all_subset.obs['condition'] == 'Regen'
adata_all_subset = adata_all_subset[Regen_index].concatenate(adata_all_subset[~Regen_index])
adata_all_subset.obs

# %%
# plot the umap
sc.pl.umap(adata_all_subset, color=['scvi_HAReg_manual','scvi_fib_leiden045', 'condition','batch'], ncols=2, size=40,) # save="_HA_4MU_subset_27062024.pdf")
sc.pl.umap(adata_all_subset, color='scvi_HAReg_manual', palette=UMAPcolor_manualCluster, size=40, save = '_HA_4MU_subset_clus_27062024.pdf')
sc.pl.umap(adata_all_subset, color='condition', palette=UMAPcolor_3color, size=40, save = '_HA_4MU_subset_cond_27062024.pdf')

# %%
# do the wilcoxon on only the 4MU clusters
sc.tl.rank_genes_groups(adata_4MU_subset, groupby='scvi_HAReg_manual', method='wilcoxon', n_genes=2000, key_added='wilcoxon_4MU_manual')

# %%
# plot the wilcoxon # check if the genes have similar wilcoxon marker genes
sc.pl.rank_genes_groups(adata_4MU_subset, key='wilcoxon_4MU_manual', n_genes=25, save = "_HA_4MU_fib_perfectClust_wilcoxon.pdf")

# %%
# create a dot plot with manual defined genes for each cluster 
# create a list of genes
genelist = {'0': ['Msx1', 'Mdk','Cxcl14'],
            '1': ['Ibsp', 'Spp1', 'Alpl'],
            '2': ['Birc5', 'Stmn1', 'Mki67']
            }
sc.pl.dotplot(adata_4MU_subset, genelist, groupby='scvi_HAReg_manual',standard_scale='var', color_map="Blues", save = "_HA_4MU_fib_perfectClust.pdf")

# %%
# save the adata_all_subset
adata_all_subset.write_h5ad("./anndata/HA_4MU_RegenP3_subset_03072024.h5ad")

# %%
# save the adata_4MU_subset
adata_4MU_subset.write_h5ad("./anndata/HA_4MU_subset_03072024.h5ad")

# %%
# load the adata_4MU_subset
adata_4MU_subset = sc.read_h5ad("./anndata/HA_4MU_subset_03072024.h5ad")

# %%
# create bar plots for the proportion of cells in each condition
all_df = pd.DataFrame(adata_4MU_subset.obs.groupby(['batch','scvi_HAReg_manual']).size(), columns = ['count'])
# add an extra column of percentage
all_df['percentage'] = all_df.groupby(level=0, group_keys=False).apply(lambda x: 100 * x / float(x.sum()))
all_df

# %%
# save the all_df to a csv file
all_df.to_csv("./HA_4MU_fib_Clust_proportion_counts_03072024.csv")

# %%
# plot a bar chart of the percentage of cells in each cluster
ax = all_df['percentage'].unstack().plot(kind='barh', stacked=True, figsize=(10,5),
                      title='Percentage of cell type in batch', 
                      ylabel='Batch', xlabel='Percentage', legend=False,
                      )
labels = adata_all_subset.obs['scvi_HAReg_manual'].cat.categories
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.savefig('figures/Uninj_fib_percentage_batch.pdf', bbox_inches='tight')

# %%
count_df = all_df['count'].unstack()

# --- Plot stacked bar chart
ax = count_df.plot(
    kind='barh', stacked=True, figsize=(10,5),
    title='Cell counts per batch',
    xlabel='Number of cells', ylabel='Batch',
    legend=False
)

# Add legend
labels = adata_all_subset.obs['scvi_HAReg_manual'].cat.categories
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig(
    'figures/CellTypeProportion_4MU_fib_count_batch_20251008.pdf',
    bbox_inches='tight'
)
plt.show()

# %%
# calculate the average percentage for each level
Treat_avg = all_df.loc[pd.IndexSlice[['4MU_1','4MU_2'],:], 'percentage'].groupby("scvi_HAReg_manual", observed = False).mean()
Ctrl_avg = all_df.loc[pd.IndexSlice[['Control_1','Control_2'],:], 'percentage'].groupby("scvi_HAReg_manual", observed = False).mean()

# %%
d = {'Treated 4MU': Treat_avg, 'Control': Ctrl_avg}
clus_avg = pd.DataFrame(data=d)
clus_avg

# %%
# plot a horizontal bar plot for clus_avg
ax = clus_avg.T.plot(kind='barh', stacked=True, figsize=(10,5),
                     title='Percentage of cell types in 4MU and Control', 
                      ylabel='Location', xlabel='Percentage', legend=False,
                      )
labels = adata_all_subset.obs['scvi_HAReg_manual'].cat.categories
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

# %%
# prepare information to feed into propeller 
clust = adata_all_subset.obs['scvi_HAReg_manual'].values
biorep = adata_all_subset.obs['batch'].values
grp = adata_all_subset.obs['condition'].values

# %%
%%R -i grp -i clust -i biorep -o propeller_res
library(speckle)
library(ggplot2)

propeller_res = propeller(clusters = clust, sample = biorep, group = grp,
  robust = FALSE, trend = FALSE, transform="asin")

# %%
# create a new row inidicating whether FDR < 0.05
propeller_res["FDR_TF"] = ["True" if x < 0.05 else "False" for x in propeller_res["FDR"]]
propeller_res

# %%
# take the last 3 column
propeller_res[["P.Value", "FDR", "FDR_TF"]]

# %%
# plot a horizontal bar plot for clus_avg
ax = clus_avg.T.plot(kind='barh', stacked=True, figsize=(10,5),
                     title='Percentage of cell types in 4MU and Control', 
                      ylabel='Location', xlabel='Percentage', legend=False,
                      )
labels = adata_all_subset.obs['scvi_HAReg_manual'].cat.categories
labels = [f'{cat} #' if propeller_res["FDR_TF"][cat]=="True" else cat for cat in labels]
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.show()
# save the plot
plt.savefig('figures/HA-4MU-fibro_percentage.pdf', bbox_inches='tight')

# %%
# calculate the probability of the cells in each cluster from each subcluster. 
# calculate the sum across the rows 
clus_sum = clus_avg.sum(axis = 1)

# %%
# apply Baysian statistics to 
clus_avg['Treated 4MU_Perct'] = (clus_avg['Treated 4MU'] / clus_sum ) * 100
clus_avg['Control_Perct'] = (clus_avg['Control'] / clus_sum) * 100
clus_avg

# %%
# create a copy of the dataframe and remove the first two columns 
clus_avg_cp  = clus_avg[['Treated 4MU_Perct', 'Control_Perct']].copy()
clus_avg_cp

# %%
# plot the bar plots 
# plot a horizontal bar plot for clus_avg
ax = clus_avg_cp.plot(kind='bar', stacked=True, figsize=(10,5),
                     title='Percentage of cell types in 4MU and Control', 
                      ylabel='Location', xlabel='Percentage', legend=False,
                      )
labels = ['4MU', 'Control']
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.show()
# save the plot
plt.savefig('figures/HA-4MU-Fib_percentage_cond.pdf', bbox_inches='tight')

# %%
# plot the gene expression on umap and violin plot 
key = 'Mdk'
sc.pl.umap(adata_4MU_subset, color=[key], cmap=mymap)
sc.pl.violin(adata_4MU_subset, keys=key, groupby='scvi_HAReg_manual', rotation=90)

# %%
# plot a heatmap from rank_genes_groups
sc.pl.rank_genes_groups_heatmap(adata_4MU_subset, key='wilcoxon_4MU_manual', n_genes=10, standard_scale='var', cmap='RdBu_r')

# %%
# figure out all the indexes from cluster 3 of scvi_HAReg_manual
index_rm = adata_4MU_subset.obs[adata_4MU_subset.obs['scvi_HAReg_manual'] == '3'].index
index_rm

# %%
# save the index_rm to a csv file
pd.DataFrame(index_rm).to_csv("./anndata/HA_4MU_fib_manualCluster3_remove_27062024.csv")


