# %% [markdown]
# # Uninjured P2 P3 and Regen Non Regen 14DPA DEG

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
import seaborn as sns
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

# %%
# load full counts anndata object
adata_all = ad.read_h5ad("./anndata/Uninj_RNR_fullcounts_25052024.h5ad")

# %%
# load the adata_all_subset
adata_all_subset = sc.read_h5ad("./anndata/Uninj_RNR_HvgSubset_30052024.h5ad")
adata_all_subset


# %%
# Annotation
sc.pl.umap(adata_all_subset, color=["Amp_location","batch","annotation_int2", "Condition_R_NR_Unj"], color_map=mymap, size= 10)

# %%
adata_all_subset.obs['batch'] = adata_all_subset.obs['batch'].replace({
    'NRB1': 'Storer et al. NonRegeneration Dataset 1',
    'NRB2': 'Mui et al. NonRegeneration Dataset 1',
    'NRB3': 'Mui et al. NonRegeneration Dataset 2',
    'RB1': 'Storer et al. Regeneration Dataset 1',
    'RB2': 'Storer et al. Regeneration Dataset 2',
    'RJS': 'Johnson et al. Regeneration Dataset',
    'B_P2X_2': 'Mui et al. Uninjured P2 Dataset 2',
    'B_P2X_3': 'Mui et al. Uninjured P2 Dataset 3',
    'B_P2_1': 'Mui et al. Uninjured P2 Dataset 1',
    'B_P3X_1': 'Mui et al. Uninjured P3 Dataset 1',
    'B_P3X_2': 'Mui et al. Uninjured P3 Dataset 2',
    'JS_P3_1': 'Johnson et al. Uninjured P3 Dataset',
    'MS_P3_1': 'Storer et al. Uninjured P3 Dataset 1',
    'MS_P3_2': 'Storer et al. Uninjured P3 Dataset 2'
    })
# rank them based on the order above
batch_order = ['Mui et al. NonRegeneration Dataset 1', 'Mui et al. NonRegeneration Dataset 2', 'Storer et al. NonRegeneration Dataset 1',
                'Storer et al. Regeneration Dataset 1', 'Storer et al. Regeneration Dataset 2', 'Johnson et al. Regeneration Dataset',
                'Mui et al. Uninjured P2 Dataset 1', 'Mui et al. Uninjured P2 Dataset 2', 'Mui et al. Uninjured P2 Dataset 3', 
                'Mui et al. Uninjured P3 Dataset 1', 'Mui et al. Uninjured P3 Dataset 2', 'Johnson et al. Uninjured P3 Dataset',
               'Storer et al. Uninjured P3 Dataset 1', 'Storer et al. Uninjured P3 Dataset 2']
adata_all_subset.obs['batch'] = adata_all_subset.obs['batch'].astype("category")

adata_all_subset.obs['batch'] = adata_all_subset.obs['batch'].cat.reorder_categories(batch_order, ordered=True)

# %%
# plot the batches
sc.pl.umap(adata_all_subset, color=["batch"], color_map=mymap, size= 10)

# %%
# export the plots as pdf
sc.pl.umap(adata_all_subset, color=["Amp_location","batch","annotation_int2", "Condition_R_NR_Unj", "Pdgfra"], ncols=1, use_raw=True, frameon=False, color_map=mymap, size= 10, save="_RNR14_Uninj_metadata_20250926.pdf")


# %%
adata_all_subset.obs['batch'].value_counts()

# %%
# plot the pdgfra positive cells with the umap 
sc.pl.umap(adata_all_subset, color=["Pdgfra", "scvi_leiden05"], use_raw=True, color_map=mymap, size= 40, legend_loc="on data")
sc.pl.umap(adata_all_subset, color=["annotation_int2"], color_map=mymap, size= 40)
# plot violin plots of Pdgfra
sc.pl.violin(adata_all_subset, keys=["Pdgfra"], groupby="scvi_leiden05", use_raw=True)


# %%
# create a heatmap for the marker genes 
# Fibroblast, Pericyte/SMC, Schwann, Immune, Keratinocyte, Endothelial, Lymphatic Endothelial, Chondrocyte
marker_genes_plHeat = ['Pdgfra','Col1a1','Lum',  # Fibroblast 
                        'Sox10','Plp1','Cdh19', # Schwann
                        'Ptprc','Lyz2','Cd53', # Immune
                        'Rgs5','Acta2','Myh11', # Pericyte/SMC
                        'Krt5','Krt14','Cdh1', # Keratinocyte
                        'Cdh5','Pecam1','Eng', # Endothelial
                        'Prox1','Flt4','Lyve1', # Lymphatic Endothelial
                        'Acan','Col2a1','Sox9', # Chondrocyte
                        ]
# create a new copy of the adata 
adata_all_subset_copy = adata_all_subset.raw.to_adata().copy()
# subset the genes based on the gene list
adata_all_subset_copy = adata_all_subset_copy[:, marker_genes_plHeat]
# change the layer to float32
adata_all_subset_copy.X = adata_all_subset_copy.X.astype(np.float32)
# scale and store results in layer
# Scale in-place
sc.pp.scale(adata_all_subset_copy)

# Now the scaled data is in adata_all_subset_copy.X
adata_all_subset_copy.layers["scaled"] = adata_all_subset_copy.X.copy()

adata_all_subset_copy.obs['annotation_int2'] = adata_all_subset_copy.obs['annotation_int2'].cat.reorder_categories(['Fibroblast','Schwann','Immune','Pericyte/SMC','Keratinocyte','Endothelial','Lymphatic Endothelial', 'Chondrocyte'])
sc.pl.heatmap(adata_all_subset_copy, marker_genes_plHeat, groupby='annotation_int2', 
                swap_axes=False,
                layer = "scaled",
                vmin=-2,
                vmax=2,
                cmap="coolwarm", 
                save="_RNR14_Uninj_markerGenes_20251003.pdf"
                )
del adata_all_subset_copy

# %% [markdown]
# ### Matrisome scoring

# %%
# read the Mm_Matrisome_MGI.xlsx file 
matrisome = pd.read_excel("Mm_Matrisome_Masterlist_Naba.xlsx")
# select only the division of core matrisome
core_matrisome = matrisome[matrisome['Division'] == 'Core matrisome']
# subset the Collagens and proteoglycans
core_matrisome = core_matrisome[core_matrisome['Category'].isin(['Collagens', 'Proteoglycans'])]
# # subset the proteoglycans
# core_matrisome = core_matrisome[core_matrisome['Category'] == 'Proteoglycans']
# take the Symbol column and convert it to a list
matrisome_genes = core_matrisome['Gene Symbol'].tolist()
# export this list as a csv file
matrisome_genes_df = pd.DataFrame(matrisome_genes, columns=['Gene Symbol'])
matrisome_genes_df.to_csv("Core_Matrisome_Collagens_Proteoglycans_GeneList_20251003.csv", index=False)
matrisome_genes

# %%
# subset the fibroblast
adata_fibroblast = adata_all_subset[adata_all_subset.obs['annotation_int2'] == 'Fibroblast'].copy()
# do scoring for the matrisome genes
sc.tl.score_genes(adata_fibroblast, matrisome_genes, score_name='matrisome_score', use_raw=True)

# %%
# plot the matrisome scoring on the umap
sc.pl.umap(adata_fibroblast, color='matrisome_score', cmap='RdBu_r', size=20,) # save='_RNR14DPA_fibroblast_matrisome_score.pdf')
# update the category sequence
adata_fibroblast.obs['Condition_R_NR_Unj'] = adata_fibroblast.obs['Condition_R_NR_Unj'].cat.reorder_categories(['Uninjured_P2','Uninjured_P3','NonRegen','Regen'])
# export the violin plot as pdf
sc.pl.violin(adata_fibroblast, keys='matrisome_score', groupby='Condition_R_NR_Unj', rotation=90, save='_RNR14DPA_fibroblast_matrisome_score_violin_20250926.pdf')

# %%
sc.pl.correlation_matrix(adata_fibroblast, groupby="Condition_R_NR_Unj",show_correlation_numbers = True, figsize=(6, 6), save="_RNR14DPA_fibroblast_matrisome_score_correlation_20250926.pdf")

# %%
# proportional analysis on the fibroblast dataset
# create bar plots for the proportion of cells in each condition
all_df = pd.DataFrame(adata_fibroblast.obs.groupby(['batch','scvi_fib_leiden06']).size(), columns = ['count'])
# add an extra column of percentage
all_df['percentage'] = all_df.groupby(level=0, group_keys=False).apply(lambda x: 100 * x / float(x.sum()))
all_df

# %%
# Pivot for plotting
count_df = all_df['count'].unstack()

# --- Plot stacked bar chart
ax = count_df.plot(
    kind='barh', stacked=True, figsize=(10,5),
    title='Cell counts per batch',
    xlabel='Number of cells', ylabel='Batch',
    legend=False
)

# Add legend
labels = adata_fibroblast.obs['scvi_fib_leiden06'].cat.categories
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig(
    'figures/CellTypeProportion_RNR14_Uninj_fib_count_batch_20250926.pdf',
    bbox_inches='tight'
)
plt.show()

# %%
# export the df as a csv file
all_df.to_csv("RNR14_Uninj_fib_Leiden06_CellTypeProportion_20250926.csv")


