# %% [markdown]
# # Regen - NonRegen analysis

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
from matplotlib import colors
#Define a nice colour map for gene expression
colors2 = plt.cm.Reds(np.linspace(0, 1, 128))
colors3 = plt.cm.Greys_r(np.linspace(0.7,0.8,40))
colorsComb = np.vstack([colors3, colors2])
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', colorsComb)

# %%
adata_all = sc.read_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/R_NR_14DPA_25032024.h5ad")
adata_all

# %%
adata_fibroblast = sc.read_h5ad("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/anndata/R_NR_14DPA_fibroblast.h5ad")
adata_fibroblast

# %%
adata_all.raw.to_adata()

# %%
# define the active pca and umap 
adata_all.obsm['X_pca'] = adata_all.obsm['X_scVI']
adata_all.obsm['X_umap'] = adata_all.obsm['X_scVI_MDE']

# %% [markdown]
# ## invesitgate the splicing factors

# %%
# plot of the umap 
sc.pl.umap(adata_fibroblast, color = ['annotation_int2'], legend_loc = "on data", frameon=False, )
sc.pl.umap(adata_fibroblast, color = ['Amp_location'],  legend_loc = "on data", frameon=False, )

# %%
key = ["Esrp1", "Esrp2", "Rbfox2"]
sc.pl.umap(adata_all, color = key, use_raw=True, color_map=mymap, size= 40)
sc.pl.violin(adata_all, key, groupby='Amp_location', rotation=90, stripplot=True, multi_panel=True)
sc.pl.violin(adata_all, key, groupby='annotation_pre3', rotation=90, stripplot=True, multi_panel=True)


# %%
key = ["Cdh1", "Ctnnb1"]
sc.pl.umap(adata_all, color = key, use_raw=True, color_map=mymap, size= 40)
sc.pl.violin(adata_all, key, groupby='Amp_location', rotation=90, stripplot=True, multi_panel=True)

# %%
adata_all.obs['annotation_int2_Amp_location'] = adata_all.obs['annotation_int2'].astype(str) + "_" + adata_all.obs['Amp_location'].astype(str)
adata_all.obs['annotation_int2_Amp_location'] = adata_all.obs['annotation_int2_Amp_location'].astype('category')

# %%
key = ["Fgf8", "Fgf4", "Fgf9", "Fgf17", "Shh", "Ihh", "Nfatc1", "Pdk1", "Camk2a", "Camk2b","Camk2d","Camk2g", "Prkca", "Prkcb", "Prkcg", "Hif1a"]
sc.pl.umap(adata_all, color = key, use_raw=True, color_map=mymap, size= 40)
sc.pl.violin(adata_all, key, groupby='Amp_location', rotation=90, stripplot=True, multi_panel=True)
sc.pl.violin(adata_all, key, groupby='annotation_int2_Amp_location', rotation=90, stripplot=True, multi_panel=True)

# %%
key = ["Fgf8", "Fgf9", "Ctsk", "Sufu", "Ihh", "Ostf1", "Pth1r", "Gli3"]
sc.pl.umap(adata_fibroblast, color = key, use_raw=True, color_map=mymap, size= 40)
sc.pl.violin(adata_fibroblast, key, groupby='Amp_location', rotation=90, stripplot=True, multi_panel=True)
sc.pl.violin(adata_fibroblast, key, groupby='scvi_fib_leiden07', rotation=90, stripplot=True, multi_panel=True)

# %%
key = ["Lum"]
sc.pl.umap(adata_fibroblast, color = key, use_raw=True, color_map=mymap, size= 40)
sc.pl.violin(adata_fibroblast, key, groupby='Amp_location', rotation=90, stripplot=True, multi_panel=True)
sc.pl.violin(adata_fibroblast, key, groupby='scvi_fib_leiden045', rotation=90, stripplot=True, multi_panel=True)

# %%
key = ["Hand2","Wnt1", "Wnt2", "Wnt2b", "Wnt3a", "Wnt4", "Wnt5a", "Wnt5b", "Wnt6", "Wnt7a", "Wnt7b","Wnt8a", "Wnt8b",  "Wnt9a", "Wnt9b", "Wnt10a", "Wnt10b",  "Wnt11", "Wnt11", "Wnt16",]
sc.pl.umap(adata_fibroblast, color = key, use_raw=True, color_map=mymap, size= 40)
sc.pl.violin(adata_fibroblast, key, groupby='Amp_location', rotation=90, stripplot=True, multi_panel=True)
sc.pl.violin(adata_fibroblast, key, groupby='scvi_fib_leiden045', rotation=90, stripplot=True, multi_panel=True)

# %% [markdown]
# ## plots for publication

# %%
# swap the category groups in scvi_fib_leiden045, 0 to 1 and 1 to 0
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].astype(str)
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].replace({'0':'1', '1':'0'})
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].astype('category')

# %%
sc.tl.rank_genes_groups(adata_fibroblast,groupby='scvi_fib_leiden045',
                        use_raw=True,
                        method='wilcoxon', 
                        corr_method='benjamini-hochberg')

# %%
gene_list = {
    '0': ['Cd34','Tnxb','Clec3b'],
    '1': ['Sulf2','Thbs4','Cthrc1'],
    '2': ['Msx1','Mdk','Cxcl14'],
    '3': ['Runx2','Comp','Fmod'],
    '4': ['Spp1','Alpl','Ibsp'],
    '5': ['Stmn1','Birc5','Mki67'],
    '6': ['Slit2','Lama2','Thy1'],
    '7': ['S100a4','Timp1','Col6a3']
}

# %%
sc.pl.dotplot(adata_fibroblast, gene_list, groupby='scvi_fib_leiden045', dendrogram=False, save='regNonreg_fib_finalizedGenes2.pdf', cmap = 'Blues', standard_scale = 'var')

# %%
# create a new clustering by combining scvi_fib_leiden045 and Amp_location
adata_fibroblast.obs['clus045_amp'] = adata_fibroblast.obs['scvi_fib_leiden045'].astype(str) + "_" + adata_fibroblast.obs['Amp_location'].astype(str)

# %%
sc.tl.rank_genes_groups(adata_fibroblast,groupby='clus045_amp',
                        method='wilcoxon', 
                        corr_method='benjamini-hochberg')

# %%
sc.pl.rank_genes_groups(adata_fibroblast, groupby='scvi_fib_leiden045', save = '_regNonreg_fib_p2p3_comp_scores.pdf')

# %%
fib_df = pd.DataFrame(adata_fibroblast.obs.groupby(['batch','scvi_fib_leiden045']).size(), columns = ['count'])
# add an extra column of percentage
fib_df['percentage'] = fib_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
fib_df
# show the RJS 
fib_df.xs('NRB1')

# %%
# export an excel file for fib_df
fib_df.to_excel('figures/fibroblast_cluster_distribution.xlsx')

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
Rclus_avg = fib_df.loc[pd.IndexSlice[['RB1','RB2', 'RJS'],:], 'percentage'].groupby("scvi_fib_leiden045").mean()
NRclus_avg = fib_df.loc[pd.IndexSlice[['NRB1','NRB2', 'NRB3'],:], 'percentage'].groupby("scvi_fib_leiden045").mean()

# %%
# create a dataframe of the average percentages 
d = {'Regen': Rclus_avg, 'NonRegen': NRclus_avg}
clus_avg = pd.DataFrame(data=d)
clus_avg

# %%
# save an excel for clus_avg
clus_avg.to_excel('figures/fibroblast_cluster_distribution_avg.xlsx')

# %%
# plot a horizontal bar plot for clus_avg
ax = clus_avg.T.plot(kind='barh', stacked=True, figsize=(10,5),
                     title='Percentage of cells in each cluster of PDGFRa+ cells', 
                      ylabel='Batch', xlabel='Percentage', legend=False,
                      )
labels = adata_fibroblast.obs['scvi_fib_leiden045'].cat.categories
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

# %% [markdown]
# ## Propeller

# %%
# prepare information to feed into propeller 
clust = adata_fibroblast.obs['scvi_fib_leiden045'].values
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
clus_avg.iloc[:,0:2]

# %%
# plot the percentage bar plot again
clus_avg = clus_avg.iloc[:,0:2]
ax = clus_avg.T.plot(kind='barh', stacked=True, figsize=(10,5),
                     title='Percentage of cells in each cluster of Pdgfra+ cells', 
                      ylabel='Amputation condition', xlabel='Percentage', legend=False,
                      )
labels = adata_fibroblast.obs['scvi_fib_leiden045'].cat.categories
labels = [f'{cat} #' if propeller_res["FDR_TF"][cat]=="True" else cat for cat in labels]
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig('figures/fib_percentage_AmpLoc.pdf', bbox_inches='tight')

# %%
# reverse the order of the cells
adata_fibroblast = adata_fibroblast[::-1]

# %%
# plot the umap of clusters in scvi_fib_leiden045
sc.pl.umap(adata_fibroblast, color = 'batch', frameon=False, size = 40, save = 'fib_batch_umap.pdf')

# %%
# plot the umap of clusters in scvi_fib_leiden045
sc.pl.umap(adata_fibroblast, color = 'scvi_fib_leiden045', legend_loc = "on data", frameon=False, size = 40, save = 'fib_clus_umap.pdf')

# %%
# create a dictionary for the palette
palette_AmpLoc = {"P2": "#0a437a", "P3": "#CC9900"}

# %%
# plot the umap of clusters in scvi_fib_leiden045
sc.pl.umap(adata_fibroblast[::-1], color = 'Amp_location', frameon=False, size = 40,
           palette = palette_AmpLoc,  save = '_fib_AmpLoc_umap.pdf')

# %%
# Trial
key = "Thbs4"
sc.pl.umap(adata_fibroblast, color=[key], use_raw=True, color_map=mymap, size= 40, save= f"_fib_{key}.pdf")
# sc.pl.umap(adata_all, color=[key], color_map=mymap, size= 40, layer='X_scVI_normalized')
# adata_all.X = adata_all.layers['X_scVI_normalized']
sc.pl.violin(adata_fibroblast, [key], groupby='scvi_fib_leiden045', rotation=90, stripplot=True, multi_panel=True, save = f"_fib_{key}.pdf")
# sc.pl.violin(adata_all, [key], groupby='Amp_location', rotation=90, stripplot=True, multi_panel=True, layer='X_scVI_normalized', use_raw=False)

# %%
key = ["Arsi", "Mest", "Lrrc15", "Fbn2"]
sc.pl.umap(adata_fibroblast, color=key, use_raw=True, color_map=mymap, size= 40 )
sc.pl.violin(adata_fibroblast, key, groupby='scvi_fib_leiden045', rotation=90, stripplot=True, multi_panel=True,)

# %%
key = ["Msx2", "Dlx5", "Bmp5"]
sc.pl.umap(adata_fibroblast, color=key, use_raw=True, color_map=mymap, size= 40 )
sc.pl.violin(adata_fibroblast, key, groupby='scvi_fib_leiden045', rotation=90, stripplot=True, multi_panel=True)

# %%
# umap for the clustering fibroblast
sc.pl.umap(adata_fibroblast, color=['batch','Amp_location','scvi_fib_leiden045'], size = 40, save = '_fib_batch-Amploc-clus.pdf')

# %%
# create new clustering of .obs by combining scvi_fib_leiden045 and amp_locaiton
adata_fibroblast.obs['AmpLoc_scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].astype(str) + "_" + adata_fibroblast.obs['Amp_location'].astype(str)

# %%
key = ["Sdc4","Sulf2", "Thbs4", "Thbs1", "Thbs2"]
sc.pl.umap(adata_fibroblast, color=key, use_raw=True, color_map=mymap, size= 40 )
sc.pl.violin(adata_fibroblast, key, groupby='AmpLoc_scvi_fib_leiden045', rotation=90, stripplot=True, multi_panel=True)

# %%
# create the rank_genes_groups for all the cell types in annotation_int2
sc.tl.rank_genes_groups(adata_all, groupby='annotation_int2',
                        use_raw=True,
                        method='wilcoxon', 
                        corr_method='benjamini-hochberg')

# %%
# get the genes
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
Markers = Markers[Markers['lfc']>0]

# %%
%%R -i Markers 
# Perform pathway analysis on the top 20 genes 
library("clusterProfiler")
library("org.Mm.eg.db")
library("dplyr")
library("ggplot2")

# take the top 20 genes from each group
Markers_top <- Markers %>% group_by(group) %>% top_n(n = 10, wt = scores) 
# print(Markers_top)
# add extra column of entrez id
Markers_top$entrez <- mapIds(org.Mm.eg.db, Markers_top$Gene_symb, "ENTREZID", "SYMBOL", multiVals = "first")
# remove the lines that could not be mapped
Markers_top <- Markers_top[!is.na(Markers_top$entrez),]
# create a list of genes for each cluster
Markers_top <- split(Markers_top$entrez, Markers_top$group)
# print(Markers_top)

# compareCluster
# One of "groupGO", "enrichGO", "enrichKEGG", "enrichDO" or "enrichPathway"
clustersummary <- compareCluster(geneClusters = Markers_top, fun = "enrichGO", OrgDb = org.Mm.eg.db, ont = "BP", pvalueCutoff = 0.05, pAdjustMethod = "BH", qvalueCutoff = 0.05, readable = TRUE)
print(clustersummary)
pdf("C:/Users/jjyw2/OneDrive - University of Cambridge/Documents/Digit Regeneration/Byron For Jo/temp.pdf", 
    width = 5, height = 10)
title <- "GO analysis of the top 10 upregulated genes in each group"
p1 <- dotplot(clustersummary, showCategory = 3) +
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


