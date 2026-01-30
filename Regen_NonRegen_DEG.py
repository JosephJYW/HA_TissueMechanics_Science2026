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
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import numpy as np
import random
import sc_toolbox
import scvi
import pertpy  # must import this after loading the R environment # upgraded the setuptools from 59.5.0 to 69.1.0 to make installation work
sc.settings.verbosity = 0
sc.settings._vector_friendly = False

# %%
# Load the dataset of Integrated Regen Nonregen 14DPA
# load the one with full counts 
adata_all = sc.read_h5ad("./anndata/R_NR_14DPA_fullCounts_25032024.h5ad")
# load the one with subsets 
adata_all_subset = sc.read_h5ad("./anndata/R_NR_14DPA_25032024.h5ad")

# %%
# load the fibroblast dataset of Uninjured P2/P3
# load the one with subsets
adata_fibroblast_subset = sc.read_h5ad("./anndata/R_NR_14DPA_fibroblast.h5ad")

# %%
# remove the cells that does not exist in adata_all_subset
adata_all = adata_all[adata_all.obs_names.isin(adata_all_subset.obs_names)]

# %%
# transfer information from adata_all_subset to adata_all
adata_all.obs = adata_all_subset.obs
adata_all.obsm = adata_all_subset.obsm
adata_all.uns = adata_all_subset.uns

# %% [markdown]
# # MAST

# %%
# ensure the datasets are with raw counts 
adata_all.X = adata_all.layers["counts"].copy()

# %%
# Normalize adata with the size factors calcuated
adata_all.X /= adata_all.obs['size_factors'].values[:,None]
sc.pp.log1p(adata_all)

# %%
# a function to take care of object type conversion between R and python 
# additionally filter genes that has less than 3 cells expressed 
def prep_anndata(adata_):
    def fix_dtypes(adata_):
        df = pd.DataFrame(adata_.X.A, index=adata_.obs_names, columns=adata_.var_names)
        df = df.join(adata_.obs)
        return sc.AnnData(df[adata_.var_names], obs=df.drop(columns=adata_.var_names))

    adata_ = fix_dtypes(adata_)
    sc.pp.filter_genes(adata_, min_cells=3)
    return adata_

# %%
# remove the cells that does not exist in adata_all_subset
adata_fibroblast = adata_all[adata_all.obs_names.isin(adata_fibroblast_subset.obs_names)]

# %%
# transfer information from adata_all_subset to adata_all
adata_fibroblast.obs = adata_fibroblast_subset.obs
adata_fibroblast.obsm = adata_fibroblast_subset.obsm
adata_fibroblast.uns = adata_fibroblast_subset.uns

# %%
# swap group 0 and 1 for plotting purposes
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].astype(str)
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].replace({'0':'1', '1':'0'})
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].astype('category')

# %%
# calculate the rank gene scores for the fibroblast dataset
sc.tl.rank_genes_groups(adata_fibroblast, groupby="scvi_fib_leiden045", method="wilcoxon")
sc.pl.rank_genes_groups(adata_fibroblast, n_genes=25, sharey=False)

# %%
# quick inspection of the clusters in the fibroblast
sc.pl.umap(adata_fibroblast, color="scvi_fib_leiden045")

# %%
# rename the categories with the expected names 
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].replace({'0':'P2-Firbroblast-1',
                                                                                                 '1':'P2-Firbroblast-2', 
                                                                                                 '2':'Blastema-Cells',
                                                                                                 '3':'OsteoChondro-cells',
                                                                                                 '4':'Osteo-cells',
                                                                                                 '5':'Cycling-cells',
                                                                                                 '6':'BMSCs',
                                                                                                 '7':'Fibroblast-3',
                                                                                                 })

# %%
# rmeove all cells that are P2 but not in P2-Firbroblast-1 and P2-Firbroblast-2 
adata_fibroblast = adata_fibroblast[~((adata_fibroblast.obs['Amp_location'] == 'P2') & ~(adata_fibroblast.obs['scvi_fib_leiden045'].isin(['P2-Firbroblast-1', 'P2-Firbroblast-2'])))]


# %%
# rmeove all cells that are P3 and in P2-Firbroblast-1 and P2-Firbroblast-2 
adata_fibroblast = adata_fibroblast[~((adata_fibroblast.obs['Amp_location'] == 'P3') & (adata_fibroblast.obs['scvi_fib_leiden045'].isin(['P2-Firbroblast-1', 'P2-Firbroblast-2'])))]


# %%
# rmeove all cells that are not in osteo lineago, nor in P2-Firbroblast-1 or P2-Firbroblast-2
adata_fibroblast = adata_fibroblast[~((adata_fibroblast.obs['scvi_fib_leiden045'].isin(['Blastema-Cells', 'OsteoChondro-cells', 'Cycling-cells', 'BMSCs', 'Fibroblast-3'])))]

# %%
adata_fibroblast_cp = adata_fibroblast.copy()
adata_fibroblast.obs['scvi_fib_leiden045'].value_counts()

# %%
# remove cells in P2-Firbroblast-2
adata_fibroblast_1 = adata_fibroblast[~(adata_fibroblast.obs['scvi_fib_leiden045'] == 'P2-Firbroblast-2')].copy()
# remove cells in P2-Firbroblast-1
adata_fibroblast_2 = adata_fibroblast[~(adata_fibroblast.obs['scvi_fib_leiden045'] == 'P2-Firbroblast-1')].copy()

# %%
adata_fibroblast = adata_fibroblast_2.copy()

# %%
# rename the categories with the expected names
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].astype(str)
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].replace({'P2-Firbroblast-1':'Others',
                                                                                                 'P2-Firbroblast-2':'Others', 
                                                                                                 'Blastema-Cells':'Others',
                                                                                                 'OsteoChondro-cells':'Others',
                                                                                                #  'Osteo-cells':'Others',
                                                                                                 'Cycling-cells':'Others',
                                                                                                 'BMSCs':'Others',
                                                                                                 'Fibroblast-3':'Others',})
adata_fibroblast.obs['scvi_fib_leiden045'] = adata_fibroblast.obs['scvi_fib_leiden045'].astype('category')

# %%
# rename the annotation_int2 to cell_type
adata_fibroblast.obs['cell_type'] = adata_fibroblast.obs['annotation_int2']
# rename the scvi_fib_leiden045 to label
adata_fibroblast.obs['label'] = adata_fibroblast.obs['scvi_fib_leiden045']

# %%
sc.pp.filter_genes(adata_fibroblast, min_cells=3)
adata_fibroblast
adata_fibroblast = prep_anndata(adata_fibroblast)
adata_fibroblast

# %%
adata_fibroblast.obs['scvi_fib_leiden045'].value_counts()

# %%
%%R -i adata_fibroblast
# save a .rds for zlmCond as checkpoint
saveRDS(adata_fibroblast, "adata_fibroblast_OsteoVsF2.rds")

# %%
# save the adata_fibroblast
adata_fibroblast.write_h5ad("./anndata/R_NR_14DPA_fib_fullcounts_08052024.h5ad")

# %%
# load the adata_fibroblast
adata_fibroblast = sc.read_h5ad("./anndata/R_NR_14DPA_fib_fullcounts_08052024.h5ad")
adata_fibroblast

# %%
# select a specific group to investigate 
# remove the cells that does not exist in adata_all_subset
adata_fibClus = adata_fibroblast[adata_fibroblast.obs.scvi_fib_leiden045.isin(['P2-Firbroblast-1', 'P2-Firbroblast-2','Blastema-Cells'])].copy()
adata_fibClus

# %%
# rename the categories with the expected names
adata_fibClus.obs['scvi_fib_leiden045'] = adata_fibClus.obs['scvi_fib_leiden045'].astype(str)
adata_fibClus.obs['scvi_fib_leiden045'] = adata_fibClus.obs['scvi_fib_leiden045'].replace({'P2-Firbroblast-1':'Fibrosis-cells', 'P2-Firbroblast-2':'Fibrosis-cells'})
adata_fibClus.obs['scvi_fib_leiden045'] = adata_fibClus.obs['scvi_fib_leiden045'].astype('category')

# %%
# rename the annotation_int2 to cell_type
adata_fibClus.obs['cell_type'] = adata_fibClus.obs['annotation_int2']
# rename the scvi_fib_leiden045 to label
adata_fibClus.obs['label'] = adata_fibClus.obs['scvi_fib_leiden045']

# %%
sc.pp.filter_genes(adata_fibClus, min_cells=3)
adata_fibClus

# %%
adata_fibClus = prep_anndata(adata_fibClus)
adata_fibClus

# %%
adata_fibClus.obs["cell_type"] = [
    ct.replace(" ", "_") for ct in adata_fibClus.obs["cell_type"]
]

# %%
%%R
find_de_MAST_RE <- function(adata_){
    # create a MAST object
    sca <- SceToSingleCellAssay(adata_, class = "SingleCellAssay")
    print("Dimensions before subsetting:")
    print(dim(sca))
    print("")
    # keep genes that are expressed in more than 10% of all cells
    sca <- sca[freq(sca)>0.1,]
    print("Dimensions after subsetting:")
    print(dim(sca))
    print("")
    # add a column to the data which contains scaled number of genes that are expressed in each cell
    cdr2 <- colSums(assay(sca)>0)
    colData(sca)$ngeneson <- scale(cdr2)
    # store the columns that we are interested in as factors
    label <- factor(colData(sca)$label)
    # set the reference level
    label <- relevel(label,"Blastema-Cells")
    colData(sca)$label <- label
    celltype <- factor(colData(sca)$cell_type)
    colData(sca)$celltype <- celltype
    # same for donors (which we need to model random effects)
    replicate <- factor(colData(sca)$batch) # changed $replicate to $batch to match dataframe
    colData(sca)$replicate <- replicate
    # create a group per condition-celltype combination
    colData(sca)$group <- paste0(colData(adata_)$label, ".", colData(adata_)$cell_type)
    colData(sca)$group <- factor(colData(sca)$group)
    # define and fit the model
    zlmCond <- zlm(formula = ~ngeneson + group + (1 | replicate), # (1 | replicate): This is a random effects term.
                   sca=sca, 
                   method='glmer', 
                   ebayes=F, 
                   strictConvergence=F,
                   fitArgsD=list(nAGQ = 0)) # to speed up calculations
    # save a .rds for zlmCond as checkpoint
    saveRDS(zlmCond, "zlm_temp.rds")
    # perform likelihood-ratio test for the condition that we are interested in    
    summaryCond <- summary(zlmCond, doLRT='groupBlastema-Cells.Fibroblast')
    # get the table with log-fold changes and p-values
    summaryDt <- summaryCond$datatable
    result <- merge(summaryDt[contrast=='groupBlastema-Cells.Fibroblast' & component=='H',.(primerid, `Pr(>Chisq)`)], # p-values
                     summaryDt[contrast=='groupBlastema-Cells.Fibroblast' & component=='logFC', .(primerid, coef)],
                     by='primerid') # logFC coefficients
    # MAST uses natural logarithm so we convert the coefficients to log2 base to be comparable to edgeR
    result[,coef:=result[,coef]/log(2)]
    # do multiple testing correction
    result[,FDR:=p.adjust(`Pr(>Chisq)`, 'fdr')]
    result = result[result$FDR<0.01,, drop=F]

    result <- stats::na.omit(as.data.frame(result))
    return(result)
}

# %%
# reduce the adata_fibClus to 30 cells per cluster
# Assuming 'label' is the column in adata_fibClus.obs that contains the labels
labels = adata_fibClus.obs['label']

# Group the labels and select 10 cells per label
selected_cells = labels.groupby(labels).apply(lambda x: x.sample(n=30, replace=False)).index

# Select the cells in the AnnData object
adata_fibClus_subset = adata_fibClus[selected_cells.get_level_values(1)]

# %%
%%R -i adata_fibClus
# save a .rds for zlmCond as checkpoint
saveRDS(adata_fibClus, "adata_fibClus_3grp.rds")

# %%
%%R 
# load the adata_fibClus
adata_fibClus <- readRDS("./adata_fibClus_3grp.rds")
adata_fibClus

# %%
%%R
# load the saved rds
adata_ <- readRDS("./adata_fibClus_3grp.rds")
# create a MAST object
sca <- SceToSingleCellAssay(adata_, class = "SingleCellAssay")
print("Dimensions before subsetting:")
print(dim(sca))
print("")
# keep genes that are expressed in more than 10% of all cells
sca <- sca[freq(sca)>0.1,]
print("Dimensions after subsetting:")
print(dim(sca))
print("")
# add a column to the data which contains scaled number of genes that are expressed in each cell
# cellular detection rate 
cdr2 <- colSums(assay(sca)>0)
colData(sca)$ngeneson <- scale(cdr2)
# store the columns that we are interested in as factors
label <- factor(colData(sca)$label)
# set the reference level
label <- relevel(label,"Blastema-Cells")
colData(sca)$label <- label
celltype <- factor(colData(sca)$cell_type)
colData(sca)$celltype <- celltype
# same for donors (which we need to model random effects)
replicate <- factor(colData(sca)$batch) # changed $replicate to $batch to match dataframe
colData(sca)$replicate <- replicate
# create a group per condition-celltype combination
colData(sca)$group <- paste0(colData(adata_)$label, ".", colData(adata_)$cell_type)
colData(sca)$group <- factor(colData(sca)$group)
# define and fit the model
zlmCond <- zlm(formula = ~ngeneson + group + (1 | replicate), # (1 | replicate): This is a random effects term.
                sca=sca, 
                method='glmer', 
                ebayes=F, 
                strictConvergence=F,
                fitArgsD=list(nAGQ = 0)) # to speed up calculations
# save a .rds for zlmCond as checkpoint
saveRDS(zlmCond, "zlm_fibClus_subset.rds")

# %%
%%R 
summary(zlmCond)

# %%
%%R
# load the zlm_temp.rds
zlmCond <- readRDS("zlm_fibClus_subset.rds")
keys <- unique(summary(zlmCond)$datatable$contrast)
maxcount <- length(keys)
count <- 0
for (key in keys) {
    count <- count + 1
    if (!((count == maxcount) | (count == 1))) {
        print(key)
        # perform likelihood-ratio test for the condition that we are interested in    
        summaryCond <- summary(zlmCond, doLRT=key)
        # get the table with log-fold changes and p-values
        summaryDt <- summaryCond$datatable
        result <- merge(summaryDt[contrast==key & component=='H',.(primerid, `Pr(>Chisq)`)], # p-values
                        summaryDt[contrast==key & component=='logFC', .(primerid, coef)],
                        by='primerid') # logFC coefficients
        # MAST uses natural logarithm so we convert the coefficients to log2 base to be comparable to edgeR
        result[,coef:=result[,coef]/log(2)]
        # do multiple testing correction
        result[,FDR:=p.adjust(`Pr(>Chisq)`, 'fdr')]
        result = result[result$FDR<0.01,, drop=F]

        result <- stats::na.omit(as.data.frame(result))
        saveRDS(result, paste0("result_", count, ".rds"))
    }
}

# %%
%%R -o res
res <- readRDS("./rds_files/MastResults/Res_summary_Blas_groupOthers_Fibroblast.rds")

# %%
res[:5]

# %%
# make all columns in res to be str
res = res.astype(str)

# %%
# save the results 
res.to_csv("./R_NR_14DPA_fib_subsetClus_MAST_DEG_08052024.csv", index=False)

# %%
# read the results of csv # renamed
res = pd.read_csv("./rds_files/MastResults/RegenNonRegen14DPA/Res_summary_BlasVsF1F2_Fibroblast.csv")

# %%
res_copy = res.copy()

# %%
adata_fibClus.uns['Fibrosis_vs_Blastema']['scores']

# %%
res["gene_symbol"] = res["primerid"]
res["cell_type"] = "Fibroblast"
sc_toolbox.tools.de_res_to_anndata(
    adata_fibClus,
    res,
    groupby="cell_type",
    score_col="coef",
    pval_col="Pr(>Chisq)",
    pval_adj_col="FDR",
    lfc_col="coef",
    key_added="MAST_Fibroblast",
)

# %%
adata_fibClus.uns['Fibrosis_vs_Blastema']

# %%
# save the adata_fibClus 
adata_fibClus.write_h5ad("./anndata/R_NR_14DPA_fib_subsetClus_08052024_2.h5ad")

# %% [markdown]
# ## save info to adata_fibroblast

# %%
%%R -o res
res <- readRDS("./rds_files/MastResults/Res_summary_F2vsAllpLog_groupP2-Firbroblast-2_Fibroblast.rds")

# %%
# make all columns in res to be str
res = res.astype(str)

# %%
# rename the annotation_int2 to cell_type
adata_fibroblast.obs['cell_type'] = adata_fibroblast.obs['annotation_int2']
# rename the scvi_fib_leiden045 to label
adata_fibroblast.obs['label'] = adata_fibroblast.obs['scvi_fib_leiden045']

# %%
res["gene_symbol"] = res["primerid"]
res["cell_type"] = "Fibroblast"
sc_toolbox.tools.de_res_to_anndata(
    adata_fibroblast,
    res,
    groupby="cell_type",
    score_col="coef",
    pval_col="Pr(>Chisq)",
    # pval_adj_col="FDR",
    pval_adj_col="FDRlog",
    lfc_col="coef",
    key_added="MAST-F2VsAllpLog_Fibroblast",
)

# %%
# save the adata_fibroblast 
# fullcounts 2nd version has MAST results 
adata_fibroblast.write_h5ad("./anndata/R_NR_14DPA_fib_fullcounts_08052024_2.h5ad")

# %%
# load the adata_fibroblast
adata_fibroblast = sc.read_h5ad("./anndata/R_NR_14DPA_fib_fullcounts_08052024_2.h5ad")
adata_fibroblast

# %%
adata_fibroblast.uns['MAST-OsteoVsF2pLog_Fibroblast']

# %%
result['-logQ'] = result['-logQ'].replace([np.inf, -np.inf], 700)
result

# %%
# swap the lfcs for some groups since the key was messed up 
adata_fibroblast.uns["MAST-OsteoVsF2pLog_Fibroblast"]['logfoldchanges'] = np.array((-adata_fibroblast.uns["MAST-OsteoVsF2pLog_Fibroblast"]['logfoldchanges'].astype(float)).astype(str), dtype = [('Fibroblast', 'O')])

# %% [markdown]
# ### Visualisation of MAST

# %%
FDR = 0.01

def volcano_plot(adata_, group_key, group_name="cell_type", groupby="label", title=None, pdf_pages=None, pngsave=None, LOG_FOLD_CHANGE = 0.5, LogP = False, xlimdef = None, ylimdef = None):
    cell_type = "_".join(group_key.split("_")[1:])
    result = sc.get.rank_genes_groups_df(adata_, group=cell_type, key=group_key).copy()
    print(result[:10])
    if LogP == False:
        result["-logQ"] = -np.log(result["pvals_adj"].astype("float"))
        result['-logQ'] = result['-logQ'].replace([np.inf, -np.inf], 800)
        result["logfoldchanges"] = result["logfoldchanges"].astype("float")
    else:
        result["-logQ"] = result["pvals_adj"].astype("float")
        result["logfoldchanges"] = result["logfoldchanges"].astype("float")
        
    # for all logfold changes above xlimdef, set it as xlimdef
    if xlimdef is not None:
        result.loc[result["logfoldchanges"] > xlimdef, "logfoldchanges"] = xlimdef*0.99
        result.loc[result["logfoldchanges"] < (-xlimdef), "logfoldchanges"] = (-xlimdef*0.99)
    # for all -logQ above ylimdef, set it as ylimdef
    if ylimdef is not None:
        result.loc[result["-logQ"] > ylimdef, "-logQ"] = ylimdef*0.99
    
    print(result[:10])
    lowqval_de = result.loc[abs(result["logfoldchanges"]) > LOG_FOLD_CHANGE]
    other_de = result.loc[abs(result["logfoldchanges"]) <= LOG_FOLD_CHANGE]
    # print the genes that are differentially expressed
    print("Genes that are differentially expressed (up):")
    print(lowqval_de[lowqval_de["logfoldchanges"] > 0 ]['names'].to_list())
    print("Genes that are differentially expressed (down):")
    print(lowqval_de[lowqval_de["logfoldchanges"] < 0 ]['names'].to_list())
    fig, ax = plt.subplots()
    sns.regplot(
        x=other_de["logfoldchanges"],
        y=other_de["-logQ"],
        fit_reg=False,
        scatter_kws={"s": 9, "color": "black"},
    )
    sns.regplot(
        x=lowqval_de["logfoldchanges"],
        y=lowqval_de["-logQ"],
        fit_reg=False,
        scatter_kws={"s": 9, "color": "red"},
    )
    ax.set_xlabel("log2 FC")
    ax.set_ylabel("-log Q-value")
    if xlimdef is not None:
        ax.set_xlim(-xlimdef, xlimdef)
    if ylimdef is not None:
        ax.set_ylim(0, ylimdef)

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+.02, point['y'], str(point['val']))

    label_point(lowqval_de["logfoldchanges"], lowqval_de["-logQ"], lowqval_de['names'], plt.gca()) 
    
    if title is None:
        title = group_key.replace("_", " ")

    if pdf_pages is not None:
        plt.title(title)
        plt.show()
        pdf_pages.savefig(ax.figure, bbox_inches='tight')
    elif pngsave is not None:
        plt.savefig(pngsave, bbox_inches='tight')
    else:
        plt.title(title)
        plt.show()


# %%
FDR = 0.01
gene_to_plot = ['Acan','Ibsp','Alpl','Spp1','Pth1r','Cd200','Runx2','Hapln1','Col2a1','Sp7','Sparc','Sox9','Sox6','Bglap','Id1','S100a4','Loxl1','Cd44','Tnc','Acta2']
def volcano_plot_gene(adata_, group_key, group_name="cell_type", groupby="label", title=None, pdf_pages=None, pngsave=None, LOG_FOLD_CHANGE = 0.5, LogP = False, xlimdef = None, ylimdef = None):
    cell_type = "_".join(group_key.split("_")[1:])
    result = sc.get.rank_genes_groups_df(adata_, group=cell_type, key=group_key).copy()
    print(result[:10])
    if LogP == False:
        result["-logQ"] = -np.log(result["pvals_adj"].astype("float"))
        result['-logQ'] = result['-logQ'].replace([np.inf, -np.inf], 800)
        result["logfoldchanges"] = result["logfoldchanges"].astype("float")
    else:
        result["-logQ"] = result["pvals_adj"].astype("float")
        result["logfoldchanges"] = result["logfoldchanges"].astype("float")
    
    # for all logfold changes above xlimdef, set it as xlimdef
    if xlimdef is not None:
        result.loc[result["logfoldchanges"] > xlimdef, "logfoldchanges"] = xlimdef*0.99
        result.loc[result["logfoldchanges"] < (-xlimdef), "logfoldchanges"] = (-xlimdef*0.99)
    # for all -logQ above ylimdef, set it as ylimdef
    if ylimdef is not None:
        result.loc[result["-logQ"] > ylimdef, "-logQ"] = ylimdef*0.99

    # print(result[:10])
    lowqval_de = result.loc[abs(result["logfoldchanges"]) > LOG_FOLD_CHANGE]
    other_de = result.loc[abs(result["logfoldchanges"]) <= LOG_FOLD_CHANGE]
    # filter the lowqval_de with only genes in the gene_to_plot
    lowqval_de2 = lowqval_de[lowqval_de['names'].isin(gene_to_plot)]
    print(lowqval_de)
    # print the genes that are differentially expressed
    # print("Genes that are differentially expressed (up):")
    # print(lowqval_de[lowqval_de["logfoldchanges"] > 0 ]['names'].to_list())
    # print("Genes that are differentially expressed (down):")
    # print(lowqval_de[lowqval_de["logfoldchanges"] < 0 ]['names'].to_list())
    fig, ax = plt.subplots()
    sns.regplot(
        x=other_de["logfoldchanges"],
        y=other_de["-logQ"],
        fit_reg=False,
        scatter_kws={"s": 9, "color": "black"},
    )
    sns.regplot(
        x=lowqval_de["logfoldchanges"],
        y=lowqval_de["-logQ"],
        fit_reg=False,
        scatter_kws={"s": 9, "color": "red"},
    )
    sns.regplot(
        x=lowqval_de2["logfoldchanges"],
        y=lowqval_de2["-logQ"],
        fit_reg=False,
        scatter_kws={"s": 9, "color": "blue"},
    )
    ax.set_xlabel("log2 FC")
    ax.set_ylabel("-log Q-value")
    if xlimdef is not None:
        ax.set_xlim(-xlimdef, xlimdef)
    if ylimdef is not None:
        ax.set_ylim(0, ylimdef)

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x']+.02, point['y'], str(point['val']))

    label_point(lowqval_de2["logfoldchanges"], lowqval_de2["-logQ"], lowqval_de2['names'], plt.gca()) 
    
    if title is None:
        title = group_key.replace("_", " ")

    if pdf_pages is not None:
        plt.title(title)
        plt.show()
        pdf_pages.savefig(ax.figure, bbox_inches='tight')
    elif pngsave is not None:
        plt.savefig(pngsave, bbox_inches='tight')
    else:
        plt.title(title)
        plt.show()


# %%
FDR = 0.01

def volcano_plot_noLabel(adata_, group_key, group_name="cell_type", groupby="label", title=None, pdf_pages = None, pngsave = None, LOG_FOLD_CHANGE = 0.5, LogP = False, xlimdef = None, ylimdef = None):
    cell_type = "_".join(group_key.split("_")[1:])
    result = sc.get.rank_genes_groups_df(adata_, group=cell_type, key=group_key).copy()
    if LogP == False:
        result["-logQ"] = -np.log(result["pvals_adj"].astype("float"))
        result['-logQ'] = result['-logQ'].replace([np.inf, -np.inf], 800)
        result["logfoldchanges"] = result["logfoldchanges"].astype("float")
    else:
        result["-logQ"] = result["pvals_adj"].astype("float")
        result["logfoldchanges"] = result["logfoldchanges"].astype("float")
    
    # for all logfold changes above xlimdef, set it as xlimdef
    if xlimdef is not None:
        result.loc[result["logfoldchanges"] > xlimdef, "logfoldchanges"] = xlimdef*0.99
        result.loc[result["logfoldchanges"] < (-xlimdef), "logfoldchanges"] = (-xlimdef*0.99)
    # for all -logQ above ylimdef, set it as ylimdef
    if ylimdef is not None:
        result.loc[result["-logQ"] > ylimdef, "-logQ"] = ylimdef*0.99

    lowqval_de = result.loc[abs(result["logfoldchanges"]) > LOG_FOLD_CHANGE]
    other_de = result.loc[abs(result["logfoldchanges"]) <= LOG_FOLD_CHANGE]
    # print the genes that are differentially expressed
    print("Genes that are differentially expressed (up):")
    print(lowqval_de[lowqval_de["logfoldchanges"] > 0 ]['names'].to_list())
    print("Genes that are differentially expressed (down):")
    print(lowqval_de[lowqval_de["logfoldchanges"] < 0 ]['names'].to_list())

    fig, ax = plt.subplots()
    sns.regplot(
        x=other_de["logfoldchanges"],
        y=other_de["-logQ"],
        fit_reg=False,
        scatter_kws={"s": 1, "color": "black"},

    )
    sns.regplot(
        x=lowqval_de["logfoldchanges"],
        y=lowqval_de["-logQ"],
        fit_reg=False,
        scatter_kws={"s": 1, "color": "red"},
    )
    ax.set_xlabel("log2 FC")
    ax.set_ylabel("-log Q-value")
    if xlimdef is not None:
        ax.set_xlim(-xlimdef, xlimdef)
    if ylimdef is not None:
        ax.set_ylim(0, ylimdef)
    
    plt.setp(ax.spines.values(), linewidth=1)
    plt.rcParams['figure.figsize']=(2.25,2.25) #rescale figures
    plt.rcParams.update({'font.size': 8})
    
    if title is None:
        title = group_key.replace("_", " ")

    if pdf_pages is not None:
        plt.title(title)
        plt.show()
        pdf_pages.savefig(ax.figure, bbox_inches='tight')
    elif pngsave is not None:
        plt.savefig(pngsave, bbox_inches='tight', dpi=600)
    else:
        plt.title(title)
        plt.show()


# %%
volcano_plot_gene(adata_fibroblast, "MAST-F2VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibro 2 vs All cells", LOG_FOLD_CHANGE=0.5, LogP=True)
volcano_plot_noLabel(adata_fibroblast, "MAST-F2VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibro 2 vs All cells", LOG_FOLD_CHANGE=0.5, LogP=True)

# %%
volcano_plot(adata_fibroblast, "MAST-F1VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibro 1 vs All cells", LOG_FOLD_CHANGE=0.5, LogP=True) #xlimdef=1.5, ylimdef=2200)
volcano_plot_gene(adata_fibroblast, "MAST-F1VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibro 1 vs All cells", LOG_FOLD_CHANGE=0.5, LogP=True, xlimdef=1.6, ylimdef=2200, pngsave="F1vsAll_gene.png")
volcano_plot_noLabel(adata_fibroblast, "MAST-F1VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibro 1 vs All cells", LOG_FOLD_CHANGE=0.5, LogP=True, xlimdef=1.6, ylimdef=2200, pngsave="F1vsAll_nolabel.png")

# %%
volcano_plot(adata_fibroblast, "MAST-F2VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibro 2 vs All cells", LOG_FOLD_CHANGE=0.5, LogP=True) #xlimdef=1.5, ylimdef=2200)
volcano_plot_gene(adata_fibroblast, "MAST-F2VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibro 2 vs All cells", LOG_FOLD_CHANGE=0.5, LogP=True, xlimdef=1.6, ylimdef=2200, pngsave="F2vsAll_gene.png")
volcano_plot_noLabel(adata_fibroblast, "MAST-F2VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibro 2 vs All cells", LOG_FOLD_CHANGE=0.5, LogP=True, xlimdef=1.6, ylimdef=2200, pngsave="F2vsAll_nolabel.png")

# %%
volcano_plot_gene(adata_fibroblast, "MAST-OsteoVsF1pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 2 cells", LOG_FOLD_CHANGE=1, LogP=True, xlimdef=4, ylimdef=2200)
volcano_plot_noLabel(adata_fibroblast, "MAST-OsteoVsF1pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 1 cells", LOG_FOLD_CHANGE=1, LogP=True, xlimdef=4, ylimdef=2200, pngsave="OsteoF1.png")
volcano_plot_noLabel(adata_fibroblast, "MAST-OsteoVsF2pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 2 cells", LOG_FOLD_CHANGE=1, LogP=True, xlimdef=4, ylimdef=2200, pngsave="OsteoF2.png")

# %%
volcano_plot_gene(adata_fibroblast, "MAST-OsteoVsF1pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 1 cells", LOG_FOLD_CHANGE=1, LogP=True)

# %%
volcano_plot(adata_fibroblast, "MAST-F1F2vsAllTrim_Fibroblast", title = "Comparison (MAST) of Fibrotic cells vs All P3 cells")

# %%
volcano_plot(adata_fibroblast, "MAST-F1F2vsAllTrim2_Fibroblast", title = "Comparison (MAST) of Fibrotic cells vs Most P3 cells", LOG_FOLD_CHANGE=1)

# %%
def plotGraph(X,Y):
      fig = plt.figure()
      ### Plotting arrangements ###
      return fig

# %%
# create a pdf file that contains all volcano plots 
from matplotlib.backends.backend_pdf import PdfPages

pdf_pages = PdfPages('RegenNonRegenVolcano.pdf')
volcano_plot(adata_fibroblast, "MAST-F1F2vsAllTrim_Fibroblast", title = "Comparison (MAST) of Fibrotic cells vs All P3 cells", pdf_pages = pdf_pages)
volcano_plot_noLabel(adata_fibroblast, "MAST-F1F2vsAllTrim_Fibroblast", title = "Comparison (MAST) of Fibrotic cells vs All P3 cells", pdf_pages = pdf_pages)

volcano_plot(adata_fibroblast, "MAST-BlasVsAll_Fibroblast", title = "Comparison (MAST) of blastema cells vs all", pdf_pages = pdf_pages)
volcano_plot_noLabel(adata_fibroblast, "MAST-BlasVsAll_Fibroblast", title = "Comparison (MAST) of blastema cells vs all", pdf_pages = pdf_pages)

volcano_plot(adata_fibroblast, "MAST-F1F2VsAll_Fibroblast", title = "Comparison (MAST) of Fibrotic cells vs all", pdf_pages = pdf_pages)
volcano_plot_noLabel(adata_fibroblast, "MAST-F1F2VsAll_Fibroblast", title = "Comparison (MAST) of Fibrotic cells vs all", pdf_pages = pdf_pages)

volcano_plot(adata_fibroblast, "MAST-F1VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibrotic cells grp1 vs all", pdf_pages = pdf_pages, LogP=True)
volcano_plot_gene(adata_fibroblast, "MAST-F1VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibrotic cells grp1 vs all", pdf_pages = pdf_pages, LogP=True)
volcano_plot_noLabel(adata_fibroblast, "MAST-F1VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibrotic cells grp1 vs all", pdf_pages = pdf_pages, LogP=True)

volcano_plot(adata_fibroblast, "MAST-F2VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibrotic cells grp2 vs all", pdf_pages = pdf_pages, LogP=True)
volcano_plot_gene(adata_fibroblast, "MAST-F2VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibrotic cells grp2 vs all", pdf_pages = pdf_pages, LogP=True)
volcano_plot_noLabel(adata_fibroblast, "MAST-F2VsAllpLog_Fibroblast", title = "Comparison (MAST) of Fibrotic cells grp2 vs all", pdf_pages = pdf_pages, LogP=True)

# volcano_plot(adata_fibroblast, "MAST-BlasVsF1F2_Fibroblast", title = "Comparison (MAST) of Blastema cells vs Fibrotic cells", pdf_pages = pdf_pages)
# volcano_plot_noLabel(adata_fibroblast, "MAST-BlasVsF1F2_Fibroblast", title = "Comparison (MAST) of Blastema cells vs Fibrotic cells", pdf_pages = pdf_pages)

volcano_plot(adata_fibroblast, "MAST-OsteoVsF1pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 1 cells", pdf_pages = pdf_pages, LOG_FOLD_CHANGE=1, LogP=True)
volcano_plot_gene(adata_fibroblast, "MAST-OsteoVsF1pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 1 cells", pdf_pages = pdf_pages, LOG_FOLD_CHANGE=1, LogP=True)
volcano_plot_noLabel(adata_fibroblast, "MAST-OsteoVsF1pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 1 cells", pdf_pages = pdf_pages, LOG_FOLD_CHANGE=1, LogP=True)

volcano_plot(adata_fibroblast, "MAST-OsteoVsF2pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 2 cells", pdf_pages = pdf_pages, LOG_FOLD_CHANGE=1, LogP=True)
volcano_plot_gene(adata_fibroblast, "MAST-OsteoVsF2pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 2 cells", pdf_pages = pdf_pages, LOG_FOLD_CHANGE=1, LogP=True)
volcano_plot_noLabel(adata_fibroblast, "MAST-OsteoVsF2pLog_Fibroblast", title = "Comparison (MAST) of Osteo cells vs Fibro 2 cells", pdf_pages = pdf_pages, LOG_FOLD_CHANGE=1, LogP=True)
pdf_pages.close()

# %%
def GetGenes(adata_, group_key, dictionary):
    cell_type = "_".join(group_key.split("_")[1:])
    result = sc.get.rank_genes_groups_df(adata_, group=cell_type, key=group_key).copy()
    result["-logQ"] = -np.log(result["pvals_adj"].astype("float"))
    result["logfoldchanges"] = result["logfoldchanges"].astype("float")
    lowqval_de = result.loc[abs(result["logfoldchanges"]) > LOG_FOLD_CHANGE]
    up_genes = lowqval_de[lowqval_de["logfoldchanges"] > 0 ]
    dictionary[f"{group_key}_up_genes"] = pd.DataFrame(up_genes)
    down_genes = lowqval_de[lowqval_de["logfoldchanges"] < 0 ]
    dictionary[f"{group_key}_down_genes"] = pd.DataFrame(down_genes)

# %%
Gene_dict = {}
GetGenes(adata_fibroblast, "MAST-BlasVsAll_Fibroblast", Gene_dict)
GetGenes(adata_fibroblast, "MAST-F1F2VsAll_Fibroblast", Gene_dict)
GetGenes(adata_fibroblast, "MAST-F1VsAll_Fibroblast", Gene_dict)
GetGenes(adata_fibroblast, "MAST-F2VsAll_Fibroblast", Gene_dict)
GetGenes(adata_fibroblast, "MAST-BlasVsF1F2_Fibroblast", Gene_dict)

# %%
for key in Gene_dict.keys():
    Gene_dict[key].to_csv(f"./rds_files/MastResults/RegenNonRegen14DPA/{key}.csv", index=False)

# %% [markdown]
# # GO analysis
# ## use results from MAST

# %% [markdown]
# Completed in Regen-NonRegen_14DPA_MAST_GO.ipynb

# %%
# create the Markers dataframe to input (Blastema vs all/ Fibrotic cells vs all /  Fibrotic cells grp1 vs all / Fibrotic cells grp2 vs all)
# take the upregulated genes only 
# Dataframe contains the "Gene_symb	pvals_adj	lfc	scores	group"
result1 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-BlasVsAll_Fibroblast').copy()
result1['group'] = 'BlastemaVsAll'
result2 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-F1F2VsAll_Fibroblast').copy()
result2['group'] = 'FibroticVsAll'
result3 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-F1VsAll_Fibroblast').copy()
result3['group'] = 'Fibrotic1VsAll'
result4 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-F2VsAll_Fibroblast').copy()
result4['group'] = 'Fibrotic2VsAll'
result5 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-OsteoVsF1pLog_Fibroblast').copy()
result5['group'] = 'OsteoVsF1'
result6 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-OsteoVsF2pLog_Fibroblast').copy()
result6['group'] = 'OsteoVsF2'
# merge the dataframes
Markers = pd.concat([result5, result6])
Markers

# %%
result5 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-OsteoVsF1pLog_Fibroblast').copy()
result6 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-OsteoVsF2pLog_Fibroblast').copy()

# %%
result7 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-F1VsAllpLog_Fibroblast').copy()
result8 = sc.get.rank_genes_groups_df(adata_fibroblast, group='Fibroblast', key='MAST-F2VsAllpLog_Fibroblast').copy()

# %%
# create a new column that groups upregulated genes and downregulated genes
# all pvals_adj are in log scale
result5['logfoldchanges'] = result5['logfoldchanges'].astype(float)
result5['up_downGroup'] = np.where(result5['logfoldchanges'] > 0, 'upregulated', 'downregulated')
# remove cells pvals_adj that are below log0.01
result5 = result5[result5['pvals_adj'].astype(float) > (np.log(0.01))]
# filter genes that have abs(lfc > 0.5)
result5 = result5[abs(result5['logfoldchanges']) > 0.5]
result5 = result5.sort_values(by='logfoldchanges', ascending=False)
result5

result6['logfoldchanges'] = result6['logfoldchanges'].astype(float)
result6['up_downGroup'] = np.where(result6['logfoldchanges'] > 0, 'upregulated', 'downregulated')
# remove cells pvals_adj that are below log0.01
result6 = result6[result6['pvals_adj'].astype(float) > (np.log(0.01))]
# filter genes that have abs(lfc > 0.5)
result6 = result6[abs(result6['logfoldchanges']) > 0.5]
result6 = result6.sort_values(by='logfoldchanges', ascending=False)
result6


# %%

result7['logfoldchanges'] = result7['logfoldchanges'].astype(float)
result7['up_downGroup'] = np.where(result7['logfoldchanges'] > 0, 'upregulated', 'downregulated')
# remove cells pvals_adj that are below log0.01
result7 = result7[result7['pvals_adj'].astype(float) > (np.log(0.01))]
# filter genes that have abs(lfc > 0.5)
result7 = result7[abs(result7['logfoldchanges']) > 0.5]
result7 = result7.sort_values(by='logfoldchanges', ascending=False)
result7

result8['logfoldchanges'] = result8['logfoldchanges'].astype(float)
result8['up_downGroup'] = np.where(result8['logfoldchanges'] > 0, 'upregulated', 'downregulated')
# remove cells pvals_adj that are below log0.01
result8 = result8[result8['pvals_adj'].astype(float) > (np.log(0.01))]
# filter genes that have abs(lfc > 0.5)
result8 = result8[abs(result8['logfoldchanges']) > 0.5]
result8 = result8.sort_values(by='logfoldchanges', ascending=False)
result8

# %%
%%R 
library(GO.db)
org.Mm.eg.db

# %%
Markers = result8

# %%
%%R -i Markers 
# Perform pathway analysis on the top 20 genes 
library("clusterProfiler")
library("org.Mm.eg.db")
library("dplyr")
library("ggplot2")
library("enrichplot")
# get the top 10 genes
n_genes <- 50
Markers_fil <- head(Markers, n_genes)
# get the top and bottom 10 genes
# Markers_fil <- rbind(head(Markers, n_genes), tail(Markers, n_genes))
# Markers_fil <- Markers # not filtering (not good)
# Markers_fil <- Markers %>% group_by(up_downGroup) %>% top_n(n=10, wt = logfoldchanges ) # use scores or lfc or pvals_adj??
# print(Markers_fil)
# add extra column of entrez id
Markers_fil$entrez <- mapIds(org.Mm.eg.db, Markers_fil$names, "ENTREZID", "SYMBOL", multiVals = "first")
# show the shape of the Markers_fil
print(dim(Markers_fil))
# remove the lines that could not be mapped
Markers_fil <- Markers_fil[!is.na(Markers_fil$entrez),]
print(dim(Markers_fil))
# # create a list of genes for each cluster
# Markers_fil <- split(Markers_fil$entrez, Markers_fil$up_downGroup)
# print(Markers_fil)

# compareCluster
# One of "groupGO", "enrichGO", "enrichKEGG", "enrichDO" or "enrichPathway"
clustersummary <- compareCluster(geneClusters = Markers_fil, fun = "enrichGO", OrgDb = org.Mm.eg.db, ont = "MF", pvalueCutoff = 0.05, pAdjustMethod = "BH", qvalueCutoff = 0.05, readable = TRUE)
# print(clustersummary)

# Extract the data frame from the compareClusterResult object
df <- as.data.frame(clustersummary)

# Convert GeneRatio from a fraction string to a numeric value
df <- df %>%
  mutate(GeneRatio = sapply(GeneRatio, function(x) {
    parts <- strsplit(x, "/")[[1]]
    as.numeric(parts[1]) / as.numeric(parts[2])
  }))


# Sort the data frame by adjusted p-value
df <- df %>%
  arrange(p.adjust, ascending = TRUE)

# subset to only top 10
df <- df[1:15,]

print(df)

pdf("./F2vsAll_GO2.pdf", 
    width = 6, height = 5)
title <- "GO analysis of the upregulated genes in \n Fibro 2 vs All cells"
p1 <- ggplot(df, aes(x = GeneRatio, y = reorder(Description, p.adjust), fill = p.adjust)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "#87ceeb", high = "#2b4e6d") +
  labs(x = "Gene Ratio", y = "Pathway", fill = "Adjusted p-value") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        plot.margin = margin(1, 7, 1, 15),
        axis.title.x = element_blank(),
        legend.text = element_text(size = 5),
        legend.title = element_text(size = 5)) +
  coord_flip() + 
  theme(aspect.ratio = 3/8)
  # theme(
  #  axis.line = element_line(color='black'),
  #  plot.background = element_blank(),
  #  panel.grid.major = element_blank(),
  #  panel.grid.minor = element_blank(),
  #  panel.border = element_blank()
  # )
print(p1 + ggtitle(title))
dev.off()
print(p1)


# %% [markdown]
# ## Correlation analysis 

# %%
# create a copy of adata_fibroblast
adata_fibroblast_og = adata_fibroblast.copy()
# calculate the highly variable genes 
sc.pp.highly_variable_genes(adata_fibroblast, n_top_genes=2000, flavor="seurat_v3", subset=True)

# %%
# create new obs identity by combining the amp location and the subcluster
adata_fibroblast.obs['amp_subcluster'] = adata_fibroblast.obs['Amp_location'].astype(str) + '_' + adata_fibroblast.obs['scvi_fib_leiden045'].astype(str)
adata_fibroblast.obs['amp_subcluster'] = adata_fibroblast.obs['amp_subcluster'].astype('category')
adata_fibroblast.obs['amp_subcluster'].cat.categories

# %%
# calculate the correlation matrix
# sc.tl.rank_genes_groups(adata_fibroblast, groupby='amp_subcluster', method='wilcoxon', n_genes=2000, use_raw=False, key_added='rank_genes_groups')
# plot the correlation matrix
sc.pl.correlation_matrix(adata_fibroblast, groupby='amp_subcluster', cmap='coolwarm', show_correlation_numbers = True, save='correlation_matrix_amp_subcluster.pdf')

# %%
sc.pl.violin(adata_fibroblast, keys=['Acan','Hapln1','Cd44'], groupby='scvi_fib_leiden045', use_raw=False, rotation=90, save='violin_RNR_fib_cellsubtypes_Acan-Hapln1-Cd44_2.pdf')

# %%
# delete the new obs column
del adata_fibroblast.obs['amp_subcluster']

# %% [markdown]
# # Matrisome scoring

# %%
# read the Mm_Matrisome_MGI.xlsx file 
matrisome = pd.read_excel("Mm_Matrisome_Masterlist_Naba.xlsx")
matrisome.head()

# %%
# select only the division of core matrisome
core_matrisome = matrisome[matrisome['Division'] == 'Core matrisome']
# subset the Collagens and proteoglycans
core_matrisome = core_matrisome[core_matrisome['Category'].isin(['Collagens', 'Proteoglycans'])]
# # subset the proteoglycans
# core_matrisome = core_matrisome[core_matrisome['Category'] == 'Proteoglycans']
# take the Symbol column and convert it to a list
matrisome_genes = core_matrisome['Gene Symbol'].tolist()
matrisome_genes

# %%
# do scoring for the matrisome genes
sc.tl.score_genes(adata_fibroblast, matrisome_genes, score_name='matrisome_score', use_raw=True)

# %%
# plot the matrisome scoring on the umap
sc.pl.umap(adata_fibroblast, color='matrisome_score', cmap='RdBu_r', size=20, use_raw=True, save='_RNR14DPA_fibroblast_matrisome_score.pdf')
sc.pl.violin(adata_fibroblast, keys='matrisome_score', groupby='scvi_fib_leiden045', rotation=90)

# %%
adata_fibroblast.obs['batch'].value_counts()

# %%
adata_fibroblast.obs['scvi_fib_leiden045'].value_counts()

# %%
adata_fibroblast.obs['batch'] = adata_fibroblast.obs['batch'].replace({
    'NRB1': 'Storer et al. NonRegeneration Dataset 1',
    'NRB2': 'Mui et al. NonRegeneration Dataset 1',
    'NRB3': 'Mui et al. NonRegeneration Dataset 2',
    'RB1': 'Storer et al. Regeneration Dataset 1',
    'RB2': 'Storer et al. Regeneration Dataset 2',
    'RJS': 'Johnson et al. Regeneration Dataset',
    })
# rank them based on the order above
batch_order = ['Mui et al. NonRegeneration Dataset 1', 'Mui et al. NonRegeneration Dataset 2', 'Storer et al. NonRegeneration Dataset 1',
                'Storer et al. Regeneration Dataset 1', 'Storer et al. Regeneration Dataset 2', 'Johnson et al. Regeneration Dataset'
]
adata_fibroblast.obs['batch'] = adata_fibroblast.obs['batch'].astype("category")

adata_fibroblast.obs['batch'] = adata_fibroblast.obs['batch'].cat.reorder_categories(batch_order, ordered=True)

# %%
# proportional analysis on the fibroblast dataset
# create bar plots for the proportion of cells in each condition
all_df = pd.DataFrame(adata_fibroblast.obs.groupby(['batch','scvi_fib_leiden045']).size(), columns = ['count'])
# add an extra column of percentage
all_df['percentage'] = all_df.groupby(level=0, group_keys=False).apply(lambda x: 100 * x / float(x.sum()))
all_df

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
labels = adata_fibroblast.obs['scvi_fib_leiden045'].cat.categories
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.savefig(
    'figures/CellTypeProportion_RNR14_fib_count_batch_20250926.pdf',
    bbox_inches='tight'
)
plt.show()


# %%
# export the df as a csv file
all_df.to_csv('RNR14_fib_Leiden045_CellTypeProportion_20250926.csv')

# %%
# plot a bar chart of the percentage of cells in each cluster
ax = all_df['percentage'].unstack().plot(kind='barh', stacked=True, figsize=(10,5),
                      title='Percentage of cell type in batch', 
                      ylabel='Batch', xlabel='Percentage', legend=False,
                      )
labels = adata_fibroblast.obs['scvi_fib_leiden045'].cat.categories
ax.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# %%
sc.tl.score_genes(adata_all_subset, matrisome_genes, score_name='matrisome_score', use_raw=True)

# %%
sc.pl.umap(adata_all_subset, color='matrisome_score', cmap='RdBu_r', size=20, use_raw=True, save='_RNR14DPA_all_matrisome_score.pdf')

# %%
adata_all_subset

# %%
# take the hexcodes for the umap for batches
batchColor = adata_all_subset.uns['batch_colors']
# change the last one to olive
batchColor[-1] = '#808000'
# change the 3rd last to red
batchColor[-3] = '#FF0000'

# %%
# reverse the order of the adata_all_subset
sc.pl.umap(adata_all_subset[::-1], color='batch', size = 20, save="_all_batch.pdf")

# %%
# reverse the order of the adata_all_subset
sc.pl.umap(adata_all_subset[::-1], color='Amp_location', size = 20, save="_all_ampLocation.pdf")

# %%
# reverse the order of the adata_all_subset
sc.pl.umap(adata_all_subset[::-1], color='annotation_int2', size = 20, save="_all_annotation.pdf")


