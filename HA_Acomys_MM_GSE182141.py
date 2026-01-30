# %% [markdown]
# # Exploring the HA and Fibrillar collagen in other systems
# Is fibrillar collagen versus HA dichotomy application to regenerative outcomes or specific to digit tip amputation? 
# 
# Dataset: GSE182141

# %%
import os
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.3'

import rpy2.rinterface_lib.callbacks
import logging
from rpy2.robjects import pandas2ri
import anndata2ri

# %%
# Import required packages
import pandas as pd
import numpy as np
import os
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
import torch
import anndata as ad 
import seaborn as sb
import biomart
scvi.settings.seed = 0
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")
sc.settings._vector_friendly = False

# %%
data_folder = "AS_GSE182141"
datasets = {}

# Automatically detect unique prefixes (e.g., "GSM5519169_Mus00_", "GSM5519170_Mus03_", etc.)
all_files = os.listdir(data_folder)
prefixes = sorted(
    set(f.split("_")[:2][0] + "_" + f.split("_")[1] + "_" for f in all_files if f.endswith(".gz"))
)

for prefix in prefixes:
    adata = sc.read_10x_mtx(
        path=data_folder,
        prefix=prefix,               # Use the detected prefix
        var_names="gene_symbols",    # or "gene_ids"
        
        cache=True
    )
    datasets[prefix] = adata

# Example: examine one of the data objects
print(datasets["GSM5519169_Mus00_"])


# %%
# Examine the loaded datasets and define species groups
print("Loaded datasets:")
for prefix, adata in datasets.items():
    print(f"{prefix}: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Show first few gene names to understand naming convention
    print(f"  First 10 genes: {list(adata.var_names[:10])}")
    print(f"  Sample gene names: {list(adata.var_names[100:110])}")
    print()

# Identify Acomys vs Mus datasets
acomys_datasets = [prefix for prefix in datasets.keys() if 'Aco' in prefix]
mus_datasets = [prefix for prefix in datasets.keys() if 'Mus' in prefix]

print(f"Acomys datasets: {acomys_datasets}")
print(f"Mus datasets: {mus_datasets}")

# Check if gene names differ between species
if acomys_datasets and mus_datasets:
    aco_genes = set(datasets[acomys_datasets[0]].var_names)
    mus_genes = set(datasets[mus_datasets[0]].var_names)
    
    print(f"\nAcomys genes (first dataset): {len(aco_genes)} total")
    print(f"Mus genes (first dataset): {len(mus_genes)} total")
    print(f"Common genes: {len(aco_genes.intersection(mus_genes))}")
    print(f"Acomys-specific genes: {len(aco_genes - mus_genes)}")
    print(f"Mus-specific genes: {len(mus_genes - aco_genes)}")
    
    # Show some examples of different genes
    aco_specific = list(aco_genes - mus_genes)[:10]
    mus_specific = list(mus_genes - aco_genes)[:10]
    
    print(f"\nExample Acomys-specific genes: {aco_specific}")
    print(f"Example Mus-specific genes: {mus_specific}")

# %%
# Import additional packages for filtering and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("Setting up quality control filtering pipeline...")

def calculate_qc_metrics(adata, species_type):
    """Calculate QC metrics for filtering"""
    # Basic metrics
    adata.var['n_cells'] = (adata.X > 0).sum(0).A1
    adata.obs['n_genes'] = (adata.X > 0).sum(1).A1
    adata.obs['total_counts'] = adata.X.sum(1).A1
    
    # Species-specific metrics (only for mouse)
    if species_type == 'mouse':
        # Mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith('mt-') | adata.var_names.str.startswith('Mt-') | adata.var_names.str.startswith('MT-')
        adata.obs['pct_counts_mt'] = (adata[:, adata.var['mt']].X.sum(1).A1 / adata.obs['total_counts']) * 100
        
        # Hemoglobin genes
        hb_genes = adata.var_names.str.contains('^Hb[^(p)]') | adata.var_names.str.contains('^HB[^(P)]')
        adata.var['hb'] = hb_genes
        adata.obs['pct_counts_hb'] = (adata[:, adata.var['hb']].X.sum(1).A1 / adata.obs['total_counts']) * 100
    else:
        # For Acomys, set placeholders
        adata.obs['pct_counts_mt'] = 0
        adata.obs['pct_counts_hb'] = 0
    
    return adata

print("Calculating QC metrics for all datasets...")

# %%
# Calculate QC metrics for all datasets
datasets_with_qc = {}

for prefix, adata in datasets.items():
    print(f"Processing {prefix}...")
    adata_copy = adata.copy()
    
    # Determine species type
    species_type = 'mouse' if 'Mus' in prefix else 'acomys'
    
    # Calculate QC metrics
    adata_with_qc = calculate_qc_metrics(adata_copy, species_type)
    datasets_with_qc[prefix] = adata_with_qc
    
    print(f"  {prefix}: {adata_with_qc.n_obs} cells, {adata_with_qc.n_vars} genes")
    print(f"  Mean genes/cell: {adata_with_qc.obs['n_genes'].mean():.1f}")
    print(f"  Mean counts/cell: {adata_with_qc.obs['total_counts'].mean():.1f}")
    
    if species_type == 'mouse':
        print(f"  Mean MT%: {adata_with_qc.obs['pct_counts_mt'].mean():.2f}%")
        print(f"  Mean HB%: {adata_with_qc.obs['pct_counts_hb'].mean():.2f}%")
    print()

print("QC metrics calculated for all datasets!")

# %%
# Visualize QC metrics before filtering
def plot_qc_metrics(datasets_dict, title_suffix=""):
    """Plot QC metrics for all datasets"""
    
    # Prepare data for plotting
    all_data = []
    for prefix, adata in datasets_dict.items():
        species = 'Mouse' if 'Mus' in prefix else 'Acomys'
        timepoint = prefix.split('_')[1]
        
        temp_df = adata.obs.copy()
        temp_df['dataset'] = prefix
        temp_df['species'] = species
        temp_df['timepoint'] = timepoint
        all_data.append(temp_df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Quality Control Metrics {title_suffix}', fontsize=16)
    
    # 1. Number of genes per cell
    sns.violinplot(data=combined_df, x='species', y='n_genes', hue='timepoint', ax=axes[0, 0])
    axes[0, 0].set_title('Number of Genes per Cell')
    axes[0, 0].axhline(y=500, color='red', linestyle='--', alpha=0.7, label='Filter threshold')
    axes[0, 0].legend()
    
    # 2. Total counts per cell
    sns.violinplot(data=combined_df, x='species', y='total_counts', ax=axes[0, 1])
    axes[0, 1].set_title('Total Counts per Cell')
    axes[0, 1].set_yscale('log')
    
    # 3. Genes per dataset (number of cells expressing each gene)
    gene_counts = []
    for prefix, adata in datasets_dict.items():
        species = 'Mouse' if 'Mus' in prefix else 'Acomys'
        gene_counts.extend([(species, prefix, count) for count in adata.var['n_cells']])
    
    gene_df = pd.DataFrame(gene_counts, columns=['species', 'dataset', 'n_cells'])
    sns.violinplot(data=gene_df, x='species', y='n_cells', ax=axes[0, 2])
    axes[0, 2].set_title('Cells per Gene')
    axes[0, 2].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Filter threshold')
    axes[0, 2].legend()
    
    # 4. Mitochondrial percentage (Mouse only)
    mouse_data = combined_df[combined_df['species'] == 'Mouse']
    if len(mouse_data) > 0 and mouse_data['pct_counts_mt'].max() > 0:
        sns.violinplot(data=mouse_data, x='timepoint', y='pct_counts_mt', ax=axes[1, 0])
        axes[1, 0].set_title('Mitochondrial % (Mouse only)')
        axes[1, 0].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Filter threshold')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No MT data available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Mitochondrial % (Mouse only)')
    
    # 5. Hemoglobin percentage (Mouse only)
    if len(mouse_data) > 0 and mouse_data['pct_counts_hb'].max() > 0:
        sns.violinplot(data=mouse_data, x='timepoint', y='pct_counts_hb', ax=axes[1, 1])
        axes[1, 1].set_title('Hemoglobin % (Mouse only)')
        axes[1, 1].axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='Filter threshold')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No HB data available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Hemoglobin % (Mouse only)')
    
    # 6. Summary statistics
    axes[1, 2].axis('off')
    summary_text = f"""Summary Statistics {title_suffix}:
    
Total cells: {len(combined_df):,}
Total genes: {combined_df.groupby('dataset')['n_genes'].first().iloc[0] if len(combined_df) > 0 else 0:,}

Mouse datasets: {len(combined_df[combined_df['species'] == 'Mouse']['dataset'].unique())}
Acomys datasets: {len(combined_df[combined_df['species'] == 'Acomys']['dataset'].unique())}

Mean genes/cell: {combined_df['n_genes'].mean():.1f}
Mean counts/cell: {combined_df['total_counts'].mean():.1f}"""
    
    if len(mouse_data) > 0:
        summary_text += f"""
Mean MT% (Mouse): {mouse_data['pct_counts_mt'].mean():.2f}%
Mean HB% (Mouse): {mouse_data['pct_counts_hb'].mean():.2f}%"""
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', family='monospace')
    
    plt.tight_layout()
    plt.show()

# Plot QC metrics before filtering
print("Plotting QC metrics BEFORE filtering...")
plot_qc_metrics(datasets_with_qc, "(Before Filtering)")

# %%
# Apply filtering with specified criteria
def apply_filtering(datasets_dict):
    """
    Apply filtering criteria:
    - Remove cells with < 500 genes
    - Remove cells with MT% > 20% (mouse only)
    - Remove cells with HB% > 2.5% (mouse only)
    - Keep genes with >= 1 read
    """
    
    filtered_datasets = {}
    filtering_stats = {}
    
    for prefix, adata in datasets_dict.items():
        print(f"\nFiltering {prefix}...")
        adata_filtered = adata.copy()
        species_type = 'mouse' if 'Mus' in prefix else 'acomys'
        
        # Store original counts
        n_cells_orig = adata_filtered.n_obs
        n_genes_orig = adata_filtered.n_vars
        
        # 1. Filter cells with < 500 genes
        cell_filter = adata_filtered.obs['n_genes'] >= 500
        print(f"  Cells with >= 500 genes: {cell_filter.sum()}/{len(cell_filter)} ({cell_filter.mean()*100:.1f}%)")
        
        # 2. Species-specific filtering for mouse
        if species_type == 'mouse':
            # Mitochondrial filtering
            mt_filter = adata_filtered.obs['pct_counts_mt'] <= 20
            print(f"  Cells with MT% <= 20%: {mt_filter.sum()}/{len(mt_filter)} ({mt_filter.mean()*100:.1f}%)")
            
            # Hemoglobin filtering
            hb_filter = adata_filtered.obs['pct_counts_hb'] <= 2.5
            print(f"  Cells with HB% <= 2.5%: {hb_filter.sum()}/{len(hb_filter)} ({hb_filter.mean()*100:.1f}%)")
            
            # Combine all cell filters
            combined_cell_filter = cell_filter & mt_filter & hb_filter
        else:
            # For Acomys, only gene count filter
            combined_cell_filter = cell_filter
        
        # Apply cell filtering
        adata_filtered = adata_filtered[combined_cell_filter, :].copy()
        n_cells_filtered = adata_filtered.n_obs
        
        # 3. Filter genes with >= 1 read
        gene_filter = adata_filtered.var['n_cells'] >= 1
        adata_filtered = adata_filtered[:, gene_filter].copy()
        n_genes_filtered = adata_filtered.n_vars
        
        print(f"  Gene filtering: {gene_filter.sum()}/{len(gene_filter)} genes kept ({gene_filter.mean()*100:.1f}%)")
        
        # Store filtered dataset
        filtered_datasets[prefix] = adata_filtered
        
        # Store filtering statistics
        filtering_stats[prefix] = {
            'cells_original': n_cells_orig,
            'cells_filtered': n_cells_filtered,
            'cells_removed': n_cells_orig - n_cells_filtered,
            'cell_retention_pct': (n_cells_filtered / n_cells_orig) * 100,
            'genes_original': n_genes_orig,
            'genes_filtered': n_genes_filtered,
            'genes_removed': n_genes_orig - n_genes_filtered,
            'gene_retention_pct': (n_genes_filtered / n_genes_orig) * 100
        }
        
        print(f"  Final: {n_cells_filtered} cells ({(n_cells_filtered/n_cells_orig)*100:.1f}% retained), "
              f"{n_genes_filtered} genes ({(n_genes_filtered/n_genes_orig)*100:.1f}% retained)")
    
    return filtered_datasets, filtering_stats

# Apply filtering
print("Applying filtering criteria...")
datasets_filtered, filtering_stats = apply_filtering(datasets_with_qc)

# Print filtering summary
print("\n" + "="*60)
print("FILTERING SUMMARY")
print("="*60)
for prefix, stats in filtering_stats.items():
    print(f"{prefix}:")
    print(f"  Cells: {stats['cells_original']:,} → {stats['cells_filtered']:,} "
          f"({stats['cell_retention_pct']:.1f}% retained)")
    print(f"  Genes: {stats['genes_original']:,} → {stats['genes_filtered']:,} "
          f"({stats['gene_retention_pct']:.1f}% retained)")
    print()

# %%
# Recalculate QC metrics for filtered datasets and visualize
print("Recalculating QC metrics for filtered datasets...")

# Recalculate QC metrics for filtered datasets
datasets_filtered_qc = {}
for prefix, adata in datasets_filtered.items():
    species_type = 'mouse' if 'Mus' in prefix else 'acomys'
    adata_with_qc = calculate_qc_metrics(adata.copy(), species_type)
    datasets_filtered_qc[prefix] = adata_with_qc

# Plot QC metrics after filtering
print("Plotting QC metrics AFTER filtering...")
plot_qc_metrics(datasets_filtered_qc, "(After Filtering)")

# Update the main datasets dictionary with filtered data
print("Updating main datasets dictionary with filtered data...")
datasets = datasets_filtered_qc.copy()

# %%
# common genes verification and conversion (harmonisation)
def verify_genes(acomys_datasets, mus_datasets, datasets):
    
    # Step 1: Collect all unique Acomys genes across all datasets
    print("Step 1: Collecting all unique Acomys genes...")
    all_acomys_genes = set()
    for aco_prefix in acomys_datasets:
        aco_adata = datasets[aco_prefix]
        all_acomys_genes.update(aco_adata.var_names)
        print(f"  {aco_prefix}: {aco_adata.n_vars} genes")
    
    all_acomys_genes = sorted(list(all_acomys_genes))
    print(f"Total unique Acomys genes: {len(all_acomys_genes)}")
    
    # Step 2: Get all mouse genes for matching
    mouse_genes = set()
    for prefix in mus_datasets:
        mouse_genes.update(datasets[prefix].var_names)
    
    print(f"Total unique mouse genes: {len(mouse_genes)}")
    
    # Step 3: Find common genes for all Acomys genes
    print("\nStep 2: Finding unified Acomys gene set...")
    harmonise_mapping = {}
    
    # Direct exact matches
    all_acomys_genes_set = set(all_acomys_genes)
    direct_matches = all_acomys_genes_set.intersection(mouse_genes)
    for gene in direct_matches:
        harmonise_mapping[gene] = gene
    
    # Case-insensitive matches for remaining genes
    remaining_genes = set(all_acomys_genes) - direct_matches
    mouse_genes_upper = {g.upper(): g for g in mouse_genes}
    
    for acomys_gene in remaining_genes:
        if acomys_gene.upper() in mouse_genes_upper:
            mouse_gene = mouse_genes_upper[acomys_gene.upper()]
            harmonise_mapping[acomys_gene] = mouse_gene
    
    print(f"  Total Acomys genes: {len(all_acomys_genes)}")
    print(f"  Direct matches: {len(direct_matches)}")
    print(f"  Case-insensitive matches: {len(harmonise_mapping) - len(direct_matches)}")
    print(f"  Total common found: {len(harmonise_mapping)}")
    print(f"  Genes not shared: {len(all_acomys_genes) - len(harmonise_mapping)}")
    
    # Step 4: Get the unified gene set (mouse gene names)
    harmonised_genes_mouse_names = sorted(list(set(harmonise_mapping.values())))
    print(f"  Unique mouse names: {len(harmonised_genes_mouse_names)}")
    
    # Step 5: Apply harmonisation to each Acomys dataset
    print("\nStep 3: Applying harmonisation to each Acomys dataset...")
    converted_datasets = {}
    
    for aco_prefix in acomys_datasets:
        print(f"\nProcessing {aco_prefix}...")
        aco_adata = datasets[aco_prefix].copy()
        acomys_genes_in_dataset = aco_adata.var_names
        
        # Find which genes are present in this dataset
        present_acomys_genes = [g for g in acomys_genes_in_dataset if g in harmonise_mapping]
        
        # Filter to genes
        aco_adata_filtered = aco_adata[:, present_acomys_genes].copy()
        
        # Rename genes to mouse names
        new_gene_names = [harmonise_mapping[gene] for gene in aco_adata_filtered.var_names]
        aco_adata_filtered.var_names = new_gene_names
        
        # Make gene names unique in case of duplicates
        aco_adata_filtered.var_names_unique = True
        
        print(f"  Original genes: {len(acomys_genes_in_dataset)}")
        print(f"  Genes with harmonised names: {len(present_acomys_genes)}")
        print(f"  Final dataset: {aco_adata_filtered.n_obs} cells × {aco_adata_filtered.n_vars} genes")
        
        # Now ensure this dataset has harmonised genes (fill missing with zeros)
        current_genes = set(aco_adata_filtered.var_names)
        missing_genes = set(harmonised_genes_mouse_names) - current_genes
        
        if len(missing_genes) > 0:
            print(f"  Adding {len(missing_genes)} missing harmonised genes with zeros")
            
            # Create zero matrix for missing genes
            import scipy.sparse as sp
            n_cells = aco_adata_filtered.n_obs
            n_missing = len(missing_genes)
            zero_matrix = sp.csr_matrix((n_cells, n_missing))
            
            # Ensure the existing matrix is in CSR format
            if not sp.issparse(aco_adata_filtered.X):
                existing_matrix = sp.csr_matrix(aco_adata_filtered.X)
            else:
                existing_matrix = aco_adata_filtered.X.tocsr()
            
            # Combine existing data with zeros
            combined_matrix = sp.hstack([existing_matrix, zero_matrix], format='csr')
            
            # Create new var dataframe
            new_var = aco_adata_filtered.var.copy()
            missing_var = pd.DataFrame(index=list(missing_genes))
            for col in new_var.columns:
                missing_var[col] = 0 if new_var[col].dtype in ['int64', 'float64'] else False
            
            combined_var = pd.concat([new_var, missing_var])
            
            # Create unified dataset
            from anndata import AnnData
            aco_adata_unified = AnnData(
                X=combined_matrix,
                obs=aco_adata_filtered.obs.copy(),
                var=combined_var
            )
            
            # Reorder genes to match harmonised_genes_mouse_names order
            gene_order_indices = [list(aco_adata_unified.var_names).index(gene) for gene in harmonised_genes_mouse_names]
            aco_adata_unified = aco_adata_unified[:, gene_order_indices].copy()
        else:
            print(f"  No missing genes - reordering to standard order")
            gene_order_indices = [list(aco_adata_filtered.var_names).index(gene) for gene in harmonised_genes_mouse_names if gene in aco_adata_filtered.var_names]
            aco_adata_unified = aco_adata_filtered[:, gene_order_indices].copy()
        
        converted_datasets[aco_prefix] = aco_adata_unified
        print(f"  Unified dataset: {aco_adata_unified.n_obs} cells × {aco_adata_unified.n_vars} genes")
        
        # Show some example conversions
        examples = [(k, v) for k, v in harmonise_mapping.items() if k in present_acomys_genes][:5]
        if examples:
            print(f"  Example conversions: {examples}")
    
    return converted_datasets, harmonised_genes_mouse_names

# Apply conversion to Acomys datasets using unified approach
print("Converting Acomys datasets to mouse gene names using unified approach...")
converted_acomys_datasets, final_harmonised_genes = verify_genes(acomys_datasets, mus_datasets, datasets)

# Update the main datasets dictionary with converted Acomys datasets
for prefix, converted_adata in converted_acomys_datasets.items():
    datasets[prefix] = converted_adata

print(f"\nConversion complete! All Acomys datasets now use mouse gene names.")
print(f"All Acomys datasets have {len(final_harmonised_genes)} harmonised genes.")

# %%
# Verify the harmonisation conversion results
print("HARMONISATION CONVERSION SUMMARY")
print("=" * 50)

# Check final gene counts for all datasets
all_genes_post_conversion = set()
for prefix, adata in datasets.items():
    all_genes_post_conversion.update(adata.var_names)
    print(f"{prefix}: {adata.n_obs} cells × {adata.n_vars} genes")

print(f"\nTotal unique genes across all datasets (post-conversion): {len(all_genes_post_conversion)}")

# Compare gene overlap between species after conversion
mus_genes_post = set()
aco_genes_post = set()

for prefix in mus_datasets:
    mus_genes_post.update(datasets[prefix].var_names)

for prefix in acomys_datasets:
    aco_genes_post.update(datasets[prefix].var_names)

common_genes_post = mus_genes_post.intersection(aco_genes_post)

print(f"\nGene overlap after conversion:")
print(f"Mus genes: {len(mus_genes_post)}")
print(f"Acomys genes (converted): {len(aco_genes_post)}")
print(f"Common genes: {len(common_genes_post)}")
print(f"Overlap percentage: {len(common_genes_post)/len(mus_genes_post)*100:.1f}%")

# Show some example common genes
print(f"\nExample common genes: {list(common_genes_post)[:10]}")

# Prepare for merging - all datasets now have compatible gene names
print(f"\n✓ All datasets ready for merging!")
print(f"✓ Acomys datasets converted to mouse gene names")
print(f"✓ {len(common_genes_post)} genes available for cross-species analysis")

# %%
# Filter the Mouse datasets to contain same genes as Acomys (harmonised genes only)
print("Filtering Mouse datasets to match Acomys gene set...")

# Use the final harmonised genes from the unified conversion
shared_genes = set(final_harmonised_genes)
print(f"Harmonised genes to match: {len(shared_genes)}")

# Filter each Mouse dataset to contain only the harmonised genes
for mus_prefix in mus_datasets:
    print(f"\nFiltering {mus_prefix}...")
    mus_adata = datasets[mus_prefix]
    
    print(f"  Original: {mus_adata.n_obs} cells × {mus_adata.n_vars} genes")
    
    # Step 1: Find genes that are both in mouse dataset and in harmonised gene set
    available_shared_genes = list(set(mus_adata.var_names).intersection(shared_genes))
    
    # Filter to shared genes
    mus_adata_filtered = mus_adata[:, available_shared_genes].copy()
    
    print(f"  After filtering: {mus_adata_filtered.n_obs} cells × {mus_adata_filtered.n_vars} genes")
    print(f"  Genes retained: {len(available_shared_genes)}/{len(shared_genes)} ({len(available_shared_genes)/len(shared_genes)*100:.1f}%)")
    
    # Step 2: Add missing harmonised genes with zeros
    current_genes = set(mus_adata_filtered.var_names)
    missing_genes = shared_genes - current_genes
    
    if len(missing_genes) > 0:
        print(f"  Adding {len(missing_genes)} missing harmonised genes with zeros")
        
        # Create zero matrix for missing genes
        import scipy.sparse as sp
        n_cells = mus_adata_filtered.n_obs
        n_missing = len(missing_genes)
        zero_matrix = sp.csr_matrix((n_cells, n_missing))
        
        # Ensure the existing matrix is in CSR format
        if not sp.issparse(mus_adata_filtered.X):
            existing_matrix = sp.csr_matrix(mus_adata_filtered.X)
        else:
            existing_matrix = mus_adata_filtered.X.tocsr()
        
        # Combine existing data with zeros
        combined_matrix = sp.hstack([existing_matrix, zero_matrix], format='csr')
        
        # Create new var dataframe
        new_var = mus_adata_filtered.var.copy()
        missing_var = pd.DataFrame(index=list(missing_genes))
        for col in new_var.columns:
            missing_var[col] = 0 if new_var[col].dtype in ['int64', 'float64'] else False
        
        combined_var = pd.concat([new_var, missing_var])
        
        # Create unified dataset
        from anndata import AnnData
        mus_adata_unified = AnnData(
            X=combined_matrix,
            obs=mus_adata_filtered.obs.copy(),
            var=combined_var
        )
        
        # Reorder genes to match final_harmonised_genes order
        gene_order_indices = [list(mus_adata_unified.var_names).index(gene) for gene in final_harmonised_genes]
        mus_adata_unified = mus_adata_unified[:, gene_order_indices].copy()
    else:
        print(f"  No missing genes - reordering to standard order")
        gene_order_indices = [list(mus_adata_filtered.var_names).index(gene) for gene in final_harmonised_genes if gene in mus_adata_filtered.var_names]
        mus_adata_unified = mus_adata_filtered[:, gene_order_indices].copy()
    
    print(f"  Final unified: {mus_adata_unified.n_obs} cells × {mus_adata_unified.n_vars} genes")
    
    # Update the datasets dictionary
    datasets[mus_prefix] = mus_adata_unified
    
print(f"\n✓ All Mouse datasets now have {len(final_harmonised_genes)} genes (with zeros for missing genes)")

# Verify that all datasets now have the same gene set
print("\nFinal verification - Gene counts per dataset:")
for prefix, adata in datasets.items():
    print(f"{prefix}: {adata.n_obs} cells × {adata.n_vars} genes")

# Check gene consistency
all_gene_sets = []
for prefix, adata in datasets.items():
    all_gene_sets.append(set(adata.var_names))

# Check if all datasets have identical gene sets
genes_identical = all([gene_set == all_gene_sets[0] for gene_set in all_gene_sets])

print(f"\n✓ All datasets have identical gene sets: {genes_identical}")
if genes_identical:
    print(f"✓ Ready for merging with {len(all_gene_sets[0])} common genes")
else:
    print("⚠ Warning: Gene sets are not identical across datasets")
    # Show which datasets have different gene counts
    for i, (prefix, gene_set) in enumerate(zip(datasets.keys(), all_gene_sets)):
        if gene_set != all_gene_sets[0]:
            print(f"  {prefix} has {len(gene_set)} genes (expected {len(all_gene_sets[0])})")

# %% [markdown]
# # normalization and log transform

# %%
import os
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.3'

import rpy2.rinterface_lib.callbacks
import logging
from rpy2.robjects import pandas2ri
import anndata2ri
# Ignore R warning messages
#Note: this can be commented out to get more verbose R output
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

# Automatically convert rpy2 outputs to pandas dataframes
pandas2ri.activate()
anndata2ri.activate()
%load_ext rpy2.ipython

# %%
from scipy.sparse import csr_matrix
import numpy as np

for prefix, adata in datasets.items():
    print(f"Processing {prefix}...")

    # Store raw counts in a layer
    adata.layers["counts"] = adata.X.copy()

    # Preprocess for clustering
    adata_pp = adata.copy()
    sc.pp.normalize_total(adata_pp)
    sc.pp.log1p(adata_pp)
    sc.pp.pca(adata_pp, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(adata_pp)
    sc.tl.leiden(adata_pp, resolution=1, key_added="groups")

    # Extract clustering groups and transpose counts for scran
    input_groups = adata_pp.obs['groups'].astype(str)  # ensure string labels
    data_mat = csr_matrix(adata.layers["counts"].T, dtype=np.float32)

    # Pass variables to R
    %R -i data_mat -i input_groups

    # Run scran in R
    %R library(scran)
    %R library(Matrix)
    %R library(MatrixExtra)
    %R data_mat <- as.csc.matrix(data_mat)
    %R sce <- SingleCellExperiment::SingleCellExperiment(list(counts=data_mat))
    %R size_factors <- sizeFactors(computeSumFactors(sce, clusters=input_groups, min.mean=0.1))
    %R -o size_factors
    # Back in Python: store size factors
    adata.obs['size_factors'] = size_factors

    print(f"Finished {prefix}")


# %%
del adata_pp, size_factors, data_mat, input_groups  # Clean up to free memory

for prefix, adata in datasets.items():
    print(f"Processing {prefix}...")
    # add species and time factor into the adata by extracting the name
    temp = prefix.split('_')[-1]
    # split the First 3 characters
    adata.obs['species'] = temp[:3]
    adata.obs['time'] = temp[3:]

    # normalize with the computed size factors
    adata.X /= adata.obs['size_factors'].values[:,None]
    sc.pp.log1p(adata)

    # store in raw
    adata.raw = adata

    print(f"Finished processing {prefix}.")


# %%
# Compute the cell cycle effect
# Use relative paths for CSV files
s_genes = pd.read_csv("cc_genes_s.csv", header=0, index_col=0)
s_genes = s_genes.values.tolist()
s_genes = [item for sublist in s_genes for item in sublist]

# Get gene names from any dataset (they all have the same genes now)
sample_adata = list(datasets.values())[0]

s_genes2 = []
for x in s_genes:
    if x in sample_adata.var_names:
        s_genes2.append(x)
s_genes = s_genes2

# read the g2m phase genes from csv files 
g2m_genes = pd.read_csv("cc_genes_g2m.csv", header=0, index_col=0)
g2m_genes = g2m_genes.values.tolist()
g2m_genes = [item for sublist in g2m_genes for item in sublist]

g2m_genes2 = []
for x in g2m_genes:
    if x in sample_adata.var_names:
        g2m_genes2.append(x)
g2m_genes = g2m_genes2

# combine the s phase and g2m phase genes that appears in the dataset
cc_genes = s_genes + g2m_genes
print(f"Found {len(s_genes)} S-phase genes and {len(g2m_genes)} G2M-phase genes")

for prefix, adata in datasets.items():
    print(f"Processing {prefix}...")

    adata_pp = adata.copy()
    sc.tl.score_genes_cell_cycle(adata_pp, s_genes=s_genes, g2m_genes=g2m_genes)
    adata_pp.obs["phase"] = adata_pp.obs["phase"].astype("category")
    # save the s_score, g2m_score and phase_pred
    adata.obs["S_score"] = adata_pp.obs["S_score"].values
    adata.obs["G2M_score"] = adata_pp.obs["G2M_score"].values
    adata.obs["phase"] = adata_pp.obs["phase"].values
    print(f"Finished {prefix}.")
del adata_pp

# %% [markdown]
# # HVG and visualization

# %%

# compute the highly variable genes after normalization/log1p but without scaling the data
for prefix, adata in datasets.items():
    print(f"Processing {prefix}...")
    sc.pp.highly_variable_genes(adata,  flavor='cell_ranger', n_top_genes=4000)
    print('\n','Number of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))
    # plot the highly variable genes
    sc.pl.highly_variable_genes(adata, log=False)

    # calculate pca
    sc.pp.pca(adata, n_comps=50, svd_solver='arpack', use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_pca', n_pcs=30)
    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, resolution=1.0, key_added='group')
    sc.pl.umap(adata, color=['phase', 'n_genes', 'total_counts','group'], frameon=False, wspace=0.4, hspace=0.4)

# %% [markdown]
# # Integrate datasets

# %%
# Concatenate all datasets for integration
print("Concatenating all datasets...")

# Create a list of datasets for concatenation
adata_list = []
batch_labels = []

for prefix, adata in datasets.items():
    print(f"Adding {prefix}: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Make a copy to avoid modifying original data
    adata_copy = adata.copy()
    
    # Change X to counts layer (raw counts) before concatenation
    print(f"  Changing X from normalized to counts layer...")
    adata_copy.X = adata_copy.layers['counts'].copy()
    
    # Add batch information
    adata_copy.obs['batch'] = prefix
    adata_copy.obs['dataset'] = prefix
    
    adata_list.append(adata_copy)
    batch_labels.append(prefix)

# Concatenate all datasets
adata_combined = sc.concat(adata_list, join='outer', index_unique='_')

print(f"\nCombined dataset: {adata_combined.n_obs} cells × {adata_combined.n_vars} genes")
print(f"Batches: {adata_combined.obs['batch'].unique()}")

# Add some metadata summaries
print(f"Species distribution:")
print(adata_combined.obs['species'].value_counts())
print(f"\nTime distribution:")
print(adata_combined.obs['time'].value_counts())

# Verify that X now contains counts (should be integers)
print(f"\nVerification - X data type: {adata_combined.X.dtype}")
print(f"X contains counts (should be mostly integers): min={adata_combined.X.min():.1f}, max={adata_combined.X.max():.1f}")

# Save the combined dataset
print("\nSaving combined dataset...")
adata_combined.write("adata_combined_all.h5ad")
print("✓ Saved as 'adata_combined_all.h5ad'")

# %%
# Preprocess for clustering
adata_pp = adata_combined.copy()
sc.pp.normalize_total(adata_pp)
sc.pp.log1p(adata_pp)
sc.pp.pca(adata_pp, n_comps=50, svd_solver="arpack")
sc.pp.neighbors(adata_pp)
sc.tl.leiden(adata_pp, resolution=1, key_added="groups_combined")

# Extract clustering groups and transpose counts for scran
input_groups = adata_pp.obs['groups_combined'].astype(str)  # ensure string labels
data_mat = csr_matrix(adata_combined.layers["counts"].T, dtype=np.float32)

# Pass variables to R
%R -i data_mat -i input_groups

# Run scran in R
%R library(scran)
%R library(Matrix)
%R library(MatrixExtra)
%R data_mat <- as.csc.matrix(data_mat)
%R sce <- SingleCellExperiment::SingleCellExperiment(list(counts=data_mat))
%R size_factors <- sizeFactors(computeSumFactors(sce, clusters=input_groups, min.mean=0.1))
%R -o size_factors
# Back in Python: store size factors
adata_combined.obs['size_factors'] = size_factors

# %%
# use the size factor to normalize the counts
adata_combined.X /= adata_combined.obs['size_factors'].values[:, None]
sc.pp.log1p(adata_combined)

# %%
adata_combined.to_df()

# %%
# Store raw data
adata_combined.raw = adata_combined

# Plot UMAP before integration to see batch effects
print("Generating UMAP before integration...")
# Calculate HVGs, PCA, neighbors, and UMAP for the combined dataset
sc.pp.highly_variable_genes(adata_combined, flavor='cell_ranger', n_top_genes=2000, batch_key='batch')
print(f"Number of highly variable genes: {np.sum(adata_combined.var['highly_variable'])}")



# save the adata_combined_all.h5ad
adata_combined.write("adata_combined_all.h5ad")

# Keep only highly variable genes for downstream analysis
adata_combined = adata_combined[:, adata_combined.var.highly_variable]

# PCA
sc.tl.pca(adata_combined, svd_solver='arpack', n_comps=50)

# Neighbors and UMAP
sc.pp.neighbors(adata_combined, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata_combined, min_dist=0.3)


# %%
adata_combined.obs['batch'].str.split('_').str[1].str[:3]

# %%
# update species with batch info
adata_combined.obs['species'] = adata_combined.obs['batch'].str.split('_').str[1].str[:3]
# update the time information with batch info 
adata_combined.obs['time'] = adata_combined.obs['batch'].str.split('_').str[1].str[3:].astype(int)

# %%
# Plot UMAP before integration
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('UMAP Before Integration', fontsize=16)

# Plot by batch
sc.pl.umap(adata_combined, color='batch', ax=axes[0, 0], show=False, frameon=False)
axes[0, 0].set_title('Batch')

# Plot by species
sc.pl.umap(adata_combined, color='species', ax=axes[0, 1], show=False, frameon=False)
axes[0, 1].set_title('Species')

# Plot by time
sc.pl.umap(adata_combined, color='time', ax=axes[0, 2], show=False, frameon=False)
axes[0, 2].set_title('Time')

# Plot by cell cycle phase
sc.pl.umap(adata_combined, color='phase', ax=axes[1, 0], show=False, frameon=False)
axes[1, 0].set_title('Cell Cycle Phase')

# Plot by total genes
sc.pl.umap(adata_combined, color='n_genes', ax=axes[1, 1], show=False, frameon=False)
axes[1, 1].set_title('Number of Genes')

# Plot by total counts
sc.pl.umap(adata_combined, color='total_counts', ax=axes[1, 2], show=False, frameon=False)
axes[1, 2].set_title('Total Counts')

plt.tight_layout()
plt.show()

print("UMAP before integration completed")

# %%
# save the subset adata
adata_combined.write("adata_combined_subset.h5ad")

# %% [markdown]
# # Integration with SCVI

# %%
# load the subset adata 
adata_combined = sc.read("adata_combined_subset.h5ad")

# %%
adata_combined = adata_combined.copy()

# %%
scvi.model.SCVI.setup_anndata(
    adata_combined,
    layer = "counts",
    batch_key="batch"
)

# %%
model = scvi.model.SCVI(adata_combined, n_latent=30, n_layers=2, gene_likelihood="nb")

# %%
model.train(max_epochs=800, 
            early_stopping=True, 
            check_val_every_n_epoch=5, 
            early_stopping_patience=20, 
            early_stopping_monitor='elbo_validation')

# %%
# save the model 
model.save("models/HA_Acomys_MM_GSE182141", overwrite=True)

# %%
# Load the model
model = scvi.model.SCVI.load("models/HA_Acomys_MM_GSE182141", adata = adata_combined)

# %%
# read the embeddings and save to the adata_all
SCVI_LATENT_KEY = "X_scVI"
adata_combined.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

# %%
if "X_diffmap" in adata_combined.obsm.keys():
    del adata_combined.obsm["X_diffmap"]
sc.pp.neighbors(adata_combined, use_rep=SCVI_LATENT_KEY,
                n_neighbors=30)
sc.tl.leiden(adata_combined, key_added="scvi_leiden10", resolution=1.0)
sc.tl.leiden(adata_combined, key_added="scvi_leiden08", resolution=0.8)
sc.tl.leiden(adata_combined, key_added="scvi_leiden05", resolution=0.5)
sc.tl.leiden(adata_combined, key_added="scvi_leiden04", resolution=0.4)
sc.tl.leiden(adata_combined, key_added="scvi_leiden03", resolution=0.3)
sc.tl.leiden(adata_combined, key_added="scvi_leiden02", resolution=0.2)

# %%
SCVI_MDE_KEY = "X_scVI_MDE"
adata_combined.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(adata_combined.obsm[SCVI_LATENT_KEY])
SCVI_NORMALIZED_KEY = "X_scVI_normalizedCounts"
adata_combined.layers[SCVI_NORMALIZED_KEY] = model.get_normalized_expression(adata_combined, library_size=10e4)

# %%
# define the current X_pca and X_umap
adata_combined.obsm['X_pca'] = adata_combined.obsm['X_scVI']
adata_combined.obsm['X_umap'] = adata_combined.obsm['X_scVI_MDE']

# %%
# save the adata
# adata_combined.write("adata_combined_subset.h5ad") # obsolete
adata_combined.write("adata_combined_final_20250923.h5ad")

# %% [markdown]
# # Downstream analysis

# %%
# load the adata
# adata_combined = sc.read("adata_combined_subset.h5ad") # obsolete
adata_combined = sc.read("adata_combined_final_20250923.h5ad")      

# %%
adata_combined.obs.columns

# %% [markdown]
# ### Identify the fibroblast in the combined dataset

# %%
# create plots for visualization
# plot umap after integration
sc.pl.umap(adata_combined, color=['batch', 'species', 'time', 'phase', 'n_genes', 'total_counts', 'scvi_leiden10'], 
           frameon=False)
sc.pl.umap(adata_combined, color=['Lum','Dcn', 'Pdgfra',  # fibroblast markers
                                  'Ltf','Il1b', 'Cd68', # macrophage markers
                                  'Krt5', 'Krt17', # keratinocyte markers
                                  'Cdh5','Pecam1', # endothelial markers
                                  'Cd3e', 'Cd3g','Cd8a', # T cell markers
                                  'Sox9','Col2a1', 'Acan', # chondrocyte markers
                                  'Rgs5', # pericyte markers
                                  'Runx2', 'Sp7', # osteoblast markers
                                  'Prox1','Lyve1', # lymphatic endothelial markers
                                    'Mki67','Top2a', # proliferation markers
                                    'Scn7a','Plp1', # Schwann cell markers

                                  ], frameon=False, color_map='Reds', size=20)
sc.pl.umap(adata_combined, color=['scvi_leiden10'], frameon=False, size=20, legend_loc='on data')

# %%
# get the marker genes for each cluster 
sc.tl.rank_genes_groups(adata_combined, groupby='scvi_leiden10', method='wilcoxon', n_genes=20, raw=True)
# plot the marker genes
sc.pl.rank_genes_groups(adata_combined, n_genes=20, sharey=False)

# %%
# annotate the clusters based on marker genes
adata_combined.obs['celltype'] = adata_combined.obs['scvi_leiden10'].copy()
adata_combined.obs['celltype'] = adata_combined.obs['celltype'].astype(str)
adata_combined.obs['celltype'] = adata_combined.obs['celltype'].replace({
    '0': 'Macrophages',
    '1': 'Keratinocytes',
    '2': 'Keratinocytes',
    '3': 'Keratinocytes',
    '4': 'Keratinocytes',
    '5': 'Keratinocytes',
    '6': 'Keratinocytes',
    '7': 'Macrophages',
    '8': 'Keratinocytes',
    '9': 'Keratinocytes',
    '10': 'Endothelial cells',
    '11': 'Keratinocytes',
    '12': 'Keratinocytes',
    '13': 'Keratinocytes',
    '14': 'Keratinocytes',
    '15': 'T cells',
    '16': 'Fibroblasts',
    '17': 'Chondroblasts',
    '18': 'Macrophages',
    '19': 'Keratinocytes',
    '20': 'Lymphatic endothelial cells',
    '21': 'Endothelial cells',
    '22': 'Keratinocytes',
    '23': 'Pericytes',
    '24': 'Remove',
    '25': 'Remove',
    '26': 'Remove',
})


# %%
# remove the clusters that are labeled as 'Remove'
adata_combined = adata_combined[adata_combined.obs['celltype'] != 'Remove'].copy()

# %%
# define the order of cell types
celltype_order = [
    'Endothelial cells',
    'Fibroblasts',
    'Keratinocytes',
    'Lymphatic endothelial cells',
    'Macrophages',
    'Chondroblasts',
    'Pericytes',
    'T cells',   
]
adata_combined.obs['celltype'] = pd.Categorical(adata_combined.obs['celltype'], categories=celltype_order, ordered=True)

# %%
# plot the umap with cell type annotation
sc.pl.umap(adata_combined, color=['celltype', 'scvi_leiden10', 'species'], ncols=1,
           frameon=False, size=10, save="_AcoMus_metadata_20251006.pdf")


# %%
# show the cell counts for each cell type
adata_combined.obs['celltype'].value_counts()

# %%
# save the adata_combined before overwritting
adata_combined.write("adata_combined_final_20250925.h5ad")

# %% [markdown]
# # Downstream analysis for fibroblast

# %%
# load the adata_combined
adata_combined = sc.read("adata_combined_final_20250925.h5ad")
adata_combined

# %%
# load the adata_combined with all counts without subset
adata_combined_all = sc.read("adata_combined_all.h5ad")
adata_combined_all

# %%
adata_combined.to_df()

# %%
adata_combined_all.to_df()

# %%
# match the cell barcodes in adata_combined to adata_combined_all
adata_combined_all = adata_combined_all[adata_combined.obs_names].copy()
# transfer the obs, obsm, uns from adata_combined to adata_combined_all
adata_combined_all.obs = adata_combined.obs.copy()
adata_combined_all.obsm = adata_combined.obsm.copy()
adata_combined_all.uns = adata_combined.uns.copy()

# %%
# get the fibroblast subset from adata_combined_all, 
adata_fibro = adata_combined_all[adata_combined_all.obs['celltype'] == 'Fibroblasts'].copy()
adata_fibro

# %%
adata_fibro.to_df()

# %%
# save the fibroblast subset
adata_fibro.write("adata_fibroblasts_raw_20250925.h5ad")

# %%
from scipy.sparse import csr_matrix

# Preprocess for clustering
adata_pp = adata_fibro.copy()
sc.pp.normalize_total(adata_pp)
sc.pp.log1p(adata_pp)
sc.pp.pca(adata_pp, n_comps=50, svd_solver="arpack")
sc.pp.neighbors(adata_pp)
sc.tl.leiden(adata_pp, resolution=1, key_added="groups_fibro")

# Extract clustering groups and transpose counts for scran
input_groups = adata_pp.obs['groups_fibro'].astype(str)  # ensure string labels
data_mat = csr_matrix(adata_fibro.layers["counts"].T, dtype=np.float32)

# Pass variables to R
%R -i data_mat -i input_groups

# Run scran in R
%R library(scran)
%R library(Matrix)
%R library(MatrixExtra)
%R data_mat <- as.csc.matrix(data_mat)
%R sce <- SingleCellExperiment::SingleCellExperiment(list(counts=data_mat))
%R size_factors <- sizeFactors(computeSumFactors(sce, clusters=input_groups, min.mean=0.1))
%R -o size_factors

# Back in Python: store size factors
adata_fibro.obs['size_factors'] = size_factors

# %%
# use the size factor to normalize the counts
adata_fibro.X /= adata_fibro.obs['size_factors'].values[:, None]
sc.pp.log1p(adata_fibro)

# %%
adata_fibro.to_df()

# %%
adata_fibro.raw.to_adata()

# %%
# store this in raw 
adata_fibro.raw = adata_fibro

# %%
# plot the umap
sc.pp.highly_variable_genes(adata_fibro, flavor='cell_ranger', n_top_genes=2000)
print(f"Number of highly variable genes: {np.sum(adata_fibro.var['highly_variable'])}")
# plot the highly variable genes
sc.pl.highly_variable_genes(adata_fibro, log=False)

# %%
# save this adata_fibro
adata_fibro.write("adata_fibroblasts_20250925.h5ad")

# %%
# load the adata_fibro
adata_fibro = sc.read("adata_fibroblasts_20250925.h5ad")
adata_fibro

# %%
# Calculate HVGs, PCA, neighbors, and UMAP for the fibroblast dataset
sc.pp.highly_variable_genes(adata_fibro, flavor='cell_ranger', n_top_genes=2000)
# subset to HVGs
adata_fibro = adata_fibro[:, adata_fibro.var.highly_variable].copy()
# save the adata_fibro
adata_fibro.write("adata_fibroblasts_hvg_20250925.h5ad")

# %%
# PCA
sc.pp.pca(adata_fibro, n_comps=50, svd_solver="arpack")
sc.pp.neighbors(adata_fibro, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata_fibro, min_dist=0.3, method="umap")


# %%
#plot the umap
sc.pl.umap(adata_fibro, color=["batch", "species"], frameon=False)

# %%
# value counts for each batch and species
print(adata_fibro.obs['batch'].value_counts())
print(adata_fibro.obs['species'].value_counts())

# %%
# save the adata_fibro
adata_fibro.write("adata_fibroblasts_hvg_20250925.h5ad")

# %% [markdown]
# # integration of Fibroblast with scVI 

# %%
# load the adata_fibro hvg subset
adata_fibro = sc.read("adata_fibroblasts_hvg_20250925.h5ad")
adata_fibro

# %%
adata_fibro = adata_fibro.copy()

# %%
scvi.model.SCVI.setup_anndata(
    adata_fibro,
    layer = "counts",
    batch_key='batch',
    labels_key='species' # Do i need this?
)

# %%
model = scvi.model.SCVI(adata_fibro, n_latent=30, n_layers=2, gene_likelihood="nb")

# %%
model.train(max_epochs=800,
            early_stopping=True, 
            check_val_every_n_epoch=5, 
            early_stopping_patience=20, 
            early_stopping_monitor='elbo_validation')

# %%
# save the model 
model.save("models/HA_Acomys_MM_GSE182141_fibro", overwrite=True)

# %%
# load the model 
model = scvi.model.SCVI.load("models/HA_Acomys_MM_GSE182141_fibro", adata = adata_fibro)

# %%
# read the embeddings and save to the adata_fibro
SCVI_LATENT_KEY = "X_scVI"
adata_fibro.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

# %%
if "X_diffmap" in adata_fibro.obsm.keys():
    del adata_fibro.obsm["X_diffmap"]
SCVI_LATENT_KEY = "X_scVI"
sc.pp.neighbors(adata_fibro, use_rep=SCVI_LATENT_KEY,
                n_neighbors=30)
sc.tl.leiden(adata_fibro, key_added="scvi_fib_leiden10", resolution=1.0)
sc.tl.leiden(adata_fibro, key_added="scvi_fib_leiden08", resolution=0.8)
sc.tl.leiden(adata_fibro, key_added="scvi_fib_leiden05", resolution=0.5)
sc.tl.leiden(adata_fibro, key_added="scvi_fib_leiden04", resolution=0.4)
sc.tl.leiden(adata_fibro, key_added="scvi_fib_leiden03", resolution=0.3)
sc.tl.leiden(adata_fibro, key_added="scvi_fib_leiden02", resolution=0.2)

# %%
SCVI_MDE_KEY = "X_scVI_MDE"
adata_fibro.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(adata_fibro.obsm[SCVI_LATENT_KEY])

# %%
SCVI_NORMALIZED_KEY = "X_scVI_normalizedCounts" 
adata_fibro.layers[SCVI_NORMALIZED_KEY] = model.get_normalized_expression(adata_fibro, library_size=10e4)

# %%
# define the current X_pca and X_umap
adata_fibro.obsm['X_pca'] = adata_fibro.obsm['X_scVI']
adata_fibro.obsm['X_umap'] = adata_fibro.obsm['X_scVI_MDE']

# %%
# save the adata_fibro hvg
adata_fibro.write("adata_fibroblasts_hvg_20250925.h5ad")

# %% [markdown]
# # Downstream for plots 

# %%
# load the adata_fibro hvg
adata_fibro = sc.read("adata_fibroblasts_hvg_20250925.h5ad")
adata_fibro

# %%
# make the time a category 
adata_fibro.obs['time'] = adata_fibro.obs['time'].astype('category')

# %%
# plot the umap
sc.pl.umap(adata_fibro, color=['species', 'scvi_leiden10', 'scvi_fib_leiden02','time','batch'], 
           frameon=False)
# plot the umap of each species separately on time 
sc.pl.umap(adata_fibro[adata_fibro.obs['species'] == 'Mus'], color=['time', 'scvi_fib_leiden02'], 
           frameon=False, title='Mouse Fibroblasts')
sc.pl.umap(adata_fibro[adata_fibro.obs['species'] == 'Aco'], color=['time', 'scvi_fib_leiden02'], 
           frameon=False, title='Acomys Fibroblasts')

# %%
# make a new category for time, based on days. 3 and 5 are inflammatory and 10 and 15 are repair
adata_fibro.obs['time_group'] = adata_fibro.obs['time'].astype(str)
adata_fibro.obs['time_group'] = adata_fibro.obs['time_group'].replace({
    '0': 'Uninjured',
    '3': 'Inflammatory',
    '5': 'Inflammatory',
    '10': 'Repair',
    '15': 'Repair'
})
adata_fibro.obs['time_group'] = adata_fibro.obs['time_group'].astype('category')
# plot the umap with time group
sc.pl.umap(adata_fibro, color=['time_group', 'scvi_fib_leiden02'], 
           frameon=False)

# %%
# create a new category for species and time group combined
adata_fibro.obs['species_time'] = adata_fibro.obs['species'].astype(str) + '_' + adata_fibro.obs['time_group'].astype(str)
adata_fibro.obs['species_time'] = adata_fibro.obs['species_time'].astype('category')
# put the species_time in a specific order
species_time_order = ['Aco_Uninjured', 'Aco_Inflammatory', 'Aco_Repair',
                      'Mus_Uninjured', 'Mus_Inflammatory', 'Mus_Repair']
adata_fibro.obs['species_time'] = pd.Categorical(adata_fibro.obs['species_time'], categories=species_time_order, ordered=True)
# put the time_group in a specific order
time_group_order = ['Uninjured', 'Inflammatory', 'Repair']
adata_fibro.obs['time_group'] = pd.Categorical(adata_fibro.obs['time_group'], categories=time_group_order, ordered=True)

# plot the umap with species and time group
sc.pl.umap(adata_fibro, color=['species_time', 'scvi_fib_leiden02'], 
           frameon=False, legend_loc='on data')

# %%
# export the umaps with species, time, batch and time_group
sc.pl.umap(adata_fibro, color=['species', 'time', 'batch', 'time_group'], ncols=1,
           frameon=False, save='_AcoMus_fibro_umap.pdf')

# %%
# create dotplot for some marker genes
marker_genes = ['Hapln1', 'Hapln2', 'Hapln3', 'Hapln4', # hyaluronan and proteoglycan link proteins
                'Cd44', 
                'Has2', 'Has3', # hyaluronan synthase # No Has1 in the dataset
                'Vcan', 'Acan', 'Bcan', 'Ncan', # lecticans
                 # 'Id1', 'Id2', 'Id3', 
                'Tnfaip6', 'Hmmr']
sc.pl.dotplot(adata_fibro, marker_genes, groupby='species_time', standard_scale='var', color_map='Blues', title='Aco & Mus Fibroblasts', dendrogram=False, save='_AcoMus_fibro_HA_marker_genes.svg')

# %%
# create a dotplot for each species separately
sc.pl.dotplot(adata_fibro[adata_fibro.obs['species'] == 'Aco'], marker_genes, groupby='time_group', standard_scale='var', color_map='Blues', dendrogram=False, title='Acomys Fibroblasts', smallest_dot=0.1, save='_Aco_fibro_HA_marker_genes.svg')
sc.pl.dotplot(adata_fibro[adata_fibro.obs['species'] == 'Mus'], marker_genes, groupby='time_group', standard_scale='var', color_map='Blues', dendrogram=False, title='Mouse Fibroblasts', smallest_dot=0.1, save='_Mus_fibro_HA_marker_genes.svg')

# %%
# plot another gene list for fibrosis markers
fibrosis_genes = ['Col1a1', 'Col1a2', 'Col3a1', 'Col5a2', 'Col6a1', 'Col6a2', 'Col6a3',
                   'Fn1', 'Lox', 'Loxl1', 'Loxl2', 'Loxl4',
                   'Mmp2', 'Mmp3', 'Mmp9', 'Mmp13', 
                   'Thbs1', 'Thbs2', 'Thbs3', 'Thbs4', 
                   'Tnc', 'Ctgf']
sc.pl.dotplot(adata_fibro, fibrosis_genes, groupby='species_time', standard_scale='var', color_map='Blues', dendrogram=False, title='Aco & Mus Fibroblasts', save='_AcoMus_fibro_fibrosis_marker_genes.svg')
sc.pl.dotplot(adata_fibro[adata_fibro.obs['species'] == 'Aco'], fibrosis_genes, groupby='time_group', standard_scale='var', color_map='Blues', dendrogram=False, title='Acomys Fibroblasts', smallest_dot=0.1, save='_Aco_fibro_fibrosis_marker_genes.svg')
sc.pl.dotplot(adata_fibro[adata_fibro.obs['species'] == 'Mus'], fibrosis_genes, groupby='time_group', standard_scale='var', color_map='Blues', dendrogram=False, title='Mouse Fibroblasts', smallest_dot=0.1, save='_Mus_fibro_fibrosis_marker_genes.svg')

# %%



