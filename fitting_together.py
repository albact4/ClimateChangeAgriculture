import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, lognorm, gamma, weibull_min
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Read data from excel file
# Update file path accordingly
data = pd.read_excel('/Users/alba/Desktop/ag_census_tracts17.xlsx', sheet_name='ag_census_tracts17')
print("Data Read")

# Specify the columns you want to analyze
input1 = ['av_norm']
input2 = ['b1_pp_tr_m_N100', 'b2_pp_sig_N100', 'b3_pp_ann_N100', 'b5_pp_perr_N100', 'b6_perun_m_N100']
output1 = ['b7_pp_st_1_N100', 'b4_pp_perd_N100', 'b12_lndcod_N100']
input3 = ['ag_t_N100']
current_condition = ['RuCaIndRUR']

zero_handling = ['av_norm', 'b5_pp_perr_N100', 'b4_pp_perd_N100', 'ag_t_N100']

for column in zero_handling:
    data[column] = np.log1p(data[column])
print("Handle zeros")

# Join the arrays into one
columns_to_analyze = input1 + input2 + output1 + input3 + current_condition
print("Columns joined, start loop")

# Dictionary to map column abbreviations to full names when displaying maps
column_names_mapping = {
    'av_norm': 'Pounds of pollution in water normalized',
    'b1_pp_tr_m_N100': 'Productivity_trend',
    'b2_pp_sig_N100': 'Productivity_significance',
    'b3_pp_ann_N100': 'Productivity_ann_mean',
    'b5_pp_perr_N100': 'Productivity_performance_ratio',
    'b6_perun_m_N100': 'Productivity_performance_units',
    'b7_pp_st_1_N100': 'Productivity_state_degradation',
    'b4_pp_perd_N100': 'Productivity_performance_degradation',
    'b12_lndcod_N100': 'Land_cover_degradation',
    'ag_t_N100': 'Total of money expenses',  # Fix this name
    'RuCaIndRUR': 'Rural Capacity Index',
}

# Directory to save the visualizations
output_dir = 'pdf_fitting_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Loop through each column
for column_name in columns_to_analyze:
    try:
    
        # Extract the data column
        data_column = data[column_name]

        # Remove non-finite values (NaN, inf, -inf)
        data_column = data_column.replace([np.inf, -np.inf], np.nan).dropna()

        # Check if there are any missing values after dropping NaN
        if data_column.isnull().values.any():
            print(f"Column {column_name} still contains missing values. Skipping PCA and K-means for this column.")
            continue

        # Fit different PDFs
        fit_normal = norm.fit(data_column)
        fit_exponential = expon.fit(data_column)
        fit_lognormal = lognorm.fit(data_column)
        fit_gamma = gamma.fit(data_column)
        fit_weibull = weibull_min.fit(data_column)

        # Plot the histograms
        plt.hist(data_column, bins=30, density=True, alpha=0.6, color='g')

        # Plot the fitted PDFs
        x = np.linspace(min(data_column), max(data_column), 1000)
        plt.plot(x, norm.pdf(x, *fit_normal), label='Normal', linewidth=2)
        plt.plot(x, expon.pdf(x, *fit_exponential), label='Exponential', linewidth=2)
        plt.plot(x, lognorm.pdf(x, *fit_lognormal), label='Log-Normal', linewidth=2)
        plt.plot(x, gamma.pdf(x, *fit_gamma), label='Gamma', linewidth=2)
        plt.plot(x, weibull_min.pdf(x, *fit_weibull), label='Weibull', linewidth=2)

        # Plot the fitted CDFs
        plt.plot(x, norm.cdf(x, *fit_normal), linestyle='--', label='Normal CDF', linewidth=2)
        plt.plot(x, expon.cdf(x, *fit_exponential), linestyle='--', label='Exponential CDF', linewidth=2)
        plt.plot(x, lognorm.cdf(x, *fit_lognormal), linestyle='--', label='Log-Normal CDF', linewidth=2)
        plt.plot(x, gamma.cdf(x, *fit_gamma), linestyle='--', label='Gamma CDF', linewidth=2)
        plt.plot(x, weibull_min.cdf(x, *fit_weibull), linestyle='--', label='Weibull CDF', linewidth=2)

        # Name with no abbreviation for pdf fitting
        full_name = column_names_mapping.get(column_name, column_name)

        plt.legend()
        plt.title(f'Fitted PDFs and CDFs for {full_name}')
        plt.xlabel(f'{column_name}')
        plt.ylabel('Probability Density / Cumulative Probability')

        # Save the plot to a file with the full name
        plot_filename = os.path.join(output_dir, f'{full_name}_pdf_cdf_fit.png')

        # Save the PDF figure
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to avoid overlapping plots
        print(f"Saved plot as: {plot_filename}")

    except RuntimeWarning as e:
        print(f"Warning in column {column_name}: {e}")

# Extract the relevant numeric columns for scaling and PCA
numeric_data = data[columns_to_analyze]

# Standardize the numeric data for PCA and K-means
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# Perform K-means
kmeans = KMeans(n_clusters=3)
kmeans_result = kmeans.fit_predict(scaled_data)

# Visualize PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_result, cmap='viridis', edgecolor='k', s=50)

# Add labels with full names
for i, txt in enumerate(columns_to_analyze):
    full_name = column_names_mapping.get(txt, txt)
    plt.annotate(full_name, (pca_result[i, 0], pca_result[i, 1]))

plt.title('PCA and K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Save the PCA and K-means clustering plot
plt.savefig(os.path.join(output_dir, 'PCA_KMeans_Clustering.png'))
plt.show()
