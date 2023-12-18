import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, lognorm, gamma, weibull_min
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def handle_nan(data, fill_value=None, strategy='mean'):
    """
    Handle NaN values in a DataFrame or NumPy array.

    Parameters:
    - data: DataFrame or NumPy array
    - fill_value: Value to fill NaN entries with (default: None)
    - strategy: Strategy to fill NaN values ('mean', 'median', 'mode', or 'custom') (default: 'mean')

    Returns:
    - DataFrame or NumPy array with NaN values handled
    """

    if isinstance(data, pd.DataFrame):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if strategy == 'mean':
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        elif strategy == 'median':
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        elif strategy == 'mode':
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mode().iloc[0])
        elif strategy == 'custom':
            data[numeric_columns] = data[numeric_columns].fillna(fill_value)
        else:
            raise ValueError("Invalid strategy. Choose 'mean', 'median', 'mode', or 'custom'.")
    elif isinstance(data, np.ndarray):
        # Similar modification for NumPy array
        raise NotImplementedError("Handling NaN for NumPy array needs modification.")
    else:
        raise ValueError("Input must be a DataFrame or NumPy array.")

    return data


##################################################################################################################################

# READ DATA FROM EXCEL
# windows computer in office
# data = pd.read_excel('C:\\Users\\Rojano\Desktop\\ag_census_tracts17.xlsx', sheet_name='ag_census_tracts17')
# MacOS
data = pd.read_excel('/Users/alba/Desktop/ag_census_tracts17.xlsx', sheet_name='ag_census_tracts17')

# Specify the columns you want to analyze
input1 = ['av_norm'] 
input2 = ['b1_pp_tr_m_N100', 'b2_pp_sig_N100', 'b3_pp_ann_N100', 'b5_pp_perr_N100' , 'b6_perun_m_N100']
output1 = ['b7_pp_st_1_N100', 'b4_pp_perd_N100', 'b12_lndcod_N100']
#input3 = ['ag_t_N100']
current_condition = ['RuCaIndRUR']

# Special array for columns that need to be zero-handled
zero_handling = ['b5_pp_perr_N100', 'b4_pp_perd_N100', 'b12_lndcod_N100' ]

# Handle Nan values
data = handle_nan(data, strategy='mean')

# Log transformation for zero handling
for column in zero_handling:
    data[column] = np.log1p(data[column])



# Join the arrays into one
columns_to_analyze = input1 + input2 + output1 + current_condition  # + input3
print(columns_to_analyze)



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
    'RuCaIndRUR': 'Rural Capacity Index',
}

# Directory to save the visualizations
output_dir = 'fitting_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Initialize full_name outside the loop (no need to initialize output_dir again)
full_name = ""

# Loop through each column
for column_name in columns_to_analyze:
    try:
        # Extract the data column
        data_column = data[column_name]

        # Remove non-finite values (NaN, inf, -inf)
        data_column = data_column.replace([np.inf, -np.inf], np.nan).dropna()
        #print(data[columns_to_analyze])

        # Check if there are any missing values after dropping NaN
        if data_column.isnull().values.any():
            print(f"Column {column_name} still contains missing values. Skipping PDF and CDF fitting for this column.")
            continue
        
        # Dynamically determine the range of x based on the data
        x = np.linspace(min(data_column), max(data_column), 1000)

        #Rewrite the full_name
        full_name = column_names_mapping.get(column_name, column_name)

        # Fit different PDFs
        fit_normal = norm.fit(data_column)
        fit_exponential = expon.fit(data_column)
        fit_lognormal = lognorm.fit(data_column)
        fit_gamma = gamma.fit(data_column)
        fit_weibull = weibull_min.fit(data_column)

        # Plot the histograms for PDF
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(data_column, bins=30, density=True, alpha=0.6, color='g')
        plt.plot(x, norm.pdf(x, *fit_normal), label='Normal', linewidth=2)
        plt.plot(x, expon.pdf(x, *fit_exponential), label='Exponential', linewidth=2)
        plt.plot(x, lognorm.pdf(x, *fit_lognormal), label='Log-Normal', linewidth=2)
        plt.plot(x, gamma.pdf(x, *fit_gamma), label='Gamma', linewidth=2)
        plt.plot(x, weibull_min.pdf(x, *fit_weibull), label='Weibull', linewidth=2)
        plt.legend()
        plt.title(f'PDF Fittings for {full_name}')
        plt.xlabel(f'{column_name}')
        plt.ylabel('Probability Density')

        # Plot the CDFs
        plt.subplot(1, 2, 2)
        plt.plot(x, norm.cdf(x, *fit_normal), linestyle='--', label='Normal CDF', linewidth=2)
        plt.plot(x, expon.cdf(x, *fit_exponential), linestyle='--', label='Exponential CDF', linewidth=2)
        plt.plot(x, lognorm.cdf(x, *fit_lognormal), linestyle='--', label='Log-Normal CDF', linewidth=2)
        plt.plot(x, gamma.cdf(x, *fit_gamma), linestyle='--', label='Gamma CDF', linewidth=2)
        plt.plot(x, weibull_min.cdf(x, *fit_weibull), linestyle='--', label='Weibull CDF', linewidth=2)
        plt.legend()
        plt.title(f'CDF Fittings for {full_name}')
        plt.xlabel(f'{column_name}')
        plt.ylabel('Cumulative Probability')

        # Adjust layout
        plt.tight_layout()

        # Save the plots to files with the full names
        pdf_plot_filename = os.path.join(output_dir, f'{full_name}_pdf_fit.png')

        # Save the PDF and CDF figures
        plt.savefig(pdf_plot_filename)

        plt.close()  # Close the figures to avoid overlapping plots
        print(f"Saved plot as: {pdf_plot_filename}")

    except RuntimeWarning as e:
        print(f"Warning in column {column_name}: {e}")


print(data[columns_to_analyze])

# Create a DataFrame with only the columns to analyze
data_to_analyze = data[columns_to_analyze]

# Standardize the data for PCA
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data_to_analyze)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(data_standardized)

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by Principal Components')
plt.grid(True)
plt.show()

# Choose the number of components based on the explained variance
n_components = 2  # Adjust this based on the plot
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(data_standardized)

# Add PCA components to the DataFrame
for i in range(n_components):
    data[f'PCA_Component_{i + 1}'] = pca_result[:, i]

# Perform KMeans clustering
n_clusters = 3  # Adjust this based on your analysis
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_standardized)

# Visualize PCA components by cluster
plt.figure(figsize=(12, 6))
for cluster in range(n_clusters):
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA_Component_1'], cluster_data['PCA_Component_2'], label=f'Cluster {cluster + 1}')

plt.title('PCA Components by Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
