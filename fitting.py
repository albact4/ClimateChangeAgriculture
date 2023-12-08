import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, lognorm, gamma, weibull_min, poisson


# Replace 'your_data_file.xlsx' with the actual file path
data = pd.read_excel('C:\\Users\\Rojano\Desktop\\ag_census_tracts17.xlsx', sheet_name='ag_census_tracts17')

print("Data Read")

# Specify the columns you want to analyze
input1 = ['av_norm'] #Correct column name is av_norm_N100 just trying av_norm for now
input2 = ['b1_pp_tr_m_N100', 'b2_pp_sig_N100', 'b3_pp_ann_N100', 'b5_pp_perr_N100' , 'b6_perun_m_N100']
output1 = ['b7_pp_st_1_N100', 'b4_pp_perd_N100', 'b12_lndcod_N100']
input3 = ['ag_t_N100']
current_condition = ['RuCaIndRUR']

# Join the arrays into one
columns_to_analyze = input1 + input2 + output1 + input3 + current_condition
print("Columns joined, start loop")

# Loop through each column
for column_name in columns_to_analyze:
    # Extract the data column
    data_column = data[column_name]

    print("Column Name Read")

    # Remove non-finite values (NaN, inf, -inf)
    data_column = data_column.replace([np.inf, -np.inf], np.nan).dropna()
    print("N/a Values Dropped")

    # Fit different PDFs
    fit_normal = norm.fit(data_column)
    print("Normal Fit")

    fit_exponential = expon.fit(data_column)
    print("Expo Fit")

    fit_lognormal = lognorm.fit(data_column)
    print("Log Norm Fit")

    fit_gamma = gamma.fit(data_column)
    print("Gamma Fit")


    fit_weibull = weibull_min.fit(data_column)
    print("Weibull Fit")
    
    # Fit Poisson distribution
    #fit_poisson_lambda = poisson.fit(data_column)[0]  # Poisson fitting returns a tuple, take the first element as lambda
    #fit_poisson = poisson.pmf(np.arange(min(data_column), max(data_column) + 1), fit_poisson_lambda)
    #print("Poisson Fit")

    # Plot the histograms
    plt.hist(data_column, bins=30, density=True, alpha=0.6, color='g')

    # Plot the fitted PDFs
    x = np.linspace(min(data_column), max(data_column), 1000)
    plt.plot(x, norm.pdf(x, *fit_normal), label='Normal', linewidth=2)
    plt.plot(x, expon.pdf(x, *fit_exponential), label='Exponential', linewidth=2)
    plt.plot(x, lognorm.pdf(x, *fit_lognormal), label='Log-Normal', linewidth=2)
    plt.plot(x, gamma.pdf(x, *fit_gamma), label='Gamma', linewidth=2)
    plt.plot(x, weibull_min.pdf(x, *fit_weibull), label='Weibull', linewidth=2)
    #plt.plot(np.arange(min(data_column), max(data_column) + 1), fit_poisson, label='Poisson', linewidth=2, marker='o')


    plt.legend()
    plt.title(f'Fitted PDFs for {column_name}')
    plt.xlabel(f'{column_name}')
    plt.ylabel('Probability Density')
    plt.show()
