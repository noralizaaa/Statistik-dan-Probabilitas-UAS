import pandas as pd
import numpy as np

data = pd.read_csv('Cardiovascular Disease Dataset.csv')

# Ambil 100 data
sample_data = data.sample(100)

# Ambil variabel numerik
X = sample_data[['age','gender','chestpain','restingBP', 'serumcholestrol']]
Y = sample_data['target']

# Hitung koefisien regresi menggunakan numpy
coefficients = np.linalg.inv(X.T @ X) @ X.T @ Y
print("Koefisien regresi:\n", coefficients)