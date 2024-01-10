import pandas as pd
import numpy as np
import statsmodels.api as sm

# Membaca data dari file CSV
data = pd.read_csv ('Cardiovascular Disease Dataset.csv')

# Handling Missing Values
data = data.fillna(data.mean())

# Handling Infinity
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.fillna(data.max())

# Memilih 11 kolom numerik sebagai variabel independen
independent_variables = ['age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']

# Menambahkan konstanta untuk model regresi
X = sm.add_constant(data[independent_variables])

# Membuat model regresi
model = sm.OLS(data['target'], X)

# Menyesuaikan model
results = model.fit()

# Menampilkan hasil regresi
print(results.summary())

# Mengakses nilai R-squared
rsquared = results.rsquared
print(f'\nNilai R-squared: {rsquared}')