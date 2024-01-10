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

# Fungsi uji parsial regresi
def partial_regression_test(model, variable):
    hypothesis = f"{variable} = 0"
    t_test = model.t_test(hypothesis)
    return t_test

# Uji Parsial pada setiap koefisien regresi dan interpretasinya
for var in independent_variables:
    t_test_result = partial_regression_test(results, var)
    print(f"\nUji Parsial untuk Koefisien {var}:")
    print(t_test_result)
    print(f'Interpretasi: Variabel {var} {"tidak " if t_test_result.pvalue > 0.05 else ""}signifikan terhadap variabel terikat')

# Catatan: Nilai ambang batas umumnya adalah 0.05, sehingga variabel dianggap signifikan jika p-value <= 0.05.