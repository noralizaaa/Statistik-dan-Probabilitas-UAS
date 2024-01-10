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

# Fungsi uji simultan regresi
def simultaneous_regression_test(model, variables):
    hypothesis = " + ".join([f"{var}" for var in variables])
    f_test = model.f_test(hypothesis)
    return f_test

# Uji Simultan pada keseluruhan model regresi dan interpretasinya
simultaneous_test_result = simultaneous_regression_test(results, independent_variables)
print("\nUji Simultan pada Keseluruhan Model:")
print(simultaneous_test_result)
print(f'Interpretasi: Keseluruhan model {"tidak" if simultaneous_test_result.pvalue > 0.05 else ""}signifikan terhadap variabel terikat\n')