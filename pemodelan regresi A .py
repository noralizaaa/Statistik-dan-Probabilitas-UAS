import pandas as pd
import statsmodels.api as sm
import numpy as np

# Membaca data dari file CSV
data = pd.read_csv ('Cardiovascular Disease Dataset.csv')

# Handling Missing Values
data = data.fillna(data.mean())

# Handling Infinity
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.fillna(data.max())

# Memilih 5 kolom numerik sebagai variabel independen
independent_variables = ['age','gender','chestpain','restingBP','serumcholestrol']

# Menambahkan konstanta untuk model regresi
X = sm.add_constant(data[independent_variables])

# Membuat model regresi
model = sm.OLS(data['target'], X)

# Menyesuaikan model
results = model.fit()

# Menampilkan hasil regresi
print(results.summary())

# Fungsi untuk pemodelan regresi
def predict_regression(model, data):
    X_pred = sm.add_constant(data[independent_variables])
    predictions = model.predict(X_pred)
    return predictions

# Menggunakan seluruh baris data untuk prediksi
predictions = predict_regression(results, data)
data['Predicted_target'] = predictions

# Menampilkan hasil prediksi
print("\nPredictions for all data:")
print(data[['age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol', 'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels','Predicted_target']])
