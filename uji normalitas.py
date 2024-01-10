import pandas as pd 
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np 

data = pd.read_csv ('Cardiovascular Disease Dataset.csv')

x = data[['age','gender','chestpain','restingBP','serumcholestrol','fastingbloodsugar','restingrelectro','maxheartrate','exerciseangia','oldpeak','slope','noofmajorvessels']]
y = data[['target']]
  
x = sm.add_constant(x)

model = sm.OLS(y, x).fit()

standardized_residuals = model.get_influence().resid_studentized_internal

shapiro_test_statistic, shapiro_p_value = stats.shapiro(standardized_residuals)
print(f"Shapiro-wilk test statistik: {shapiro_test_statistic}")
print(f"P-value: {shapiro_p_value}")

ks_test_statistic, ks_p_value = stats.kstest(standardized_residuals, 'norm')
print(f"KS Test Statistic: {ks_test_statistic}")
print(f"P-value (KS Test): {ks_p_value}")

plt.subplot(1, 3, 1)
plt.hist(standardized_residuals, bins=20, density=True, alpha=0.5, color='b', edgecolor='black')
plt.title('Histogram Residual Standar dan Distribusi Normal')
plt.xlabel('Residual Standar')
plt.ylabel('Frekuensi')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = stats.norm.pdf(x, np.mean(standardized_residuals), np.std(standardized_residuals))
plt.plot(x, p, 'k', linewidth=2)

plt.subplot(1, 3, 2)
stats.probplot(standardized_residuals, dist="norm", plot=plt)
plt.title('Normal Probability Plot')

plt.subplot(1, 3, 3)
stats.probplot(standardized_residuals, dist="norm", plot=plt, rvalue=True)
plt.title('Kolmogorov-Smirnov Test Plot')

plt.tight_layout()
plt.show()