import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from pylab import rcParams
rcParams['figure.figsize'] = 8, 16


X_test = pd.read_csv('models/X_test.csv', index_col = 0)
Y_test = pd.read_csv('models/Y_test.csv', index_col = 0)

shap.initjs()

xg_reg = joblib.load('models/XG_boost.model')


explainer = shap.TreeExplainer(xg_reg)
shap_values = explainer.shap_values(X_test)

# visualize the first prediction's explanation

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("plots/shap_summary_plot.png")