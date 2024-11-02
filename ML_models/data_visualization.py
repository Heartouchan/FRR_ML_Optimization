# heartouchan
import numpy as np
import matplotlib.pyplot as plt
import data_process
import pandas as pd
import seaborn as sns
import matplotlib as mpl
plt.rcParams['font.size'] = 18

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# # Correlation coefficient of each two variable
corr = data_process.data.drop(['$r_{d}$','Fire resistance'], axis=1).corr() # examining correlations
plt.figure(figsize=(12, 10))
sns.set_context("notebook", font_scale=1.2)
heatmap=sns.heatmap(corr[(corr >= 0.0) | (corr <= -0.0)],
            cmap='coolwarm', vmax=1.0, vmin=-1.0, linewidths=0.2,
            annot=True, annot_kws={"size": 16}, square=True);
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=18)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=18)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.show()
corr = data_process.data.drop(['Fire resistance'], axis=1).corr()

quantitative_features_list1 = ['$W_{t}$', 't', '$W_{s}$', '$\delta$', 'd', 'n', '$e_{d}$', 'C',
                               '$\eta$']
data_plot_data=data_mod_num = data_process.data[quantitative_features_list1]

pair_plot=sns.pairplot(data_plot_data, corner=True, diag_kind='kde',diag_kws={'color': 'black', 'edgecolor': 'black', 'linewidth': 1.5}, plot_kws={'color': 'red'})
plt.xticks(fontstyle='italic')
plt.show()