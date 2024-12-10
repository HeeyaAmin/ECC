# import matplotlib.pyplot as plt
# import numpy as np
#
# models = ['Bidirectional LSTM', 'Linear Regression', 'GRU + Linear Regression']
# mse = [48.285558126378135, 48.1371010385435, 47.91638011794124]
# rmse = [6.948781053276764, 6.938090590252011, 6.922165854553128]
# r2 = [0.6114021211960412, 0.6125968906398356, 0]  # Set 0 for GRU + Linear Regression R² score
#
# x = np.arange(len(models))
#
# width = 0.25
#
# fig, ax = plt.subplots(figsize=(10, 6))
#
# bars1 = ax.bar(x - width, mse, width, label='MSE', color='skyblue')
# bars2 = ax.bar(x, rmse, width, label='RMSE', color='lightgreen')
# bars3 = ax.bar(x + width, r2, width, label='R²', color='salmon')
#
# ax.set_xlabel('Models')
# ax.set_ylabel('Values')
# ax.set_title('Comparison of Models Based on MSE, RMSE, and R² Scores')
# ax.set_xticks(x)
# ax.set_xticklabels(models)
# ax.legend()
#
# def add_labels(bars):
#     for bar in bars:
#         yval = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom' if yval > 0 else 'top', ha='center')
#
# add_labels(bars1)
# add_labels(bars2)
# add_labels(bars3)
#
# plt.tight_layout()
# plt.show()
#

import matplotlib.pyplot as plt
import numpy as np

models = ['Bidirectional LSTM', 'Linear Regression', 'GRU + Linear Regression']
mse = [48.285558126378135, 48.1371010385435, 47.91638011794124]
rmse = [6.948781053276764, 6.938090590252011, 6.922165854553128]
r2 = [0.6114021211960412, 0.6125968906398356, 0]  # Set 0 for GRU + Linear Regression R² score

x = np.arange(len(models))

width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width, mse, width, label='MSE', color='skyblue')
bars2 = ax.bar(x, rmse, width, label='RMSE', color='lightgreen')
bars3 = ax.bar(x + width, r2, width, label='R²', color='salmon')

ax.set_xlabel('Models')
ax.set_ylabel('Values')
ax.set_title('Comparison of Models Based on MSE, RMSE, and R² Scores')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

def add_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom' if yval > 0 else 'top', ha='center')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()

# Save the plot as an image
plot_path = "/Users/heeyaamin/PycharmProjects/ECC/static/images/model_comparison.png"
plt.savefig(plot_path)
plt.close()
