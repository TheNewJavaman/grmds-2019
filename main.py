import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np

data = pd.read_csv('auto-mpg.csv', index_col='car name')
#sns.distplot(data.head(393)['mpg'])
#sns.countplot(data.head(393)['model year'])
#sns.countplot(data.head(393)['origin'])
#plt.show()

X = data.drop(columns=['mpg']).values
y = data['mpg'].values

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
model = LinearRegression()
model.fit(Xtrain, ytrain)

y_pred = model.predict(Xtest)

#plt.plot(ytest, color='blue', label='True value')
#plt.plot(y_pred, color='red', label='Predict value')
#plt.legend()
#plt.show()

data = data.sort_values(by=['mpg'], ascending=False)
#sns.barplot(data['mpg'][0:5], data['car name'][0:5])
#labels = '1', '3', '2'
#sizes = [246, 79, 68]
#plt.pie(sizes, labels=labels)
#corrmat = data.corr()
#sns.heatmap(corrmat)
#sns.pairplot(data, vars=data.columns[:-1], hue='origin')
#plt.show()

labels = [
    'mpg',
    'displacement',
    'horsepower',
    'weight',
    'acceleration',
]



num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

values += values[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

ax.plot(angles, values, linewidth=1)
ax.fill(angles, values, alpha=0.25)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_thetagrids(np.degrees(angles), labels)

for label, angle in zip(ax.get_xticklabels(), angles):
  if angle in (0, np.pi):
    label.set_horizontalalignment('center')
  elif 0 < angle < np.pi:
    label.set_horizontalalignment('left')
  else:
    label.set_horizontalalignment('right')

ax.set_ylim(0, 100)
ax.set_rlabel_position(180 / num_vars)

plt.show()