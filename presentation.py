import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, RidgeCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
import math
import numpy as np
import pprint
import random

data = pd.read_csv('auto-mpg.csv', index_col='car name')
X = data.drop(columns=['mpg']).values
y = data['mpg'].values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)

vals = dict(zip(data.columns,data.values.swapaxes(0,1)))

meds = []

for val in vals:
    if val in 'car namecylindersmodel yearorigin':
        continue
    l=vals[val][:]
    l.sort()
    o=[]
    for x in range(4):
        o.append(l[len(l)*x//4])
    meds.append(o+[l[-1]])

def randcarcircle(car=None, name=None):
    labels = 'MPG Displacement Horsepower Weight Acceleration'.split()
    if car is not None:
        values=car.drop(labels=['cylinders','model year','origin']).tolist()
    else:
        values = random.choice(data.drop(columns=['cylinders','model year','origin']).values).tolist()
        values,name = values[:-1], values[-1]
    vs=[]
    for c,x in enumerate(values):
        alx=vals[labels[c].lower()]
        n=list(sorted(alx.tolist())).index(x)
        vs.append(n*8/len(alx)+2)
        
    values = vs
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.yaxis.set_visible(False)

    ax.set_thetagrids(np.degrees(angles), labels)
    ax.title.set_text(name.title())
    ax.yaxis.set_visible(False)

    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        
    for x in range(0,5):
        ax = ptwinx(ax)
        ax.set_rlabel_position(x*72)
        ax.set_yticklabels(meds[x])

    ax.set_ylim(0, 10)
    ax.set_thetagrids(np.degrees(angles), ['']*len(angles))
    ax.set_rgrids(range(2,11,2))
    ax.plot(angles, values, linewidth=1)
    ax.fill(angles, values, alpha=0.25)
    ax.xaxis.set_visible(True)
    plt.show()

def ptwinx(ax):
    ax2 = ax.figure.add_axes(ax.get_position(), projection='polar', 
                             label='', frameon=True,
                             theta_direction=ax.get_theta_direction(),
                             theta_offset=ax.get_theta_offset())
    ax2.xaxis.set_visible(False)
    ax2._r_label_position.invalidate()
    for label in ax.get_yticklabels():
        ax.figure.texts.append(label)
    return ax2


def accuracy(ypred, ytest):
    sum_offset = 0.0

    for pred, test in zip(ypred, ytest):
        sum_offset += abs(pred - test)

    return sum_offset / len(ypred)

'''
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(Xtrain, ytrain)
rfr_pred = rfr.predict(Xtest)
rfr_acc = accuracy(rfr_pred, ytest)

lr = LinearRegression()
lr.fit(Xtrain, ytrain)
lr_pred = lr.predict(Xtest)
lr_acc = accuracy(lr_pred, ytest)

l = Lasso()
l.fit(Xtrain, ytrain)
l_pred = l.predict(Xtest)
l_acc = accuracy(l_pred, ytest)

rcv = RidgeCV(alphas=np.arange(70,100,0.1), fit_intercept=True)
rcv.fit(Xtrain, ytrain)
rcv_pred = rcv.predict(Xtest)
rcv_acc = accuracy(rcv_pred, ytest)

model_accuracy = {
    'RandomForestRegressor': rfr_acc,
    'LinearRegression': lr_acc,
    'Lasso': l_acc,
    'RidgeCV': rcv_acc
}

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(model_accuracy)


'''
data = data.sort_values(by=['mpg'], ascending=False)
best_cars = data['mpg'][0:5]
data = data.sort_values(by=['mpg'], ascending=True)
worst_cars = data['mpg'][0:5]

labels = [
    'cylinders',
    'displacement',
    'horsepower',
    'weight',
    'acceleration',
    'model year',
    'origin'
]

def graph(values):
    global labels
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

for items in best_cars.iteritems(): 
    car, mpg = items
    car_data = data.loc[car]
    randcarcircle(car_data, car)

for items in worst_cars.iteritems(): 
    car, mpg = items
    car_data = data.loc[car]
    randcarcircle(car_data, car)