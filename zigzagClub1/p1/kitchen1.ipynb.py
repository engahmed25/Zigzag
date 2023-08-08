import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()




eta = 0.01;
n_iter = 50;
random_state = 1;


rgen = np.random.RandomState(random_state)

Width_ =  rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

def predict(Xp):
 """Return class label after unit step"""
 return np.where(net_input(Xp) >= 0.0, 1, -1)

def net_input(Xn):
 """Calculate net input"""
 
 return np.dot(Xn, Width_[1:])+Width_[0]

i = 0;
errors_ = []
for _ in range(15):
    errors = 0;
    for xi, target in zip(X, y):
        update = eta*(target - predict(xi))
        print("%d-  %4.3f"%(i, update), end = "|||  ")
        Width_[1:] +=update * xi
        Width_[0] = update
        print(Width_[1:])
        errors += int(update != 0.0)    
        i+=1;
    errors_.append(errors)
        
        

    
plt.plot(range(1, len(errors_) + 1), errors_, marker='o')
    
    
