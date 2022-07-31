import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import e

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

L = 14.67197
b = 9.521999
a = 1.065381

def df(x):
    return np.array([2*x[0] + np.exp(x[0])-x[1], 2*x[1]-x[0]])

def f(x):
    return x[0]**2 + np.exp(x[0]) + x[1]**2 - x[0]*x[1]

def pgd(df, x1, projection, gamma, bound, n_steps=10, res=pd.DataFrame(columns=["step", "gamma", "projection", "xi", "yi", "bound"])):
    xi = x1
    X = []
    X.append(x1)
    for i in range(n_steps):
        xi = projection(xi - gamma(i, x1) * df(xi))
        X.append(xi)
        res = res.append({"step": i, "gamma": gamma.__name__, \
            "projection":projection.__name__, "xi":xi[0], "yi":xi[1], "bound": bound(X, i+1)}, \
                ignore_index=True)
    return xi, res

def gd(df, x1, projection, gamma, bound, n_steps=10, res=pd.DataFrame(columns=["step", "gamma", "projection", "xi", "yi", "bound"])):
    xi = x1
    X = []
    X.append(x1)
    for i in range(n_steps):
        xi = xi - gamma(i, x1) * df(xi)
        X.append(xi)
        res = res.append({"step": i, "gamma": gamma.__name__, \
            "projection":None, "xi":xi[0], "yi":xi[1], "bound": None}, \
                ignore_index=True)
    print(f"Number of steps: {i}")
    return xi, res

def circle(x):
    if x[0]**2 + x[1]**2 > 1.5:
        return x / np.sqrt((x[0]**2 + x[1]**2)/1.5)
    return x

def square(x):
    if x[0] < 2 and x[0] > -2 and x[1] < -2:
        return np.array([x[0],-2])
    if x[0] < 2 and x[0] > -2 and x[1] > 2:
        return np.array([x[0],2])

    if x[0] < -2 and x[1] > -2 and x[1] < 2:
        return np.array([-2, x[1]])
    if x[0] > 2 and x[1] > -2 and x[1] < 2:
        return np.array([2, x[1]])

    if x[0] < -2 and x[1] < -2:
        return np.array([-2,-2])
    if x[0] > 2 and x[1] > 2:
        return np.array([2,2])
    if x[0] < -2 and x[1] > 2:
        return np.array([-2,2])
    if x[1] < -2 and x[0] > 2:
        return np.array([2,-2])
    
    return x

def triangle(x, A=np.array([-1,-1]), B=np.array([1.5,-1]), C=np.array([-1, 1.5])):
    if x[0] < -1 and x[1] > -1 and x[1] < 1.5:
        return np.array([-1, x[1]])
    if x[0] < -1 and x[1] < -1:
        return A
    if x[0] > -1 and x[0] < 1.5 and x[1] < -1:
        return np.array([x[0], -1])
    if x[0] > 1.5 and x[1] - x[0] < -2.5:
        return B
    if x[1] - x[0] > -2.5 and x[1] - x[0] < 2.5 and x[0] + x[1] >= 0.5:
        print((0.08 * (x - B).dot(np.array([2.5,2.5]))))
        return B + 0.08 * ((x - B).dot(np.array([-2.5,2.5])))*np.array([-2.5,2.5]) 
    if x[1] > 1.5 and x[1] - x[0] > 2.5:
        return C
    return x 

def gamma1(k, x1, x_star=np.array([-0.432563, -0.216281])):
    return np.sqrt((x1 - x_star).dot(x1 - x_star))/L*np.sqrt(k)

def gamma2(k, x):
    return 1/ b

def gamma3(k, x):
    return 2 / (a * (k+1))

def bound1(X, k):
    x_star = np.array([-0.432563, -0.216281])
    return f(np.mean(X[:k], axis=0)) - f(x_star) <= L * np.sqrt((X[0]-x_star).T.dot(X[0]-x_star))/ np.sqrt(k)

def bound2(X, k):
    x_star = np.array([-0.432563, -0.216281])
    return f(X[k]) - f(x_star) <= (3 * b * (X[0] - x_star).T.dot(X[0] - x_star) + f(X[0]) - f(x_star)) / k

def bound3(X, k):
    x_star = np.array([-0.432563, -0.216281])
    return f(np.sum([2 * X[i] * i for i in range(k+1)], axis=0) / k / (k+1)) - f(x_star) <= 2 * L**2 / a / (k+1)

if __name__ == "__main__":
    """
    A=(-1,-1)
    B=(1.5,-1)
    C=(-1, 1.5)
    plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]])
    plt.plot(2,2, "ro")
    proj_x = triangle(np.array([2,2]))
    print(proj_x)
    plt.plot(proj_x[0], proj_x[1], "ro")
    plt.plot(1.5,1.5, "ro")
    proj_x = triangle(np.array([1.5,1.5]))
    print(proj_x)
    plt.plot(proj_x[0], proj_x[1], "ro")
    plt.show()
    """
    x1 = np.array([-1, 1])

    A=(-1,-1)
    B=(1.5,-1)
    C=(-1, 1.5)

    D = (-2,-2)
    E = (2,-2)
    F = (2,2)
    G = (-2,2)
    bounds = [([], []), \
            ([D[0], E[0], F[0], G[0], D[0]], [D[1], E[1], F[1], G[1], D[1]]), \
            ([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]])]

    tbounds = [bound1, bound2, bound3] 
    """
    plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]])

    plt.plot(0,0, "ro")
    proj_x = triangle(np.array([0,0]))
    print(proj_x)
    plt.plot(proj_x[0], proj_x[1], "ro")
    plt.show()
    """
    gammas = [gamma1, gamma2, gamma3]
    projections = [circle, square, triangle]
    res=pd.DataFrame(columns=["step", "gamma", "projection", "xi"])

    splt = 0

    fig, ax = plt.subplots(1, 3)

    for projection in projections:
        if splt == 0:
            ax[splt].add_patch(plt.Circle((0, 0), 1.5, color='b', fill=False))
        else:
            ax[splt].plot(bounds[splt][0],bounds[splt][1])

        for gamma, tbound in zip(gammas, tbounds):
            xi, res = pgd(df, x1, projection, gamma, tbound, 10, res)

        #print(res[res["projection"] == projection.__name__])
        
        colors = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), \
                            name)
                            for name, color in mcolors.TABLEAU_COLORS.items())
        ind = 0
        for gamma in gammas:
            subres = res[(res["gamma"] == gamma.__name__) & (res["projection"] == projection.__name__)]
            ax[splt].plot(subres["xi"], subres["yi"], \
                        "o", color=colors[ind][0], alpha=0.5, label=gamma.__name__)
                
            for row in subres.iterrows():
                ax[splt].annotate(row[1]["step"], (row[1]["xi"], row[1]["yi"]))
            ind += 1

        ax[splt].plot(-0.432563, -0.216281, "rx", label="minimum")
        splt += 1
    plt.legend()
    plt.show()

    print(res)
    res.to_latex("task5.tex", index=False)

    # Task 6
    a = np.array([[3, 10, 30],
                  [0.1, 10, 35],
                  [3, 10, 30],
                  [0.1, 10, 35]])
    c = np.array([1, 1.2, 3, 3.2])
    p = np.array([[0.3689, 0.117, 0.2673],
                  [0.4699, 0.4387, 0.747],
                  [0.1091, 0.8732, 0.5547],
                  [0.03815, 0.5743, 0.8828]])
    
    x1 = np.array([1,1,1]).reshape(3,1)
    f = lambda z: -sum([c[i] * e**-sum(a[i,j] * (z[j] - p[i,j])**2 for j in range(3)) for i in range(4)])[0]
    dfi = lambda z, ind: -sum([c[i] * e**-sum(a[i,j] * (z[j] - p[i,j])**2 for j in range(3)) \
        * (-2) * a[i, ind] * (z[ind] - p[i, ind]) for i in range(4)])[0]
    df = lambda z: np.array([dfi(z, 0), dfi(z, 1), dfi(z, 2)]).reshape(3,1)

    print(f(x1))
    print(df(x1))
    xi, res = gd(df, x1, projection, lambda k,X: 0.0006, lambda x: None, 10000, res)
    print(f(xi))