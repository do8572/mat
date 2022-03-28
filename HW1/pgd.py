from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

L = 14.67197
T = 10
b = 9.521999
a = 1.065381

def df(x):
    return np.array([2*x[0] + np.exp(x[0])-x[1], 2*x[1]-x[0]])

def f(x):
    print(x)
    print()
    return x[0]**2 + np.exp(x[0]) + x[1]**2 - x[0]*x[1]

def pgd(df, x1, projection, gamma, bound, n_steps=10, res=pd.DataFrame(columns=["step", "gamma", "projection", "xi", "yi", "bound"])):
    xi = x1
    X = []
    X.append(x1)
    for i in range(n_steps):
        xi = projection(xi - gamma(i, x1) * df(x1) * xi)
        X.append(xi)
        res = res.append({"step": i, "gamma": gamma.__name__, \
            "projection":projection.__name__, "xi":xi[0], "yi":xi[1], "bound": bound(X, i)}, \
                ignore_index=True)
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

def triangle(x, A=(-1,-1), B=(1.5,-1), C=(-1, 1.5)):
    print(x)
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

def gamma1(k, x1):
    return x1/L*np.sqrt(T)

def gamma2(k, x):
    return 1/ b

def gamma3(k, x):
    return 2 / (a * (k+1))

def bound1(X, k):
    return f(np.mean(X[:T], axis=0)) - 1 <= L * X[0]/ np.sqrt(T)

def bound2(X, k):
    return None

def bound3(X, k):
    return None

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
    x1 = np.array([1, 1])

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
            xi, res = pgd(df, x1, projection, gamma, tbound, 100, res)
            print(xi)

        print(res[res["projection"] == projection.__name__])
        
        ax[splt].plot(0, 0, "rx")
        colors = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), \
                            name)
                            for name, color in mcolors.TABLEAU_COLORS.items())
        print(colors)
        ind = 0
        for gamma in gammas:
            subres = res[(res["gamma"] == gamma.__name__) & (res["projection"] == projection.__name__)]
            ax[splt].plot(subres["xi"], subres["yi"], \
                        "o", color=colors[ind][0], alpha=0.5)
                
            for row in subres.iterrows():
                ax[splt].annotate(row[1]["step"], (row[1]["xi"], row[1]["yi"]))
            ind += 1
        splt += 1
    plt.show()