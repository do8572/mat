import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from time import time
from math import inf

class gradient_descent:
    def __init__(self, tol=1e-15):
        self.tol = tol

    def set_params(self, f, df, ddf, opt_params):
        self.f = f
        self.df = df
        self.ddf = ddf

        if opt_params:
            #edf = np.linalg.eig(df)
            eddf, _ = np.linalg.eig(ddf([0,0,0]))
            #self.L = None #np.max(edf)
            self.alpha = np.min(eddf)
            self.beta = np.max(eddf)

    def minimize(self, f, df, ddf, x0, method="GD", dsize=None, num_steps=1000, y0=None, mu0=None, opt_params=True, tlim=inf):
        self.set_params(f, df, ddf, opt_params)
        x_star = None
        y0 = y0 if not opt_params else None
        m0 = mu0 if not opt_params else None
        if method == "GD":
            x_star = self.GD(x0, gamma=y0, num_steps=num_steps, tlim=tlim)
        elif method == "Polyak":
            x_star = self.PolyakGD(x0, gamma=y0, mu=m0, num_steps=num_steps, tlim=tlim)
        elif method == "Nesterov":
            x_star = self.NesterovGD(x0, gamma=y0, mu=m0, num_steps=num_steps, tlim=tlim)
        elif method == "AdaGrad":
            x_star = self.AdaGradGD(x0, gamma=y0, num_steps=num_steps, tlim=tlim)
        elif method == "Newton":
            x_star = self.NewtonMethod(x0, num_steps=num_steps)
        elif method == "BFGS":
            x_star = self.bfgs(x0, num_steps=num_steps)
        elif method == "SGD":
            x_star = self.SGD(x0, gamma=y0, dsize=dsize, num_steps=num_steps, tlim=tlim)
        elif method == "LBFGS":
            x_star = self.lbfgs(x0, num_steps=num_steps)
        return x_star

    def GD(self, xk, dsize=None, gamma=None, num_steps=1000, tlim=inf):
        if gamma == None:
            gamma = 2 / (self.alpha + self.beta)
        stime = time()
        for _ in range(num_steps):
            xk = xk - gamma * self.df(xk)
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
            if time() - stime > tlim:
                break
        return xk

    def PolyakGD(self, xk, dsize=None, gamma=None, mu=None, num_steps=1000, tlim=inf):
        if gamma == None:
            gamma = 4 / (np.sqrt(self.alpha) + np.sqrt(self.beta))**2
        if mu == None:
            mu = ((np.sqrt(self.beta) - np.sqrt(self.alpha)) / (np.sqrt(self.beta) + np.sqrt(self.alpha)))**2
        stime = time()
        x0 = xk
        for _ in range(num_steps):
            x0, xk = xk, xk - gamma * self.df(xk) + mu * (xk - x0)
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
            if time() - stime > tlim:
                break
        return xk

    def NesterovGD(self, xk, dsize=None, gamma=None, mu=None, num_steps=1000, tlim=inf):
        if gamma == None:
            gamma = 1 / self.beta
        if mu == None:
            kappa = self.beta / self.alpha
            mu = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        stime = time()
        x0 = xk
        for _ in range(num_steps):
            x0, xk = xk, xk - gamma * self.df(xk + mu * (xk - x0)) + mu * (xk - x0)
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
            if time() - stime > tlim:
                break
        return xk

    def AdaGradGD(self, xk, gamma, dsize=None, num_steps=1000, tlim=inf):
        Dk = np.ones(xk.shape).astype(float)
        if gamma == None:
            gamma = 2 / (self.beta + self.alpha)
        stime = time()
        for _ in range(num_steps):
            xk = xk - gamma * (self.df(xk) / np.sqrt(Dk))
            Dk += self.df(xk)**2
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
            if time() - stime > tlim:
                break
        return xk

    def NewtonMethod(self, xk, dsize=None, num_steps=10000, tlim=inf):
        stime = time()
        for _ in range(num_steps):
            xk = xk - np.linalg.inv(self.ddf(xk)).dot(self.df(xk))
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
            if time() - stime > tlim:
                break
        return xk

    def bfgs(self, xk, dsize=None, num_steps=1000, tlim=inf):
        Bk = np.eye(len(xk)) * 0.001 # Task 5: 0.001 / Task 6: 0.0001
        stime = time()
        x0 = xk
        for _ in range(num_steps):
            x0, xk = xk, xk - Bk.dot(self.df(xk))
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
            if time() - stime > tlim:
                break
            dk = (xk - x0)
            yk = self.df(xk) - self.df(x0)
            dk = dk.reshape(len(dk),1)
            yk = yk.reshape(len(yk),1)
            Bk = Bk - (dk.dot(yk.T.dot(Bk)) + Bk.dot(yk.dot(dk.T))) / (dk.T.dot(yk)) \
                 + (1 + (yk.T.dot(Bk.dot(yk)))/(dk.T.dot(yk))) * (dk.dot(dk.T)) / (dk.T.dot(yk))
        return xk

    def SGD(self, xk, dsize, gamma=None, batch_size=5, num_steps=1000, tlim=inf):
        if gamma == None:
            gamma = 2 / (self.alpha + self.beta)
        stime = time()
        for _ in range(num_steps):
            s = np.random.choice(np.arange(dsize), batch_size, replace = False)
            xk = xk - gamma * self.df(xk, s=s)
            if np.max(self.df(xk)) < self.tol:
                break
            if time() - stime > tlim:
                break
        return xk
    
    def lbfgs(self, xk, window=10, num_steps=1000, tlim=inf):
        # TODO: fix lbgfs
        Bk = np.eye(len(xk)) * 0.001
        stime = time()
        gi, yi, xi, pi = [], [], [], []
        for n in range(num_steps):
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
            if time() - stime > tlim:
                break
            g = self.df(xk)
            gi.append(g)
            a = [0] * min(window, len(xi))
            for i in range(min(window, len(xi))):
                i = window - i
                j = i + n - window
                a[i] = pi[j] * xi[j].T.dot(g) #save ai
                g = g - a[i] * yi[j]
            x = - Bk.dot(g)
            for i in range(min(window, len(xi))):
                j = i + n - window
                bi = pi[j] * yi[j].T.dot(xi[i])
                x = x - xi[j] * (a[i] + bi)
            # save xi, yi, pi
            xi.append(x)
            yi.append(self.df(xk) - gi[0])
            pi.append(1 / yi[0].T.dot(xk - x))
            if len(xi) > window:
                xi.pop(window)
                yi.pop(window)
                pi.pop(window)
            xk = x
        return xk

class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        
    def fit(self, N, opt, steps=100, time=inf):
        self.generate_data(N)
        full = np.arange(self.X.shape[0]) 
        f = lambda v, s=full: 0.5 * np.sum((self.X[s, :].dot(v) - self.y[s])**2) / len(s)
        df = lambda v, s=full: self.X[s, :].T.dot(self.X[s, :].dot(v) - self.y[s]) / len(s)
        ddf = lambda v, s=full: self.X[s, :].T.dot(self.X[s, :]) / len(s)

        x0 = np.array([5, 5])
        x_star = gradient_descent().minimize(f, df, ddf, x0, dsize=N, method=opt, tlim=time, num_steps=steps, opt_params=True)
        return x_star

    def generate_data(self, N):
        self.X = np.c_[np.arange(N), np.ones(N)]
        self.y = np.arange(N) + np.random.rand(N)

if __name__ == "__main__":
    task = []
    optimizers = ["GD", "Polyak", "Nesterov", "AdaGrad", "Newton", "BFGS"]
    f = lambda x: x[0]**2 + 2 * x[1]**2 - 2 * x[1] * x[2] + 4 * x[2]**2 + 3 * x[0] - 4 * x[1] + 5 * x[2]
    df = lambda x: np.array([2 * x[0] - 3, 4 * x[1] - 2 * x[2] - 4, -2 * x[1] + 8 * x[2] + 5])
    ddf = lambda x: np.array([[2,0,0], [0,4,-2],[0,-2,8]])
    #task.append(([np.array([1,2,3])], f, df, ddf))

    f2 = lambda x: (x[0] - x[2])**2 + (2*x[1] + x[2])**2 + (4*x[0] - 2*x[1] + x[2])**2 + x[0] + x[1]
    df2 = lambda x: np.array([34*x[0] - 16*x[1] + 6 * x[2] + 1,\
                              -16*x[0] + 16*x[1] + 1,\
                              6*(x[0]+x[2])
                            ])
    ddf2 = lambda x: np.array([[34, -16, 6], [-16,16,0], [6,0,6]])
    task.append(([np.array([0,0,0]), np.array([1,1,0])], f2, df2, ddf2))

    f3 = lambda x: (x[0]-1)**2 + (x[1]-1)**2 + 100 * (x[1] - x[0]**2)**2 + 100 * (x[2]-x[1]**2)**2
    df3 = lambda x: np.array([400 * x[0]**3 + (2 - 400 * x[1]) * x[0] - 2, \
                              400 * x[1]**3 + (202 - 400 * x[2]) * x[1] - 200 * x[0]**2 - 2, \
                              200 * (x[2] - x[1]**2)])
    ddf3 = lambda x: np.array([[-400 * (x[1] - x[0]**2) + 800 * x[0]**2 + 2, -400 * x[0], 0],\
                               [-400 * x[0], -400 * (x[2] - x[1]**2) + 800 * x[1]**2 + 202, -400 * x[1]],
                               [0, -400 * x[1], 200]])
    task.append(([np.array([1.2,1.2,1.2]), np.array([-1,1.2,1.2])], f3, df3, ddf3))

    f4 = lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2
    df4 = lambda x: 0.25 *  np.array([(x[1]-1) * (8*(x[1]**5 + x[1]**4 + 2 * x[1]**3 - x[1] - 3)*x[0] + 21 * x[1]**2 + 39 * x[1] + 51),
                                    x[0] * (24 * x[0] * x[1]**5 + 16 * x[0] *x[1]**3 \
                                     + (63 - 24 * x[0])*x[1]**2 + (36 - 8 * x[0]) * x[1] - 8 * x[0] + 12)
                                    ])
    ddf4 = lambda x: np.array([[2 * (x[1]**3 - 1)**2 + 2 * (x[1]**2 - 1)**2 + 2 * (x[1] - 1)**2, 
                                4 *x[0] * (x[1]**2 - 1) * x[1] + 4 * x[1] * (x[0] * x[1]**2 - x[0] + 2.25) + \
                                6 * x[0] * (x[1]**3 - 1) * x[1]**2 + 6 * x[1]**2 * (x[0] * x[1]**3 - x[0] + 2.625) \
                                     + 2 * x[0] * (x[1] - 1) + 2 * (x[0] * x[1] - x[0] + 1.5)],
                               [4 *x[0] * (x[1]**2 - 1) * x[1] + 4 * x[1] * (x[0] * x[1]**2 - x[0] + 2.25) + \
                                6 * x[0] * (x[1]**3 - 1) * x[1]**2 + 6 * x[1]**2 * (x[0] * x[1]**3 - x[0] + 2.625) \
                                     + 2 * x[0] * (x[1] - 1) + 2 * (x[0] * x[1] - x[0] + 1.5),
                                     18 * x[0]**2 * x[1]**4 + 8 * x[0]**2 * x[1]**2 + 2 * x[0]**2 + \
                                     12 * x[0] * x[1] * (x[0] * x[1]**3 - x[0] + 2.625) + 4 * x[0] * (x[0] * x[1]**2 - x[0] + 2.25)]])
    task.append(([np.array([1,1]), np.array([4.5,4.5])], f4, df4, ddf4))
    """
    step_arr = [2, 5, 10, 20, 30, 50, 60, 100]
    time_arr = [0.1, 1, 2, 5]
    params = [(0.01, 0.1), (0.0001, 0.01), (0.00001, 0.01)]
    optim = [(-1/6, -11/48, 1/6), (1,1,1), (3, 0.5)]
    tlims = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    res = pd.DataFrame(columns={"function", "pnt", "method", "n_iter", "score"})
    res2 = pd.DataFrame(columns={"function", "pnt", "method", "time", "score"})
    gd = gradient_descent()
    for t, p, o, i in zip(task, params, optim, np.arange(len(task))):
        pnts, f, df, ddf = t
        for pnt in pnts:
            for s in step_arr:
                for opt in optimizers:
                    print(opt)
                    x_star = gd.minimize(f,df,ddf, pnt, method=opt, num_steps=s, y0=p[0], mu0=p[1], \
                    opt_params=False)
                    res = res.append({"function": i, "method": opt, "pnt": pnt,\
                            "n_iter": s, "score": np.mean(np.abs(x_star - o))}, 
                            ignore_index=True)
            for tl in tlims:
                for opt in optimizers:
                    x_star2 = gd.minimize(f,df,ddf, pnt, method=opt, tlim=tl, y0=p[0], mu0=p[1], \
                    opt_params=False)
                    res2 = res2.append({"function": i, "method": opt, "pnt": pnt,\
                            "time": tl, "score": np.mean(np.abs(x_star2 - o))}, 
                            ignore_index=True)
                    #print(x_star)

    print(res)
    res.to_latex("task5a.tex", index=False)
    res["pnt"] = res["pnt"]

    for i in range(3):
        plt.clf()
        sns.lineplot(data=res[res["function"] == i], x="n_iter", y="score", hue="method", err_style=None)
        plt.ylim(0,5)
        plt.xlabel("n iter")
        plt.ylabel("MAE(actual minimum)")
        plt.savefig("plots/Iter" + str(i))
        #plt.show()

    res2.to_latex("task5b.tex", index=False)

    for i in range(3):
        plt.clf()
        sns.lineplot(data=res2[res2["function"] == i], x="time", y="score", hue="method", err_style=None)
        plt.ylim(0,2)
        plt.xlabel("time (s)")
        plt.ylabel("MAE(actual minimum)")
        plt.savefig("plots/Time" + str(i))
        #plt.show()
    """

    optimizers = ["GD", "SGD","Newton", "BFGS"]
    step_arr = [2, 5, 10, 20, 30, 50, 60, 100]
    res3 = pd.DataFrame(columns={"N", "method", "n_iter", "score"})
    res4 = pd.DataFrame(columns={"N", "method", "time", "score"})
    tlims = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    o = np.array([1, 0.5])
    for N in [50, 100, 1000, 10000, 100000, 1000000]:
        for s in step_arr:
            for opt in optimizers:
                lr = LinearRegression().fit(N, opt, steps=s)
                res3 = res3.append({"N": N, "method": opt, "n_iter": s,\
                                "score": np.mean(np.abs(lr - o))}, 
                                ignore_index=True)
        for tl in tlims:
                for opt in optimizers:
                    lr = LinearRegression().fit(N, opt, time=tl)
                    res4 = res4.append({"N": N, "method": opt, "time": tl,\
                                "score": np.mean(np.abs(lr - o))}, 
                                ignore_index=True)

    print(res3)
    res3.to_latex("task6a.tex")
    print(res4)
    res4.to_latex("task6b.tex")

    for i in step_arr:
        plt.clf()
        sns.lineplot(data=res3[res3["n_iter"] == i], x="N", y="score", hue="method", err_style=None)
        #plt.ylim(0,5)
        plt.xlabel("N")
        plt.ylabel("MAE(actual minimum)")
        plt.savefig("plots/LinRegIter" + str(i))
        #plt.show()

    for i in tlims:
        plt.clf()
        sns.lineplot(data=res4[res4["time"] == i], x="N", y="score", hue="method", err_style=None)
        #plt.ylim(0,5)
        plt.xlabel("N")
        plt.ylabel("MAE(actual minimum)")
        plt.savefig("plots/LinRegTime" + str(i).replace(".", ""))
        #plt.show()