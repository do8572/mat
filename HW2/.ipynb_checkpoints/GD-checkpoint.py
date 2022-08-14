import numpy as np, pandas as pd
import time

class Observer:
    def __init__(self, eval_fun, name):
        self.name = name
        self.savefile = name + ".csv"
        self.observations = pd.DataFrame(columns=["fun", "run", "time", "iter", "score"])
        self.eval_fun = eval_fun
        self.run = 0
        
    def start_observing(self):
        self.stime = time.time()
        self.niter = 0
        self.run += 1
        
    def observe(self, citer, x):
        self.observations = self.observations.append({"fun": self.name,
                                                      "run": self.run,
                                                      "time": time.time() - self.stime,
                                                      "iter": citer,
                                                      "x": x}, ignore_index = True)
    
    def evaluate(self):
        self.observations["score"] = self.observations["x"].apply(self.eval_fun)
    
    def save_observations(self, savefile=None):
        if savefile is None:
            self.observations.to_csv(self.savefile, index=False)
        else:
            self.observations.to_csv(savefile, index=False)
            
    def reset(self):
        self.run = 0

class gradient_descent:
    def __init__(self, obs, tol=1e-15):
        self.tol = tol
        self.observer = obs

    def set_params(self, f, df, ddf):
        self.f = f
        self.df = df
        self.ddf = ddf

        eddf, _ = np.linalg.eig(ddf([0,0,0]))
        self.L = None 
        self.alpha = np.min(eddf)
        self.beta = np.max(eddf)

    def minimize(self, f, df, ddf, x0, method="GD", num_steps=1000):
        self.set_params(f, df, ddf)
        x_star = None
        if method == "GD":
            x_star = self.GD(x0, num_steps=num_steps)
        elif method == "Polyak":
            x_star = self.PolyakGD(x0, num_steps=num_steps)
        elif method == "Nesterov":
            x_star = self.NesterovGD(x0, num_steps=num_steps)
        elif method == "AdaGrad":
            x_star = self.AdaGradGD(x0, gamma=0.01, num_steps=num_steps)
        elif method == "Newton":
            x_star = self.NewtonMethod(x0, num_steps=num_steps)
        elif method == "BFGS":
            x_star = self.bfgs(x0, num_steps=num_steps)
        return x_star

    def GD(self, x0, gamma=None, num_steps=1000):
        if gamma == None:
            gamma = 2 / (self.alpha + self.beta)
        xk = x0
        self.observer.start_observing()
        for i in range(num_steps):
            self.observer.observe(i, xk)
            xk, x0 = xk - gamma * self.df(xk), xk
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
        print(f"GD score: {self.f(xk)}")
        return xk

    def PolyakGD(self, x0, gamma=None, mu=None, num_steps=1000):
        print(f"alpha: {self.alpha}")
        print(f"beta: {self.beta}")
        if gamma == None:
            gamma = 4 / (np.sqrt(self.alpha) + np.sqrt(self.beta))**2
        if mu == None:
            mu = ((np.sqrt(self.beta) - np.sqrt(self.alpha)) \
                / (np.sqrt(self.beta) + np.sqrt(self.alpha)))**2
        print(f"gamma: {gamma}")
        print(f"mu: {mu}")
        gamma = 0.01
        xk = x0
        
        self.observer.start_observing()
        for i in range(num_steps):
            self.observer.observe(i, xk)
            xk, x0 = xk - gamma * self.df(xk) + mu * (xk - x0), xk
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
        print(f"Polyak score: {self.f(xk)}")
        return xk

    def NesterovGD(self, x0, gamma=None, mu=None, num_steps=1000):
        if gamma == None:
            gamma = 1 / self.beta
        if mu == None:
            kappa = self.beta / self.alpha
            mu = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        xk = x0
        
        self.observer.start_observing()
        for i in range(num_steps):
            self.observer.observe(i, xk)
            x0 = xk
            xk = xk - gamma * self.df(xk + mu * (xk - x0)) + mu * (xk - x0)
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
        print(f"Nesterov score: {self.f(xk)}")
        return xk

    def AdaGradGD(self, x0, gamma, num_steps=1000):
        Dk = np.ones(x0.shape).astype(float)
        xk = x0
        
        self.observer.start_observing()
        for i in range(num_steps):
            self.observer.observe(i, xk)
            x0 = xk
            xk = xk - gamma * (self.df(xk) / np.sqrt(Dk))
            Dk += self.df(xk)**2
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
        print(f"AdaGrad score: {self.f(xk)}")
        return xk

    def NewtonMethod(self, x0, num_steps=1000):
        xk = x0
        
        self.observer.start_observing()
        for i in range(num_steps):
            self.observer.observe(i, xk)
            x0 = xk
            xk = xk - np.linalg.inv(self.ddf(xk)).dot(self.df(xk))
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
        print(f"Newton score: {self.f(xk)}")
        return xk

    def bfgs(self, x0, num_steps=1000):
        xk = x0
        Bk = np.eye(len(x0))
        
        self.observer.start_observing()
        for i in range(num_steps):
            self.observer.observe(i, xk)
            x0 = xk
            xk = xk - Bk.dot(self.df(xk))
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
            dk = (xk - x0)
            yk = self.df(xk) - self.df(x0)
            dk = dk.reshape(len(dk),1)
            yk = yk.reshape(len(yk),1)
            Bk = Bk - (dk.dot(yk.T.dot(Bk)) + Bk.dot(yk.dot(dk.T))) / (dk.T.dot(yk)) \
                 + (1 + (yk.T.dot(Bk.dot(yk)))/(dk.T.dot(yk))) * (dk.dot(dk.T)) / (dk.T.dot(yk))
        print(f"BFGS score: {self.f(xk)}")
        return xk

if __name__ == "__main__":
    task, optimizers = [], ["GD", "Polyak", "Nesterov", "AdaGrad", "Newton", "BFGS"]

    f1 = lambda x: (x[0] - x[2])**2 + (2*x[1] + x[2])**2 + (4*x[0] - 2*x[1] + x[2])**2 + x[0] + x[1]
    df1 = lambda x: np.array([34*x[0] - 16*x[1] + 6 * x[2] + 1,\
                              -16*x[0] + 16*x[1] + 1,\
                              6*(x[0]+x[2])])
    ddf1 = lambda x: np.array([[34, -16, 6], [-16,16,0], [6,0,6]])

    f2 = lambda x: (x[0]-1)**2 + (x[1]-1)**2 + 100 * (x[1] - x[0]**2)**2 + 100 * (x[2]-x[1]**2)**2
    df2 = lambda x: np.array([2*(x[0]-1)-400*(x[1] - x[0]**2) * x[0],\
                              2*(x[1]-1) + 200*(x[1] - x[0]**2) - 400*(x[2] - x[1]**2) * x[1],\
                              200*(x[2] - x[1]**2)])
    ddf2 = lambda x: np.array([[2-400*(x[1]- 3 * x[0]**2), -400* x[0], 0],\
                               [-400* x[0], 202 - 400*(x[2] - 3 * x[1]**2), -400* x[1]],
                               [0, -400 * x[1], 200]])

    f3 = lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2
    df3 = lambda x: 2 * np.array([(1.5-x[0] + x[0]*x[1])*(-1 + x[1]) + (2.25 - x[0] + x[0]* x[1]**2) * (-1 + x[1]**2) +\
                              (2.625 - x[0] + x[0]*x[1]**3) * (-1 + x[1]**3),\
                              (1.5-x[0] + x[0]*x[1])*x[0] + 2*(2.25 - x[0] + x[0]* x[1]**2) * x[0] * x[1] +\
                              (2.625 - x[0] + x[0]*x[1]**3) * 3 * x[0] * x[1]**2])
    ddf3 = lambda x: 2 * np.array([[(-1 + x[1])**2 + (-1 + x[1]**2)**2 + (-1 + x[1]**3)**2,\
                                (1.5-x[0] + 2 * x[0]*x[1]) + (4.5 * x[1] - 2 * x[0] * x[1]) \
                                 + (37.5 * x[1]**2) - 3 * x[0] * x[1]**2 + 6 * x[0] * x[1]**5],\
                               [(1.5-x[0] + 2 * x[0]*x[1]) + (4.5 * x[1] - 2 * x[0] * x[1]) \
                                 + (37.5 * x[1]**2) - 3 * x[0] * x[1]**2 + 6 * x[0] * x[1]**5,\
                                   x[0]**2 + (4.5 * x[0] - 2 * x[0]**2 + 6 * x[0]**2 *x[1]**2) \
                                    + 2 * (75 * x[0] * x[1] - 3 * x[0]**2 * 2 * x[1] + 15 * x[0]**2 * x[1]**4)]])
    
    toy_functions = [{"fun": f1,
                      "dfun": df1,
                      "ddfun": ddf1,
                      "obs": Observer(lambda x: np.mean(np.abs(x - np.array((-1/6, -11/48, 1/6)))), "gf1"),
                      "start_pnts": [np.array([0,0,0]), np.array([1,1,0])]},
                     {"fun": f2,
                      "dfun": df2,
                      "ddfun": ddf2,
                      "obs": Observer(lambda x: np.mean(np.abs(x - np.array((1,1,1)))), "gf2"),
                      "start_pnts": [np.array([1.2, 1.2, 1.2]), np.array([-1, 1.2, 1.2])]},
                     {"fun": f3,
                      "dfun": df3,
                      "ddfun": ddf3,
                      "obs": Observer(lambda x: np.mean(np.abs(x - np.array((3, 0.5)))), "gf3"),
                      "start_pnts": [np.array([1, 1]), np.array([4.5, 4.5])]}]

    for tfun in toy_functions:
        fi, dfi, ddfi, obs, spnts = tfun["fun"], tfun["dfun"], tfun["ddfun"], tfun["obs"], tfun["start_pnts"]
        
        gd = gradient_descent(obs)
        
        for opt in optimizers:
            for spnt in spnts:
                xmin =  gd.minimize(fi, dfi, ddfi, spnt, method=opt, num_steps=100)

                obs.evaluate()
                obs.save_observations("GD_results/" + obs.name + "-" + opt + ".csv")
                
                print(f"Method: {opt}, f={obs.name}, spnt={spnt}")
                print(f"Real x_min: {xmin}, f = {fi(xmin)}")
                # print(f"Our x_min: {xint}, f = {fi(xint)} (converged: {success}, iterations: {uiter})")
                # print(f"{'Found minimum.' if fi(xmin) >= fi(xint) else 'Failed to find minimum.'}")

                print("---------------------------------------------")
                print("---------------------------------------------")
            obs.reset()

    rf = pd.concat([tfun["obs"].observations for tfun in toy_functions])
    rf = rf.drop("x", axis=1)