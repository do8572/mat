import numpy as np

class gradient_descent:
    def __init__(self, tol=1e-15):
        self.tol = tol

    def set_params(self, f, df, ddf):
        self.f = f
        self.df = df
        self.ddf = ddf

        #edf = np.linalg.eig(df)
        eddf, _ = np.linalg.eig(ddf([0,0,0]))
        self.L = None #np.max(edf)
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
        for _ in range(num_steps):
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
        for _ in range(num_steps):
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
        for _ in range(num_steps):
            x0 = xk
            xk = xk - gamma * self.df(xk + mu * (xk - x0)) + mu * (xk - x0)
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
        print(f"Nesterov score: {self.f(xk)}")
        return xk

    def AdaGradGD(self, x0, gamma, num_steps=1000):
        Dk = np.ones(x0.shape).astype(float)
        xk = x0
        for _ in range(num_steps):
            x0 = xk
            xk = xk - gamma * (self.df(xk) / np.sqrt(Dk))
            Dk += self.df(xk)**2
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
        print(f"AdaGrad score: {self.f(xk)}")
        return xk

    def NewtonMethod(self, x0, num_steps=1000):
        xk = x0
        for _ in range(num_steps):
            x0 = xk
            xk = xk - np.linalg.inv(self.ddf(xk)).dot(self.df(xk))
            if np.max(np.abs(self.df(xk))) < self.tol:
                break
        print(f"Newton score: {self.f(xk)}")
        return xk

    def bfgs(self, x0, num_steps=1000):
        xk = x0
        Bk = np.eye(len(x0))
        for _ in range(num_steps):
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
        print(f"LBFGS score: {self.f(xk)}")
        return xk

    # TODO: BFGS with line search

    def SGD(self, x0, num_steps=1000):
        if gamma == None:
            gamma = 2 / (self.alpha + self.beta)
        xk = x0
        for _ in range(num_steps):
            x0 = xk
            xk = xk - gamma * self.df(xk)
            if np.max(self.df(xk)) < self.tol:
                break
        return xk
    
    def lbfgs(self, x0, num_steps=1000):
        pass

class LinearRegression:
    def __init__(self):
        self.X = None
        self.y = None
        
    def fit(self, N, opt):
        self.generate_data(N)
        f = lambda v: 0.5 * (self.X.dot(v) - self.y).T.dot(self.X.dot(v) - self.y)
        df = lambda v: -self.X.T.dot(self.y - self.X.dot(v))
        ddf = lambda v: -self.X.T.dot(self.X)

        x0 = np.ones((2,1))
        return gradient_descent().minimize(f, df, ddf, x0, method=opt, num_steps=100000)

    def generate_data(self, N):
        self.X = np.c_[np.arange(N), np.ones(N)]
        self.y = (np.arange(N) + np.random.rand(N)).reshape(N,1)

if __name__ == "__main__":
    task = []
    optimizers = ["GD", "Polyak", "Nesterov", "AdaGrad", "Newton", "BFGS"]
    f = lambda x: x[0]**2 + 2 * x[1]**2 - 2 * x[1] * x[2] + 4 * x[2]**2 + 3 * x[0] - 4 * x[1] + 5 * x[2]
    df = lambda x: np.array([2 * x[0] - 3, 4 * x[1] - 2 * x[2] - 4, -2 * x[1] + 8 * x[2] + 5])
    ddf = lambda x: np.array([[2,0,0], [0,4,-2],[0,-2,8]])
    #task.append(([np.array([1,2,3])], f, df, ddf))

    f2 = lambda x: (x[0] - x[2])**2 + (2*x[1] + x[2])**2 + (4*x[0] - 2*x[1] + x[2])**2 + x[0] + x[1]
    df2 = lambda x: np.array([34*x[0] -16*x[1] + 6 * x[2] + 1, \
                              -16*x[0] + 16*x[1] + 1, \
                             6*(x[0]+x[2]) 
                            ])
    ddf2 = lambda x: np.array([[34, -16, 6], [-16,16,0], [6,0,6]])
    task.append(([np.array([0,0,0]), np.array([1,1,0])], f2, df2, ddf2))

    f3 = lambda x: (x[0]-1)**2 + (x[1]-1)**2 + 100 * (x[1] - x[0]**2)**2 + 100 * (x[2]-x[1]**2)**2
    df3 = lambda x: np.array([2*(x[0]-1)-400*(x[1] - x[0]**2) * x[0],\
                              2*(x[1]-1) + 200*(x[1] - x[0]**2) - 400*(x[2] - x[1]**2) * x[1],\
                              200*(x[2] - x[1]**2)])
    ddf3 = lambda x: np.array([[2-400*(x[1]- 3 * x[0]**2), -400* x[0], 0],\
                               [-400* x[0], 202 - 400*(x[2] - 3 * x[1]**2), -400* x[1]],
                               [0, -400 * x[1], 200]])
    #task.append(([np.array([1.2,1.2,1.2]), np.array([-1,1.2,1.2])], f3, df3, ddf3))

    f4 = lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2
    df4 = lambda x: 2 * np.array([(1.5-x[0] + x[0]*x[1])*(-1 + x[1]) + (2.25 - x[0] + x[0]* x[1]**2) * (-1 + x[1]**2) +\
                              (2.625 - x[0] + x[0]*x[1]**3) * (-1 + x[1]**3),\
                              (1.5-x[0] + x[0]*x[1])*x[0] + 2*(2.25 - x[0] + x[0]* x[1]**2) * x[0] * x[1] +\
                              (2.625 - x[0] + x[0]*x[1]**3) * 3 * x[0] * x[1]**2])
    ddf4 = lambda x: 2 * np.array([[(-1 + x[1])**2 + (-1 + x[1]**2)**2 + (-1 + x[1]**3)**2,\
                                (1.5-x[0] + 2 * x[0]*x[1]) + (4.5 * x[1] - 2 * x[0] * x[1]) \
                                 + (37.5 * x[1]**2) - 3 * x[0] * x[1]**2 + 6 * x[0] * x[1]**5],\
                               [(1.5-x[0] + 2 * x[0]*x[1]) + (4.5 * x[1] - 2 * x[0] * x[1]) \
                                 + (37.5 * x[1]**2) - 3 * x[0] * x[1]**2 + 6 * x[0] * x[1]**5,\
                                   x[0]**2 + (4.5 * x[0] - 2 * x[0]**2 + 6 * x[0]**2 *x[1]**2) \
                                    + 2 * (75 * x[0] * x[1] - 3 * x[0]**2 * 2 * x[1] + 15 * x[0]**2 * x[1]**4)]])
    #task.append(([np.array([1,1]), np.array([4.5,4.5])], f4, df4, ddf4))
    
    gd = gradient_descent()
    for opt in optimizers:
        for t in task:
            print(opt)
            pnts, f, df, ddf = t
            for pnt in pnts:
                x_star = gd.minimize(f,df,ddf, pnt, method=opt, num_steps=10000)
                print(x_star)
    """
    lr = LinearRegression().fit(100, "GD")
    print(lr)
    """