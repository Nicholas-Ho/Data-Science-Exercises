import numpy as np

class PolynomialRegressor:
    def residual(self, X, y, beta):
        return y - X@beta

    def sse(self, X, y, beta):
        res = self.residual(X, y, beta)
        return res.T @ res

    def polyreg(self, data_matrix, k):
        k = len(data_matrix)-1 if k >= len(data_matrix) else k

        x = np.swapaxes(data_matrix, 0, 1)[0]
        y = np.swapaxes(data_matrix, 0, 1)[1]
        
        # Generate X
        X = np.array(list(map(lambda xpt: [xpt**power for power in range(k+1)], x)))

        # Calculate beta
        beta = np.linalg.inv(X.T@X)@X.T@y

        # Get SSE_0
        sse_0 = self.sse(np.reshape(x, (x.shape[0], 1)), y, np.array([np.sum(y)/len(y)]))

        return beta, (1-self.sse(X, y, beta)/sse_0), self.residual(X, y, beta)

    def predict(self, x, beta):
        return beta @ np.array(list(map(lambda xpt: [xpt**power for power in range(len(beta))], x))).T

class DFTRegressor:
    def residual(self, X, y, beta):
        return y - X@beta

    def sse(self, X, y, beta):
        res = self.residual(X, y, beta)
        return res.T @ res

    def fit(self, data_matrix, freqs):
        x = np.swapaxes(data_matrix, 0, 1)[0]
        y = np.swapaxes(data_matrix, 0, 1)[1]
        
        # Generate X
        sincos = (lambda x: np.sin(x), lambda x: np.cos(x))
        X = np.array(list(map(lambda xpt: [f(w*2*np.pi*xpt) for w in freqs for f in sincos], x)))

        # Calculate beta
        beta = np.linalg.inv(X.T@X)@X.T@y

        # Get SSE_0
        sse_0 = self.sse(np.reshape(x, (x.shape[0], 1)), y, np.array([np.sum(y)/len(y)]))

        return beta, (1-self.sse(X, y, beta)/sse_0), self.residual(X, y, beta)

    def predict(self, x, beta, freqs):
        sincos = (lambda x: np.sin(x), lambda x: np.cos(x))
        return beta @ np.array(list(map(lambda xpt: [f(w*2*np.pi*xpt) for w in freqs for f in sincos], x))).T

