import numpy as np

def polyreg(data_matrix, k):

    def residual(X, y, beta):
        return y - X@beta

    def sse(X, y, beta):
        res = residual(X, y, beta)
        return res.T @ res
    
    k = len(data_matrix)-1 if k >= len(data_matrix) else k

    x = np.swapaxes(data_matrix, 0, 1)[0]
    y = np.swapaxes(data_matrix, 0, 1)[1]
    
    # Generate X
    X = np.array(list(map(lambda xpt: [xpt**power for power in range(k+1)], x)))

    # Calculate beta
    beta = np.linalg.inv(X.T@X)@X.T@y

    # Get SSE_0
    sse_0 = sse(np.reshape(x, (x.shape[0], 1)), y, np.array([np.sum(y)/len(y)]))

    return beta, (1-sse(X, y, beta)/sse_0), residual(X, y, beta)

def predict(x, beta):
    return beta @ np.array(list(map(lambda xpt: [xpt**power for power in range(len(beta))], x))).T