class Adam_opt:

    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.lr = learning_rate
        self.b1 = beta1
        self.b2 = beta2
        self.epsilon = epsilon
        self.t = [0,0]
        self.m = [0,0]
        self.v = [0,0]

    def evaluate(self, grads, n):
        
        grad_updates = []
        for i, grad in enumerate(grads):
            self.t[i] += 1
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*(grad/n)
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(grad/n)**2
            m_hat = self.m[i]/(1-self.b1**self.t[i])
            v_hat = self.v[i]/(1-self.b2**self.t[i])
            grad_updates.append(self.lr*m_hat/(v_hat**(1/2) + self.epsilon))

        return grad_updates

class Mini_batch_opt:

    def __init__(self, learning_rate = 0.001):
        self.lr = learning_rate

    def evaluate(self, grads, n):
        grad_updates = []
        for grad in grads:
            grad_updates.append((self.lr/n)*grad)

        return grad_updates
        