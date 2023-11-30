import numpy as np
from sklearn.metrics import mean_squared_error

def train(N) :
    sum_uniform = 0
    sum_importance = 0
    for i in range(N) :
        rand = np.random.rand(1)
        y_value = np.sin(rand * np.pi / 2)
        sum_uniform += y_value
        importance = np.sqrt(rand) * (np.pi / 2)
        sum_importance += importance
    uniform_constant = np.pi / 2 /N
    importance_constant = 1 / N
    uniform = uniform_constant * sum_uniform
    importance = importance_constant * importance
    return uniform, importance

def main() :
    N = 16
    print(f'step 1.')
    for  i in range(10) :
        uniform, importance = train(N)


if __name__ == "__main__":
    main()