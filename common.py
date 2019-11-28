import numpy as np

def activation_sigmoid(x):
    return 1/(1+np.exp(-x))

def activation_step(x):
    y = x > 0
    return y.astype(np.int)

def activation_relu(x): # 왜 maximum을 쓰는 걸까요? max가 아니라. 
    return np.maximum(0, x)

def softmax_easy(a): # 컴퓨터의 저장공간 한계가 있기 때문에, 논리는 맞지만 exp를 개선해야 합니다.
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def error_mse(y: np.array, t: np.array) -> np.array:
    """평균 제곱 오차(mean squared error)를 구합니다.
    args:
        y:numpy.array:예측 값들을 numpy로 나타낸 것
        t:numpy.array:정답 값들을 numpy로 나타낸 것
    """

    return 0.5*np.sum((y-t)**2)

def error_cross_entropy(y: np.array, t: np.array) -> np.array:
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    # return -np.sum(t*np.log(y+delta)) # 0이 되지 않게.
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size # (batch용 크로스 엔트로피)

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    # 0에 최대한 가깝게 1e-50 정도로 하고 싶을 수도 있지만,
    # 반올림 오차 문제를 일으켜 0으로 취급될 수도 있습니다.
    # 10^-4 정도의 값을 사용하면, 좋은 결과를 얻는다고 알려져 있습니다.
    # f(x+h) - f(x) / h 보다 아래가 더 정확함
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    for idx in range(x.shape[0]):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """경사하강법. network를 update하는 방법
    
    Arguments:
        f {function} -- np.array를 return하는 함수
        init_x {numpy.array} -- f 함수의 초기값 array
    
    Keyword Arguments:
        lr {float} -- learning rate (default: {0.01})
        step_num {int} -- 몇 번 반복해서 경사하강을 할지 (default: {100})
    
    Returns:
        numpy.array -- 계산된 값을 반환
    
    >>> def function_2(x):
    >>>   return x[0]**2 + x[1]**2
    >>> init_x = np.array([-3.0, 4.0])
    >>> gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
    """
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x

# import numpy as np
# import matplotlib.pyplot as plt

# x = np.arange(-5.0, 5.0, 0.1)
# y = activation_relu(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = activation_sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = activation_sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z3 = activation_sigmoid(a3)
    
    y = z3
    return y


if __name__ == "__main__":
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)

