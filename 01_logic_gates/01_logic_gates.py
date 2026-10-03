import numpy as np

#step함수
def step_logicgate():
    def step(z):
        if z >= 0:
            return 1
        else:
            return 0

    def perceptron(A, B, w1, w2, b):
        z = w1*A+w2*B+b
        y = step(z)
        return y

    print(f"\n #### 1. STEP 함수 ####")
    for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y_and = perceptron(x1, x2, 0.5, 0.5, -0.7)
        y_or = perceptron(x1, x2, 1, 1, -0.5)
        y_nand = perceptron(x1, x2, -1, -1, 1.5)
        y_xor = perceptron(y_or, y_nand, 0.5, 0.5, -0.7)
        
        print(f"입력:({x1},{x2}) | AND:{y_and} | OR:{y_or} | NAND:{y_nand} | XOR:{y_xor}")

#2. sigmoid
def sigmoid_logicgate():
    def sigmoid(z):
        y = 1/(1+np.exp(-z))
        return y

    def perceptron(A, B, w1, w2, b):
        z = w1*A + w2*B +b
        y = sigmoid(z)
        return y

    def decision(v):
        if v>=0.5:
            return 1
        else:
            return 0

    print(f"\n#### 2. SIGMOID 함수 ####")
    for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y_and = perceptron(x1, x2, 10, 10, -15)
        y_or = perceptron(x1, x2, 10, 10, -5)
        y_nand = perceptron(x1, x2, -10, -10, 15)
        y_xor = perceptron(y_or, y_nand, 10, 10, -15)
        
        print(f"입력:({x1},{x2}) | AND:[{decision(y_and)}, {y_and:.4f}] | OR:[{decision(y_or)}, {y_or:.4f}] | NAND:[{decision(y_nand)}, {y_nand:.4f}] | XOR:[{decision(y_xor)}, {y_xor:.4f}]")

#3. tanh
def tanh_logicgate():
    def tanh(z):
        y = np.tanh(z)
        return y

    def perceptron(A, B, w1, w2, b):
        z = w1*A + w2*B + b
        y = tanh(z)
        return y

    def decision(v):
        if v>= 0:
            return 1
        else:
            return 0

    print(f"\n#### 3. TANH 함수 ####")
    for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y_and = perceptron(x1, x2, 10, 10, -15)
        y_or = perceptron(x1, x2, 10, 10, -5)
        y_nand = perceptron(x1, x2, -10, -10, 15)
        y_xor = perceptron(y_or, y_nand, 10, 10, -15)
        
        print(f"입력:({x1},{x2}) | AND:[{decision(y_and)}, {y_and:.4f}] | OR:[{decision(y_or)}, {y_or:.4f}] | NAND:[{decision(y_nand)}, {y_nand:.4f}] | XOR:[{decision(y_xor)}, {y_xor:.4f}]")

#4.ReLU
def relu_logicgate():
    def relu(z):
        if z > 0:
            return z
        else:
            return 0.0

    def perceptron(A, B, w1, w2, b):
        z = w1*A + w2*B + b
        y = relu(z)
        return y

    def decision(v):
        if v > 0:
            return 1
        else:
            return 0

    print(f"\n#### 4. RELU 함수 결과 ####")
    for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y_and = perceptron(x1, x2, 1, 1, -0.7)
        y_or = perceptron(x1, x2, 1, 1, -0.5)
        y_nand = perceptron(x1, x2, -1, -1, 1.5)
        y_xor = perceptron(y_or, y_nand, 1, 1, -0.7)
        
        print(f"입력:({x1},{x2}) | AND:[{decision(y_and)}, {y_and:.2f}] | OR:[{decision(y_or)}, {y_or:.2f}] | NAND:[{decision(y_nand)}, {y_nand:.2f}] | XOR:[{decision(y_xor)}, {y_xor:.2f}]")

step_logicgate()
sigmoid_logicgate()
tanh_logicgate()
relu_logicgate()