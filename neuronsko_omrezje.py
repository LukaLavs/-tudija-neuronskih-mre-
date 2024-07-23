import numpy as np
import random
import pickle
import os

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class Omrezje:
    def __init__(self, sizes, name):
        self.layers = len(sizes)
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])]
        self.name = name

    def output(self, a):
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, a) + b
            a = sigmoid(z)
        return a
    
    def back_propagation(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        
        aktivacija = x
        aktivacije = [x]
        z_vrednosti = []
        
        for w, b in zip(self.weights, self.bias):
            z = np.dot(w, aktivacija) + b
            z_vrednosti.append(z)
            aktivacija = sigmoid(z)
            aktivacije.append(aktivacija)
            
        #odvod cost funkcije po zadnji z vrednosti 
        #delta=partialC/partialz=(partialC/partiala)*(partiala/partialz)
        delta = (aktivacije[-1] - y) * sigmoid_derivative(z_vrednosti[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, aktivacije[-2].transpose())
        
        for l in range(2, self.layers):
            z = z_vrednosti[-l]
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_derivative(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, aktivacije[-l - 1].transpose())
        return nabla_b, nabla_w
    
    def posodobi_omrezje(self, mini_nabor, eta):
        vsote_parcialov_w = [np.zeros(w.shape) for w in self.weights]
        vsote_parcialov_b = [np.zeros(b.shape) for b in self.bias]

        for x, y in mini_nabor:
            parciali_b, parciali_w = self.back_propagation(x, y)
            vsote_parcialov_b = [nb + db for nb, db in zip(vsote_parcialov_b, parciali_b)]
            vsote_parcialov_w = [nw + dw for nw, dw in zip(vsote_parcialov_w, parciali_w)]
        
        m = len(mini_nabor)
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, vsote_parcialov_w)]
        self.bias = [b - (eta / m) * nb for b, nb in zip(self.bias, vsote_parcialov_b)]
        
    def shrani_parametre(self):
        ime_datoteke_pickle = f'parametri_{self.name}.pkl'
        ime_datoteke_txt = f'parametri_{self.name}.txt'
        pot_do_mape = os.path.dirname(os.path.abspath(__file__))
        celotna_pot_pickle = os.path.join(pot_do_mape, ime_datoteke_pickle)
        celotna_pot_txt = os.path.join(pot_do_mape, ime_datoteke_txt)

        with open(celotna_pot_pickle, 'wb') as datoteka:
            pickle.dump(self.weights, datoteka)
            pickle.dump(self.bias, datoteka)
            
        with open(celotna_pot_txt, 'w') as datoteka_txt:
            for w in self.weights:
                np.savetxt(datoteka_txt, w)
            for b in self.bias:
                np.savetxt(datoteka_txt, b)
        print(f"Parametri so bili uspešno shranjeni v {pot_do_mape}.")
    
    def evaluate(self, test_data):
        #test_results = [(self.output(x), y) for (x, y) in test_data]
        return mse_loss(np.array([y for (x, y) in test_data]), np.array([self.output(x) for (x, y) in test_data]))
    
    def sgd(self, training_data, epoch, mini_nabor_velikost, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epoch):
            random.shuffle(training_data)
            mini_nabori = [training_data[i:i + mini_nabor_velikost] for i in range(0, n, mini_nabor_velikost)]
            for mini_nabor in mini_nabori:
                self.posodobi_omrezje(mini_nabor, eta)
            print(f"Epoch {j + 1} končan.")
            if test_data:
                print(f"Testna izguba: {self.evaluate(test_data)}")
        self.shrani_parametre()



nand_data = [
    (np.array([[0], [0]]), np.array([[1]])),
    (np.array([[0], [1]]), np.array([[1]])),
    (np.array([[1], [0]]), np.array([[1]])),
    (np.array([[1], [1]]), np.array([[0]]))
]
# Ustvarjanje in treniranje omrežja za NAND operator
omrezje_nand = Omrezje([2, 10, 10, 1], "nand_operator")
omrezje_nand.sgd(nand_data, 500, 4, 20, nand_data)

# Preverjanje rezultatov
for x, y in nand_data:
    napoved = omrezje_nand.output(x)
    print(f"Vhod: {x.flatten()}, Pričakovan izhod: {y.flatten()}, Napoved: {napoved.flatten()}")
    
xor_data = [
    (np.array([[0], [0]]), np.array([[0]])),
    (np.array([[0], [1]]), np.array([[1]])),
    (np.array([[1], [0]]), np.array([[1]])),
    (np.array([[1], [1]]), np.array([[0]]))
]

omrezje_xor = Omrezje([2, 30, 1], "xor_operator")
omrezje_xor.sgd(xor_data, 500, 4, 20, xor_data)

# Preverjanje rezultatov
for x, y in xor_data:
    napoved = omrezje_xor.output(x)
    print(f"Vhod: {x.flatten()}, Pričakovan izhod: {y.flatten()}, Napoved: {napoved.flatten()}")
