class Omrezje:

    def exp(self,x):
        e = 2.718281828459045235360287471352
        def f(n):
            if n <= 1:
                return 1
            return n*f(n-1)
        def ex(x):
            return sum([(x**n)/f(n) for n in range(10)])    
        if abs(x)>1:
            return 1/((e**int(x))*ex(abs(x-int(x))))
        return ex(-x)    
    
    def uniform(self, a, b):
        self.seme = (1103515245 * self.seme + 12345) % (2**31)
        return a + (self.seme / (2**31)) * (b - a)
    
    def zmesaj(self, lst):
        n = len(lst)
        for i in range(n-1, 0, -1):
            j = int(self.uniform(0, 1) * (i + 1))
            lst[i], lst[j] = lst[j], lst[i]
            
    def sigmoid(self, z):
        return 1 / (1 + self.exp(z))
    
    def sigmoid_odvod(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    
    def __init__(self, sloji, seme=42):
        self.seme = seme
        self.sloji = sloji
        self.utezi = [[[self.uniform(-0.5, 0.5) for _ in range(x)] for _ in range(y)] for (x, y) in zip(sloji[:-1], sloji[1:])]
        self.pristranskost = [[self.uniform(-0.5, 0.5) for _ in range(y)] for y in sloji[1:]]
    
    def iskanje_parcialov(self, x, y):
        parciali_u = [0 for _ in range(len(self.sloji) - 1)]
        parciali_p = [0 for _ in range(len(self.sloji) - 1)]
        
        aktivacija = x
        aktivacije = [x]
        zv = []
        for u, p in zip(self.utezi, self.pristranskost):
            z = [sum([element1 * element2 for element1, element2 in zip(vrstica, aktivacije[-1])]) + element_p for (vrstica, element_p) in zip(u, p)]
            zv.append(z)
            aktivacija = [self.sigmoid(element) for element in z]
            aktivacije.append(aktivacija)
        
        delta = [(element1 - element2) * self.sigmoid_odvod(z) for (element1, element2, z) in zip(aktivacije[-1], y, zv[-1])]
        parciali_p[-1] = delta
        parciali_u[-1] = [[d * a for a in aktivacije[-2]] for d in delta]
        
        for l in range(2, len(self.sloji)):
            delta = [sum([self.utezi[-l+1][i][j] * delta[i] for i in range(len(self.utezi[-l+1]))]) for j in range(len(self.utezi[-l+1][0]))]
            sp = [self.sigmoid_odvod(z) for z in zv[-l]]
            delta = [d * sp for d, sp in zip(delta, sp)]
            parciali_p[-l] = delta
            parciali_u[-l] = [[d * a for a in aktivacije[-l-1]] for d in delta]
        
        return (parciali_u, parciali_p)
    
    def gradientni_spust(self, mali_nabor, eta):
        nabor_parcialov_u = [[[0 for _ in range(x)] for _ in range(y)] for (x, y) in zip(self.sloji[:-1], self.sloji[1:])]
        nabor_parcialov_p = [[0 for _ in range(y)] for y in self.sloji[1:]]
        m = len(mali_nabor)
        
        for (x, y) in mali_nabor:
            parcial_u, parcial_p = self.iskanje_parcialov(x, y)
            nabor_parcialov_p = [[a + b for (a, b) in zip(vektor1, vektor2)] for (vektor1, vektor2) in zip(nabor_parcialov_p, parcial_p)]
            nabor_parcialov_u = [[[a + b for (a, b) in zip(vrstica1, vrstica2)] for (vrstica1, vrstica2) in zip(pu, npu)] for (pu, npu) in zip(parcial_u, nabor_parcialov_u)]
        
        self.pristranskost = [[a - (eta / m) * b for (a, b) in zip(p, npp)] for (p, npp) in zip(self.pristranskost, nabor_parcialov_p)]
        self.utezi = [[[a - (eta / m) * b for (a, b) in zip(vrstica1, vrstica2)] for (vrstica1, vrstica2) in zip(u, npu)] for (u, npu) in zip(self.utezi, nabor_parcialov_u)]
    
    def ucenje(self, trening, velikost_naborov, epoch, eta, posebno=None):
        
        n = len(trening)
        for e in range(epoch):
            self.zmesaj(trening)
            mali_nabori = [trening[i:i + velikost_naborov] for i in range(0, n, velikost_naborov)]
            for mali_nabor in mali_nabori:
                self.gradientni_spust(mali_nabor, eta)
            print(f"epoch: {e}, cena: {self.cena(trening)}")
        self.test(trening, posebno)
    
    def izhod(self, x):
        for (u, p) in zip(self.utezi, self.pristranskost):
            x = [self.sigmoid(sum([element1 * element2 for element1, element2 in zip(vrstica, x)]) + element_p) for (vrstica, element_p) in zip(u, p)]
        return x
    
    def cena(self, trening):
        return (1 / (2 * len(trening))) * sum([sum([(k1 - k2) ** 2 for (k1, k2) in zip(self.izhod(x), y)]) for (x, y) in trening]) ** 0.5
        
    def test(self, trening, posebno=None):
        for (x,y) in trening[:30]:
            if posebno:
                posebno(x, self.izhod(x), y)
            else:
                print(f"{x} -> {self.izhod(x)}, y(x) = {y}")  



      

#######################################################
#Omrezje 1:
def omrezje_xor():
    trening_xor = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    omrezje = Omrezje([2, 2, 1])
    omrezje.ucenje(trening_xor, velikost_naborov=4, epoch=1000, eta=6)

#Za zagon odkomentiraj:
#omrezje_xor()

######################################################
######################################################
######################################################
#Omrezje 2:
def omrezje_parabola():
    
    def binarni_zapis(n, l):
        return bin(n)[2:].zfill(l)
    
    def decimalni_zapis(vektor):
        stevilo = "".join([str(v) for v in vektor])
        stevilo = stevilo.lstrip("0")
        if stevilo == "":
            return 0
        return int(stevilo, 2)
    
    omrezje = Omrezje([5, 70, 30, 10])
    trening_parabola = []
    for i in range(1, 31 + 1):
        x = [int(cifra) for cifra in str(binarni_zapis(i, 5))]
        y = [int(cifra) for cifra in str(binarni_zapis(i**2, 10))]
        trening_parabola.append((x, y))
        
    def posebno(x, x_izhod, y):
        stevilo_x = decimalni_zapis(x)
        x_izhod = decimalni_zapis([round(element) for element in x_izhod])
        stevilo_y = decimalni_zapis(y)
        print(f"{stevilo_x} -> {x_izhod}, y(x) = {stevilo_y}") 
        
    omrezje.ucenje(trening_parabola, velikost_naborov=20, epoch=300, eta=20, posebno=posebno)
    
#Za zagon odkomentiraj:
#omrezje_parabola()

############################################################
