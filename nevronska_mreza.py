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
    
    def ucenje(self, trening, velikost_naborov, epoch, eta, posebno=None, stej_ceno=True):
        
        n = len(trening)
        for e in range(epoch):
            
            self.zmesaj(trening)
            mali_nabori = [trening[i:i + velikost_naborov] for i in range(0, n, velikost_naborov)]
            for mali_nabor in mali_nabori:
                self.gradientni_spust(mali_nabor, eta)
                
            if stej_ceno:
                print(f"epoch: {e}, cena: {self.cena(trening)}")
            else:
                print(e)
                if e%20==0:
                    print(f"epoch: {e}, cena: {self.cena(trening)}")
                    
        self.test(trening, posebno)
    
    def izhod(self, x):
        for (u, p) in zip(self.utezi, self.pristranskost):
            x = [self.sigmoid(sum([element1 * element2 for element1, element2 in zip(vrstica, x)]) + element_p) for (vrstica, element_p) in zip(u, p)]
        return x
    
    def cena(self, trening):
        return (1 / (2 * len(trening))) * sum([sum([(k1 - k2) ** 2 for (k1, k2) in zip(self.izhod(x), y)]) for (x, y) in trening]) ** 0.5
        
    def test(self, trening, posebno=None):
        for (x,y) in trening[:200]:
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
    print("Kaj pa 32^2, tega ni v treningu:")
    print(decimalni_zapis([round(i) for i in omrezje.izhod( [int(cifra) for cifra in str(binarni_zapis(32, 5))])]))
    print("Pravi odgovor pa je 1024")
    
#Za zagon odkomentiraj:
#omrezje_parabola()

######################################################
######################################################
######################################################
#Omrežje 3:

def omrezje_napovedovanje_besed():
    import os
    pot = os.path.dirname(os.path.realpath(__file__))
    pot = os.path.join(pot, "besedilo.txt")
    
    def binarni_zapis(n, l):
        return bin(n)[2:].zfill(l)
    
    def decimalni_zapis(vektor):
        stevilo = "".join([str(v) for v in vektor])
        stevilo = stevilo.lstrip("0")
        if stevilo == "":
            return 0
        return int(stevilo, 2)
    
    c=0 #šteje koliko vrstic želimo vzeti, recimo 30 vrstic
    besedilo = ""
    with open(pot, "r", encoding="utf-8") as file:
        for line in file.readlines():
            if line != "\n":
                if c >= 30:
                    break
                c+=1
                besedilo += " " + line.strip().lower()
    
    besedilo = [a for a in besedilo.split(" ") if a not in [" ", ""]] 
    besedilo = [a[:-1] if a[-1] in ".!,?:" else a for a in besedilo]
    besede = list(set(besedilo))
    n = len(besede)
    if n > 1000:
        print("verjetno potrebuješ večje binarne številke")
        #1000 ima 10 digits v binarnem sistemu
        return
    
    def sifriraj(skupek_besed):
        sifra = []
        for b in skupek_besed:
            if b not in besede:
                if b not in besedilo:
                    print("težava z indeksom")
                    continue
            p = binarni_zapis(besede.index(b), 10) #1000 ima 10 digits v binarnem sistemu
        
            for t in p:
                sifra.append(int(t))
        return sifra
    
    trening = []
    for i in range(5, len(besedilo)):
        trening.append((sifriraj(besedilo[i-5:i]), sifriraj([besedilo[i]])))
    
    omrezje = Omrezje([50, 30, 20, 10])
    def posebno(x, x_izhod, y):
        stevilo_x = [besede[decimalni_zapis("".join([str(a) for a in x[i:i+10]]))] for i in range(0, 50, 10)] #50 ker 1000 ima 10 digits in podamo 5 besed
        x_izhod1 = besede[decimalni_zapis([round(e) for e in x_izhod])]
        stevilo_y = besede[decimalni_zapis([round(e) for e in y])]
        
        if x_izhod1 == stevilo_y:
            print("ugotovil je pravilno")
            print(f"{stevilo_x} -> {x_izhod1}, y(x) = {stevilo_y}") 
            print("######################")  
        else:
            print("zmotil se je")
            print(f"uganil je: {decimalni_zapis([round(e) for e in x_izhod])}")
            print(f"moral bi: {decimalni_zapis([round(e) for e in y])}")
            print("######################")
    #print(len(trening)) #dolžina je ~ 1850
    omrezje.ucenje(trening[:100], velikost_naborov=60, epoch=300, eta=10, posebno=posebno, stej_ceno=True)

#Za zagon odkomentiraj:    
#omrezje_napovedovanje_besed()

############################################################