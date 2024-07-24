import time
 
t = time.localtime(time.time())
localtime = time.asctime(t)
str = "Current Time:" + time.asctime(t)
print(str)

def dot(a, b, y=False):
    def skalar_matrika(s, m):
        return [[s*a for a in vrstica] for vrstica in m]
    if y == "skalar":
        return skalar_matrika(a, b)
    def vector_to_matrix(v1, v2):
        matrix = [[v1[i] * v2[j] for i in range(len(v1))] for j in range(len(v2))]
        return matrix
    if y:
        return vector_to_matrix(a, b)
    def dot_product(vec1, vec2):
        """Izračuna dot produkt med dvema vektorjema."""
        if len(vec1) != len(vec2):
            raise ValueError("Oba vektorja morata biti enake dolžine")
        return sum(a * b for a, b in zip(vec1, vec2))
    
    def matrix_vector_product(matrix, vector):
        """Izračuna produkt matrike in vektorja."""
        if len(matrix[0]) != len(vector):
            raise ValueError("Število stolpcev matrike mora ustrezati dolžini vektorja")
        return [dot_product(row, vector) for row in matrix]
    
    def matrix_matrix_product(matrix1, matrix2):
        """Izračuna produkt dveh matrik."""
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("Število stolpcev prve matrike mora ustrezati številu vrstic druge matrike")
        
        matrix2_T = list(zip(*matrix2))
        return [[dot_product(row1, row2) for row2 in matrix2_T] for row1 in matrix1]
    
    # Preverimo, kakšen tip podatkov imamo
    if isinstance(a[0], list):
        # 'a' je matrika, preverimo, ali je 'b' vektor
        if isinstance(b, list) and not isinstance(b[0], list):
            return matrix_vector_product(a, b)
        elif isinstance(b[0], list):
            return matrix_matrix_product(a, b)
        else:
            raise ValueError("Neveljavni podatki za matriko in vektor/matriko.")
    elif isinstance(b, list) and not isinstance(b[0], list):
        # 'a' in 'b' sta oba vektorja
        return dot_product(a, b)
    else:
        raise ValueError("Neveljavni podatki. Prepričajte se, da so vhodni podatki vektorji ali matrike.")


def sestej_matriki(matrix1, matrix2, znak = "+"):
    # Preverimo, če imata matriki enake dimenzije
    #if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        #raise ValueError("Matriki morata biti enakih dimenzij")
    if znak == "-":
        return [[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]
            
    return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(matrix1, matrix2)]
 


def transponiranje(matrix):
    """Transponira matriko."""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]



import random
import math
def normalna_distribucija(x):
    return math.exp(-pow(x, 2))
def aktivacijska_f(z):
    return 1/(1+math.exp(-z))
def aktivacijska_f_odvod(z):
    return aktivacijska_f(z)*(1 - aktivacijska_f(z))


  
  
  
  
  
class Omrezje:
    
    def __init__(self, plasti):
        self.plasti_dolzina = len(plasti)
        self.plasti = plasti
        self.utezi = [
            [
                [random.uniform(-0.5, 0.5) for _ in range(x)] 
                for _ in range(y)]
                    for x, y in zip(plasti[:-1], plasti[1:])
        ]
        self.pristranskost = [[random.uniform(-0.5, 0.5) for _ in range(x)] for x in plasti[1:]]

    def izhod(self, x):
        for u, p in zip(self.utezi, self.pristranskost):
            x = [aktivacijska_f(a + b) for (a,b) in zip(dot(u, x), p)]
        return x    
        
    def vzvratno_razsirjanje(self, x, y):
        parciali_u = [0 for _ in range(self.plasti_dolzina - 1)]
        parciali_p = [0 for _ in range(self.plasti_dolzina - 1)]
        aktivacija = x
        aktivacije = [x]
        z_vrednosti = []
        
        for u, p in zip(self.utezi, self.pristranskost):
            x = [aktivacijska_f(a + b) for (a,b) in zip(dot(u, x), p)]
            z_vrednosti.append(x)
            aktivacija = [aktivacijska_f(a) for a in x]
            aktivacije.append(aktivacija)
        
        delta = [(a-b)*aktivacijska_f_odvod(c) for (a,b,c) in zip(aktivacije[-1], y, z_vrednosti[-1])] ###########
        parciali_p[-1] = delta
        parciali_u[-1] = dot(aktivacije[-2], delta, True)
       
        for l in range(2, self.plasti_dolzina):
            z = z_vrednosti[-l]
            delta = [a*aktivacijska_f_odvod(b) for (a,b) in zip(dot(transponiranje(self.utezi[1-l]), delta), z)]
            parciali_p[-l] = delta
            parciali_u[-l] = dot(aktivacije[-1-l], delta, True)
        
        return (parciali_u, parciali_p)    
    
    def gradientni_spust(self, mali_nabor, eta):
        nabor_parcialov_u = [
            [
                [0 for _ in range(x)] 
                for _ in range(y)]
                    for x, y in zip(self.plasti[:-1], self.plasti[1:])
        ]
        nabor_parcialov_p = [[0 for _ in range(x)] for x in self.plasti[1:]]
        
        for x, y in mali_nabor:
            parciali_u, parciali_p = self.vzvratno_razsirjanje(x, y)
            nabor_parcialov_u = [sestej_matriki(a, b) for a, b in zip(nabor_parcialov_u, parciali_u)]
            nabor_parcialov_p = [[v1[i]+v2[i] for i in range(len(v1))] for v1, v2 in zip(nabor_parcialov_p, parciali_p)]
        m = len(mali_nabor)
        self.utezi = [sestej_matriki(u, dot((eta/m), npu, "skalar"), "-") for u, npu in zip(self.utezi, nabor_parcialov_u)]
        self.pristranskost = [[p - (eta/m)*a for a, p in zip(vektor, p_vektor)] for p_vektor,vektor in zip(self.pristranskost, nabor_parcialov_p)]
        #print("Po posodobitvi uteži:", self.utezi)
        #print("Po posodobitvi pristranskosti:", self.pristranskost)
        
        
    def funkcija_cene(self, podatki_za_ucenje):
        n = len(podatki_za_ucenje)
        return (1/(2*n))*sum([sum([(y - a)**2 for a, y in zip(self.izhod(vektor_a), vektor_y)]) for vektor_a, vektor_y in podatki_za_ucenje])
    
    def ucenje(self, podatki_za_ucenje, epoch, eta, velikost_malih_naborov, testni_podatki=None):
        if testni_podatki:
            nt = len(testni_podatki)
        n = len(podatki_za_ucenje)
        for j in range(epoch):
            random.shuffle(podatki_za_ucenje)
            mali_nabori = [podatki_za_ucenje[k:k + velikost_malih_naborov] for k in range(0, n, velikost_malih_naborov)] 
            for mali_nabor in mali_nabori:
                self.gradientni_spust(mali_nabor, eta)
            
            if j % 2 == 0:
                print(self.funkcija_cene(podatki_za_ucenje))    
            if testni_podatki:
                print("definiraj samo evaluacijo")
                
        #print(self.pristranskost[-1])
                
                
                
            
                    
       
       
       
       
       
       
       
       
       
       
       
       
       
       
        
omrezje = Omrezje([2,2, 1])
print(len(omrezje.pristranskost[0])) 
print(len(omrezje.utezi[0]))
#print(dot([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],[1,2,2,1]))



#print(dot([1,2,3],[1,2,1],True))
print("dd")
#print(omrezje.vzvratno_razsirjanje([1,1], [1]))

print("pavaza")

#print(omrezje.funkcija_cene([([1, 1], [1]), ([0,0],[1])]))
pod = [([1, 1], [1]), ([0,0], [0])]
podatki_za_ucenje = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
xor_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]





#ucenje(self, podatki_za_ucenje, epoch, eta, velikost_malih_naborov, testni_podatki=None
omrezje.ucenje(podatki_za_ucenje,100, 2, 2)
print(omrezje.izhod([1,0]))

#print(sestej_matriki(dot(3, [[1,1,1],[1,1,1]], "skalar"), [[0,0,0],[0,2,1]], "-"))
