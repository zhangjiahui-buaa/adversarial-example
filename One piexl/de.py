import numpy as np


class candidate():
    def __init__(self, x, y, R, G, B):
        self.x = x
        self.y = y
        self.R = R
        self.G = G
        self.B = B

    def get_loc(self):
        return self.x, self.y

    def get_perturbation(self):
        return self.R, self.G, self.B

    def __add__(self, other):
        return candidate(int(self.x + other.x), int(self.y + other.y), self.R + other.R, self.G + other.G, self.B + other.B)

    def mult(self, F):
        return candidate(self.x * F, self.y * F, self.R * F, self.G * F, self.B * F)



def DE(parent):
    child = []
    for i in range(400):
        r_1 = int(np.random.uniform(0, 400))
        r_2 = int(np.random.uniform(0, 400))
        while r_2 == r_1:
            r_2 = int(np.random.uniform(0, 400))
        r_3 = int(np.random.uniform(0, 400))
        while (r_3 == r_2 or r_3 == r_1):
            r_3 = int(np.random.uniform(0, 400))
        child_i = parent[r_1]+parent[r_2].mult(-0.5)+parent[r_3].mult(-0.5)
        child.append(child_i)
    return parent,child

