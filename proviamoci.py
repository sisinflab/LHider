import numpy as np
from math import prod, factorial

def disp(n, d):
    # n elementi in d posti
    assert n <= d, f'errore: n={n}, d={d}'
    return int(prod(range(n+1, d+1))/factorial(d-n))


class User:
    def __init__(self, items, ratings):
        self.items = items
        self.ratings = ratings
        self.array = np.zeros(self.items)
        self.dispositions = disp(self.ratings, self.items)

    def __str__(self):
        return str(self.array)

    def generate_permutation(self, n):
        assert n < self.dispositions

        positions = []
        resto = n
        elements = self.ratings
        posti = self.items
        last_position = 0

        for item in range(self.ratings - 1):
            resto, pos = self.element_position(resto, elements, posti)
            positions.append(pos + last_position)
            last_position = pos + last_position + 1
            elements = elements - 1
            posti = posti - pos - 1

        # posizione ultimo elemento
        assert resto < posti, f'il resto {resto} è maggiore del numero di posti {posti}'
        positions.append(int(resto) + last_position)
        return positions

    def permutation_to_array(self, permutations:list):
        new_array = self.array.copy()
        new_array[permutations] = 1
        return new_array

    @staticmethod
    def element_position(n, tot_elements, posti):
        pos = 0
        new_discard = disp(tot_elements - 1, posti - 1 - pos)
        discard = new_discard
        while n - discard >= 0:
            pos += 1
            new_discard = disp(tot_elements - 1, posti - 1 - pos)
            discard += new_discard
        return n - discard + new_discard, pos

u0 = User(4, 2)
u1 = User(4, 3)
u2 = User(4, 1)
u3 = User(4, 2)

users = [u0, u1, u2, u3]

for u in users:
    print(u.dispositions)
