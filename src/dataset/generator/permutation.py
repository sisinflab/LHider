import random
import numpy as np
from math import prod, factorial
from src.laplace_mechanism import LaplaceMechanism
import tqdm

class Generator:

    def __init__(self, data, random_seed):
        self.data = data
        self.seed = random_seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def generate(self, new_seed=0):
        seed = self.seed + new_seed
        random.seed(seed)
        np.random.seed(seed)
        n = self.data.items
        gen = dict()
        for u in tqdm.tqdm(range(self.data.users), desc=f'Generating a new dataset with seed {seed}'):
            elmts = len(self.data[u].data)
            gen[u] = self.generate_user(n, elmts)
        return gen

    def generate_user(self, n_items, n_ratings):
        user = []
        items = list(range(n_items))
        for r in range(n_ratings):
            new_item = random.choice(items)
            items.remove(new_item)
            user.append(new_item)
        return user


class BakGenerator:

    def __init__(self, data, random_seed):
        self.data = data
        self.seed = random_seed
        random.seed(self.seed)
        np.random.seed(self.seed)

    def generate(self):
        print('generating a new dataset')
        n = self.data.items
        gen = dict()
        for u in tqdm.tqdm(range(self.data.users)):
            elmts = len(self.data[u].data)
            d = ArrayDispositions(length=n, elements=elmts)
            disposition = random.randint(0, d.dispositions-1)
            print(d.disposition_to_array(d.generate_dispositions(disposition)))
        print(gen)


class ArrayDispositions:

    def __init__(self, length, elements):
        self.length = length
        self.elements = elements
        self.array = np.zeros(self.length)
        self.dispositions = disp(self.elements, self.length)

    def __str__(self):
        return str(self.array)

    def generate_dispositions(self, n):
        assert n < self.dispositions

        positions = []
        resto = n
        elements = self.elements
        posti = self.length
        last_position = 0

        for item in range(self.elements - 1):
            resto, pos = self.element_position(resto, elements, posti)
            positions.append(pos + last_position)
            last_position = pos + last_position + 1
            elements = elements - 1
            posti = posti - pos - 1

        # posizione ultimo elemento
        assert resto < posti, f'il resto {resto} è maggiore del numero di posti {posti}'
        positions.append(int(resto) + last_position)
        return positions

    def disposition_to_array(self, dispositions:list):
        new_array = self.array.copy()
        new_array[dispositions] = 1
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


def disp(n, d):
    # n num of elements
    # d num of places
    assert n <= d, f'Computing dispositions error: elements are greater than available places n={n} > d={d}'
    f = prod(range(n+1, d+1)) // factorial(d-n)
    return f
