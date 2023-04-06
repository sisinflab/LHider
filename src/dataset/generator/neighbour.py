import random
from src.dataset.generator.base import Generator
import numpy as np


class NeighboursGenerator(Generator):
    def __init__(self, random_seed=42):
        super(NeighboursGenerator, self).__init__(random_seed)

    def generate(self, dataset, n_variations):

        def select_random_interaction(removed_users=None):
            if removed_users is None:
                removed_users = []
            possible_users = np.delete(np.array(dataset.users), removed_users)
            random_user = np.random.choice(possible_users)
            possible_items = np.delete(np.array(dataset.items), dataset[random_user].nonzero()[1])
            if len(possible_items) == 0:
                random_item = None
            else:
                random_item = np.random.choice(possible_items)
            return random_user, random_item

        self.set_seed()

        new_dataset = dataset.copy_values()
        # variations = []

        for _ in n_variations:
            u, i = None, None
            removed_users = []
            while i is None:
                u, i = select_random_interaction(removed_users=removed_users)
                if i is None:
                    removed_users.append(u)

                if (u, i) not in variations and i is not None:
                    variations.append((u, i))
                    break
                else:
                    removed_users.append(u)

            random_item = np.random.choice(possible_items)
            new_dataset[random_user, random_item] = self.flip_binary_rating(dataset[random_user, random_item])
        return new_dataset

    @staticmethod
    def flip_binary_rating(r):
        return int(not r)


class NeighboursIterative(Generator):
    def __init__(self, random_seed=42):
        super(NeighboursIterative, self).__init__(random_seed)

    def generate(self, dataset, modified_ratings):
        self.set_seed()

        random_user, random_item = None, None

        while (random_user, random_item) in modified_ratings or (random_user, random_item) == (None, None):
            random_user = random.randint(0, dataset.n_users-1)
            random_item = random.randint(0, dataset.n_items-1)
        modified_ratings = (random_user, random_item)
        new_dataset = dataset.copy_values()
        new_dataset[random_user, random_item] = self.flip_binary_rating(dataset[random_user, random_item])
        return modified_ratings, new_dataset

    @staticmethod
    def flip_binary_rating(r):
        return int(not r)
