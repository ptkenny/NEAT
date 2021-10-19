import random
import math
from typing import List

import config
from genome import Genome


class Species:
    def __init__(self, genome: Genome) -> None:
        self.genomes = [genome]
        self.age = 1
        self.to_produce = 0
    
    def random_genome(self) -> Genome:
        if len(self.genomes) == 0: 
            raise RuntimeError()
        return random.choice(self.genomes)

    def get_top_of_species(self) -> List[Genome]:
        self.genomes.sort(key=lambda genome: genome.fitness)
        return self.genomes[math.floor(len(self.genomes) * config.TOP_SPECIES_PERCENT):]

    def get_average_fitness(self) -> float:
        return sum(genome.fitness for genome in self.genomes) / len(self.genomes)

    def add_genome(self, genome: Genome) -> None:
        self.genomes.append(genome)