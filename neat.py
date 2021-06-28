import random
import config

from connection import Connection
from typing import Callable, Tuple
from genome import Genome
from rng import RNG


class Neat:

    innovation_number = 1
    node_number = 1
    # A list of connections and their respective innovation numbers.
    paired_nodes = {Tuple: int}

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        population_size: int,
        fitness_function: Callable=lambda x: x,
    ) -> None:
        self.species = []
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population_size = population_size
        self.population = [Genome(num_inputs, num_outputs) for _ in range(self.population_size)]
        self.fitness_function = fitness_function


    def create_generation(self):
        pass


    def create_node(self):
        Neat.node_number += 1


    def create_connection(self, node_numbers: Tuple[int, int], weight: float) -> Connection:
        if node_numbers in Neat.paired_nodes.keys():
            return Connection(*node_numbers, weight, True, Neat.paired_nodes.get(node_numbers))
        else:
            con = Connection(*node_numbers, weight, True, self.innovation_number)
            Neat.paired_nodes.update({node_numbers: self.innovation_number})
            self.innovation_number += 1
            return con


    def calculate_fitness(self, genome: Genome) -> float:
        return self.fitness_function(genome)


    def calculate_distance(self, first: Genome, second: Genome) -> float:
        gene_differences = 0
        weight_differences = 0
        shared_weights = 0.0
        for x in range(1, self.innovation_number + 1):
            if first.has_connection(x) ^ second.has_connection(x):
                gene_differences += 1
            else:
                # If the first(or second) genome has the connection, then they
                # must both have the gene because of the logical XOR.
                if first.has_connection(x):
                    weight_differences += (
                        first.get_connection(x).weight + second.get_connection(x).weight
                    )
                    shared_weights += 1.0
        return gene_differences + (config.C3 * (weight_differences / shared_weights))


    def cross_over(self, first: Genome, second: Genome) -> Genome:
        first_fitness = self.calculate_fitness(first)
        total_fitness = first_fitness + self.calculate_fitness(second)
        first_inherit_prob = first_fitness / total_fitness
        new_connections: set = set(self._determine_gene(first, second, x, first_inherit_prob) for x in range(1, self.innovation_number + 1))        
        if None in new_connections:
            new_connections.remove(None)
        return Genome(self.num_inputs, self.num_outputs, connections=new_connections)


    def _determine_gene(self, first: Genome, second: Genome, innovation: int, first_inherit_prob: float):
        connection = None
        if first.has_connection(innovation) and second.has_connection(innovation):
            if RNG.should_inherit_average_weight():
                connection = self._calculate_connection_enabled(first.get_connection(innovation), other_enabled=second.get_connection(innovation).enabled)
                connection.weight = (first.get_connection(innovation).weight + second.get_connection(innovation).weight) / 2.0
            else: 
                if random.random() < first_inherit_prob:
                    connection = self._calculate_connection_enabled(first.get_connection(innovation), other_enabled=second.get_connection(innovation).enabled)
                else:
                    connection = self._calculate_connection_enabled(second.get_connection(innovation), other_enabled=first.get_connection(innovation).enabled)
        elif first.has_connection(innovation):
            connection = self._calculate_connection_enabled(first.get_connection(innovation))
        elif second.has_connection(innovation):
            connection = self._calculate_connection_enabled(second.get_connection(innovation))
        return connection


    def _calculate_connection_enabled(self, connection: Connection, other_enabled: bool=False) -> Connection:
        new_connection = connection.copy()
        new_connection.enabled = (not other_enabled or not connection.enabled) and RNG.should_disabled_connection_be_inherited()
        return new_connection


    def mutate(self, genome: Genome) -> Genome:
        if RNG.should_weights_change():
            self._mutate_connections(genome)


    def _mutate_connections(self, genome: Genome) -> None:
        for connection in genome.connections:
            if RNG.should_weights_be_perturbed():
                connection.weight += random.gauss(0, 1)
            else:
                connection.weight = random.random() * 2 - 1
