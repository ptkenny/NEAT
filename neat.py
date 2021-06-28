import random
import config

from connection import Connection
from typing import Callable, Tuple
from genome import Genome


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
        fitness_function: Callable(Genome),
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

    def calculate_fitness(self) -> float:
        map(self.fitness_function, self.population)

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
        return gene_differences + (Neat.C3 * (weight_differences / shared_weights))

    def cross_over(self, first: Genome, second: Genome) -> Genome:
        first_fitness = self.calculate_fitness(first)
        total_fitness = first_fitness + self.calculate_fitness(second)
        first_inherit_prob = first_fitness / total_fitness
        new_connections: set = set()
        for x in range(1, self.innovation_number + 1):
            connection = None
            if first.has_connection(x) and second.has_connection(x):
                if random.random() < config.AVERAGE_WEIGHT_INHERITANCE_PROBABILITY:
                    connection = self._calculate_connection_enabled(first.get_connection(x), other_enabled=second.get_connection(x).enabled)
                    connection.weight = (first.get_connection(x).weight + second.get_connection(x).weight) / 2.0
                else: 
                    if random.random() < first_inherit_prob:
                        connection = self._calculate_connection_enabled(first.get_connection(x), other_enabled=second.get_connection(x).enabled)
                    else:
                        connection = self._calculate_connection_enabled(second.get_connection(x), other_enabled=first.get_connection(x).enabled)
            elif first.has_connection(x):
                connection = self._calculate_connection_enabled(first.get_connection(x))
            elif second.has_connection(x):
                connection = self._calculate_connection_enabled(second.get_connection(x))
            if connection is not None: new_connections.add(connection)
        return Genome(self.num_inputs, self.num_outputs, connections=new_connections)

    def _calculate_connection_enabled(self, connection: Connection, other_enabled: bool=False) -> Connection:
        new_connection = connection.copy()
        new_connection.enabled = (not other_enabled or not connection.enabled) and random.random() < config.DISABLED_CONNECTION_INHERITANCE_PROBABILITY
        return new_connection

    def mutate(self, genome: Genome) -> Genome:
        # If the weights of the genome should change
        if random.random() < config.WEIGHT_CHANGE_PROBABILITY:
            self._mutate_connections(genome)

    def _mutate_connections(self, genome: Genome) -> None:
        for connection in genome.connections:
            # If the weight should be uniformly perturbed or completely changed.
            if random.random() < config.NORMAL_WEIGHT_CHANGE_PROBABILITY:
                connection.weight += random.gauss(0, 0.5)
            else:
                connection.weight = random.random() * 2 - 1
