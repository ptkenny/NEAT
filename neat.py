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
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population_size = population_size
        self.population = [Genome(num_inputs, num_outputs) for _ in range(self.population_size)]
        self.fitness_function = fitness_function
        self.species = [[self.population[0]]]


    def create_generation(self):
        total_average_fitness = 0.0
        genomes_per_species = {}

        for genome in self.population:
            genome.fitness = self.fitness_function(genome)
            total_average_fitness += genome.fitness
        for top_of_species in self.species:
            top_of_species.sort(key=lambda genome: genome.fitness)
            genomes_per_species.update({(sum(s / len(top_of_species) for s in top_of_species) / total_average_fitness): top_of_species[:top_of_species[:int(len(top_of_species) * config.TOP_SPECIES_PERCENT)]]})
        
        new_population = []
        for crossover_per_species, top_of_species in genomes_per_species.items():
            for x in range(crossover_per_species):
                first_genome, second_genome = random.choices(top_of_species, k=2)
                new_population.append(self.mutate(self.cross_over(first_genome, second_genome)))
        
        


    def separate_species(self):
        for genome in self.population:
            for species in self.species:
                random_genome = random.choice(species)
                if self.calculate_distance(genome, random_genome) < config.COMPATABILITY_DISTANCE_THRESHOLD:
                    species.append(genome)

        if len(self.species) is not config.TARGET_SPECIES_SIZE:
            config.TARGET_SPECIES_SIZE += config.THRESHOLD_DELTA * (1 if len(self.species) > config.TARGET_SPECIES_SIZE else -1)            

                
        

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
        genome.fitness = self.fitness_function(genome)
        return genome.fitness


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
        new_connections: set = set(self._determine_gene(first, second, x, first_inherit_prob) for x in range(1, self.innovation_number + 1) if first.has_connection(x) or second.has_connection(x))
        return Genome(self.num_inputs, self.num_outputs, connections=new_connections)


    def _determine_gene(self, first: Genome, second: Genome, innovation: int, first_inherit_prob: float):
        connection = None
        if first.has_connection(innovation) and second.has_connection(innovation):
            if RNG.should_inherit_average_weight:
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
        new_connection.enabled = (not other_enabled or not connection.enabled) and RNG.should_disabled_connection_be_inherited
        return new_connection


    def mutate(self, genome: Genome) -> Genome:
        if RNG.should_weights_change:
            self._mutate_connections(genome)
        return genome


    def _mutate_connections(self, genome: Genome) -> None:
        for connection in genome.connections:
            if RNG.should_weights_be_perturbed:
                connection.weight += random.gauss(0, 0.5)
            else:
                connection.weight = random.random() * 2 - 1
        
        if RNG.should_connection_be_added:
            node_list = genome.get_node_list()
            nodes = random.choices(node_list, k=2)
            
            while not self._is_valid_connection(genome, *nodes):
                nodes = random.choices(node_list, k=2)

            genome.connections.add(self.create_connection(nodes, random.random() * 2 - 1))
        
        if RNG.should_node_be_added:
            if len(genome.connections) is 0: return
            connection_to_split = random.choice(genome.connections)
            connection_to_split.enabled = False
            new_node = self.create_node()
            genome.connections.add(self.create_connection((connection_to_split.first, new_node), 1.0))
            genome.connections.add(self.create_connection((new_node, connection_to_split.second), connection_to_split.weight))

    def _is_valid_connection(self, genome: Genome, first_node: int, second_node: int):
        return (first_node in genome.nodes["inputs"] and (second_node in genome.nodes["hidden"] or second_node in genome.nodes["output"])) or (first_node in genome.nodes["hidden"] and second_node in genome.nodes["output"])