import math
import random
import config
import asyncio

from connection import Connection
from typing import Callable, Tuple
from genome import Genome
from rng import RNG
from species import Species


class Neat:

    innovation_number = 1

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
        self.population = [self.mutate(Genome(num_inputs, num_outputs)) for _ in range(self.population_size)]
        self.best_genome = self.population[0]
        self.fitness_function = fitness_function
        self.species: list[Species] = []
        


    async def create_generation(self):
        # print(f"population size: {len(self.population)}")
        self.separate_species()
        await self.calculate_fitness()
        
        total_average_fitness = 0.0
        for genome in self.population:
            total_average_fitness += genome.fitness
            if genome.fitness > self.best_genome.fitness:
                self.best_genome = genome
        total_average_fitness /= len(self.population)

        # print(f"total_average_fitness: {total_average_fitness}")
        

        new_population = []
        # print(f"Species: {len(self.species)}")
        for species in self.species:
            if len(species.genomes) == 0:
                self.species.remove(species)
                continue
            species.age += 1
            species.to_produce = math.floor((total_average_fitness / species.get_average_fitness()) * self.population_size) 
            if species.to_produce == 0: 
                self.species.remove(species)
                continue
            genome_pool = species.get_top_of_species()
            if len(genome_pool) == 0: 
                self.species.remove(species)
                continue
            if len(species.genomes) >= 5:
                new_population.append(genome_pool[-1])
                species.to_produce -= 1
            for _ in range(species.to_produce):
                if RNG.should_mutate_without_crossover():
                    new_population.append(self.mutate(random.choice(genome_pool)))
                else:
                    new_population.append(self.mutate(self.cross_over(random.choice(genome_pool), random.choice(genome_pool))))
            species.to_produce = 0

        self.population = new_population

 
    def separate_species(self):
        self.species = [Species(self.population[0])]
        for genome in self.population:
            found_species = False
            for species in self.species:
                random_genome = species.random_genome()
                distance = self.calculate_distance(genome, random_genome)
                if distance < config.COMPATABILITY_DISTANCE_THRESHOLD:
                    species.add_genome(genome)
                    found_species = True
                    break
            if not found_species:
                self.species.append(Species(genome))

        if len(self.species) is not config.TARGET_SPECIES_SIZE:
            config.TARGET_SPECIES_SIZE += config.THRESHOLD_DELTA * (1 if len(self.species) > config.TARGET_SPECIES_SIZE else -1)


    def create_connection(self, node_numbers: Tuple[int, int], weight: float) -> Connection:
        if tuple(node_numbers) in Neat.paired_nodes.keys():
            return Connection(*node_numbers, weight, True, Neat.paired_nodes.get(tuple(node_numbers)))
        else:
            con = Connection(*node_numbers, weight, True, self.innovation_number)
            Neat.paired_nodes.update({tuple(node_numbers): self.innovation_number})
            self.innovation_number += 1
            return con


    async def calculate_fitness(self) -> float:
        for fitness_function in asyncio.as_completed([self.fitness_function(genome) for genome in self.population]):
            await fitness_function


    def calculate_distance(self, first: Genome, second: Genome) -> float:
        gene_differences = 0
        weight_differences = 0
        shared_weights = 0.0
        genes = first.connections.union(second.connections)
        for gene in genes:
            if first.has_connection(gene) ^ second.has_connection(gene):
                gene_differences += 1
            else:
                # If the first(or second) genome has the connection, then they
                # must both have the gene because of the logical XOR.
                if first.has_connection(gene) and second.has_connection(gene):
                    weight_differences += (
                        first.get_connection_weight(gene) + second.get_connection_weight(gene)
                    )
                    shared_weights += 1.0
        return gene_differences + ((config.C3 * (weight_differences / (shared_weights*2))) if shared_weights > 0.0 else 0)


    def cross_over(self, first: Genome, second: Genome) -> Genome:
        first_fitness = self.calculate_fitness(first)
        total_fitness = first_fitness + self.calculate_fitness(second)
        first_inherit_prob = first_fitness / total_fitness
        genes = first.connections.union(second.connections)
        new_connections = set(self._determine_gene(first, second, connection, first_inherit_prob) for connection in genes if first.has_connection(connection) or second.has_connection(connection))
        return Genome(self.num_inputs, self.num_outputs, connections=new_connections)


    def _determine_gene(self, first: Genome, second: Genome, connection: Connection, first_inherit_prob: float):
        connection = None
        if first.has_connection(connection) and second.has_connection(connection):
            if RNG.should_inherit_average_weight():
                connection = self._calculate_connection_enabled(first.get_connection(connection), other_enabled=second.get_connection(connection).enabled)
                connection.weight = (first.get_connection_weight(connection) + second.get_connection_weight(connection)) / 2.0
            else: 
                if random.random() < first_inherit_prob:
                    connection = self._calculate_connection_enabled(first.get_connection(connection), other_enabled=second.get_connection(connection).enabled)
                else:
                    connection = self._calculate_connection_enabled(second.get_connection(connection), other_enabled=first.get_connection(connection).enabled)
        if first.has_connection(connection):
            connection = self._calculate_connection_enabled(first.get_connection(connection))
        if second.has_connection(connection):
            connection = self._calculate_connection_enabled(second.get_connection(connection))
        return connection


    def _calculate_connection_enabled(self, connection: Connection, other_enabled: bool=False) -> Connection:
        new_connection = connection.copy()
        new_connection.enabled = (not other_enabled or not connection.enabled) and RNG.should_disabled_connection_be_inherited()
        return new_connection


    def mutate(self, genome: Genome) -> Genome:
        if RNG.should_weights_change():
            self._mutate_connections(genome)
        elif RNG.should_connection_be_added():
            print('here1')
            node_list = genome.get_node_list()
            nodes = random.choices(node_list, k=2)
            
            while not self._is_valid_connection(genome, *nodes):
                nodes = random.choices(node_list, k=2)

            genome.add_connection(self.create_connection(nodes, random.random() * 2 - 1))
        elif RNG.should_node_be_added():            
            if len(genome.connections) == 0: return
            print('here2')
            connection_to_split = random.choice(list(genome.connections))
            connection_to_split.enabled = False
            genome.add_connection(self.create_connection((connection_to_split.first, max(genome.get_node_list()) + 1), 1.0))
            genome.add_connection(self.create_connection((max(genome.get_node_list()) + 1, connection_to_split.second), connection_to_split.weight))
        return genome


    def _mutate_connections(self, genome: Genome) -> None:
        for connection in genome.connections:
            if RNG.should_weights_be_perturbed():
                connection.weight += random.gauss(0, 0.1)
            else:
                connection.weight = random.random() * 2 - 1
            

    def _is_valid_connection(self, genome: Genome, first_node: int, second_node: int):
        return (first_node != second_node) and ((first_node in genome.nodes["inputs"] and (second_node in genome.nodes["hidden"] or second_node in genome.nodes["outputs"])) or (first_node in genome.nodes["hidden"] and second_node in genome.nodes["outputs"]))