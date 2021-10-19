from genome import Genome
from neat import Neat
import unittest
import random


class TestGenomes(unittest.TestCase):
    def test_feed_forward(self):
        n = Neat(2, 1, 1)
        genome = n.population[0]
        input_node = 1
        output_node = 3
        genome.add_connection(n.create_connection((input_node, output_node), 0.5))
        results = genome.feed_forward([0, 1])
        print(f"results: {results}")
        self.assertTrue(all(result >= -1 and result <= 1 for result in results))

    def test_multiple_connections_to_same_node(self):
        n = Neat(2, 2, 1)
        genome = random.choice(n.population)
        input_nodes = genome.nodes.get("inputs")
        output_nodes = genome.nodes.get("outputs")
        genome.connections.add(n.create_connection((input_nodes[0], output_nodes[0]), 0.5))
        genome.connections.add(n.create_connection((input_nodes[1], output_nodes[0]), 0.25))
        results = genome.feed_forward([10, 25])
        self.assertTrue(all(result >= -1 and result <= 1 for result in results))

    def test_multiple_connections_to_different_nodes(self):
        n = Neat(2, 2, 1)
        genome = random.choice(n.population)
        input_nodes = genome.nodes.get("inputs")
        output_nodes = genome.nodes.get("outputs")
        genome.connections.add(n.create_connection((input_nodes[0], output_nodes[1]), 0.65))
        genome.connections.add(n.create_connection((input_nodes[1], output_nodes[0]), 0.85))
        results = genome.feed_forward([5, 25])
        self.assertTrue(all(result >= -1 and result <= 1 for result in results))

    def test_hidden_nodes(self):
        n = Neat(2, 2, 1)
        genome = random.choice(n.population)
        input_nodes = genome.nodes.get("inputs")
        output_nodes = genome.nodes.get("outputs")
        genome.nodes.get("hidden").append(5)
        genome.connections.add(n.create_connection((input_nodes[0], 5), 0.5))
        genome.connections.add(n.create_connection((input_nodes[1], output_nodes[1]), 0.25))
        genome.connections.add(n.create_connection((5, output_nodes[0]), 0.05))
        results = genome.feed_forward([95, 25])
        self.assertTrue(all(result >= -1 and result <= 1 for result in results))
    
    def test_calculate_distance(self):
        n = Neat(2, 2, 1)
        
        first = Genome(2, 2)
        second = Genome(2, 2)
        n.population = [first, second]
        first.add_connection(n.create_connection((1, 1), 0.75))
        first.add_connection(n.create_connection((1, 2), 0.5))
        second.add_connection(n.create_connection((1, 2), 0.25))
        # 1 + 2 * 
        self.assertEqual(n.calculate_distance(first, second), 1.75)
        
