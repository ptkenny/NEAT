import activations
from connection import Connection


class Genome:
    def __init__(
        self, num_inputs: int, num_outputs: int, connections: list[Connection] = set()
    ) -> None:
        """
        The class representing a "genome" in NEAT terms. It contains connections,
        nodes, and other important information within.

        num_inputs: The number of inputs for this genome.
        num_outputs: The number of outputs for this genome.
        connections(optional): The set of connections for this genome.
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        if len(connections) == 0:
            self.nodes = {"inputs": set(range(1, num_inputs + 1)), "hidden": set(), "outputs": set(range(num_inputs + 1, num_outputs + num_inputs + 1))}
            self.connections = set()
            self.connection_map = dict()
        else:
            all_nodes = set(con.first for con in connections).union(con.second for con in connections)
            self.nodes = {
                "inputs": set(range(1, num_inputs + 1)), 
                "outputs": set(range(num_inputs + 1, num_outputs + num_inputs + 1)),
            }
            self.nodes.update({
                "hidden": all_nodes - self.nodes["inputs"] - self.nodes["outputs"]
            })
            self.connections = connections
            self.create_connection_map()
        self.fitness = 0


    def has_connection(self, connection: Connection):
        return connection in self.connections


    def add_connection(self, connection: Connection) -> None:
        self.connections.add(connection)
        all_nodes = set(con.first for con in self.connections).union(con.second for con in self.connections)
        self.nodes = {
            "inputs": set(range(1, self.num_inputs + 1)), 
            "outputs": set(range(self.num_inputs + 1, self.num_outputs + self.num_inputs + 1)),
        }
        self.nodes.update({
            "hidden": all_nodes - self.nodes["inputs"] - self.nodes["outputs"]
        })
        self.create_connection_map()


    def create_connection_map(self):
        self.connection_map = {x: [] for x in self.get_node_list()}
        for connection in self.connections:
            if connection.enabled:
                self.connection_map[connection.second].append((connection.first, connection.weight))

    def get_connection_weight(self, connection: Connection):
        if connection in self.connections:
            for con in self.connections:
                if connection == con:
                    return con.weight
        return None
    
    def get_connection_enabled(self, connection: Connection):
        return any(con.enabled and con == connection for con in self.connections)

    def get_connection(self, connection: Connection):
        for con in self.connections:
            if con == connection:
                return con
        return None


    def feed_forward(self, inputs: list[float]) -> list[float]:
        
        def get_result(node):
            if node in self.nodes["inputs"]:
                return inputs[node - 1]
            else:
                if node not in self.connection_map:
                    return 0
                else:
                    return activations.neat_sigmoid(sum(get_result(in_node) * weight for in_node, weight in self.connection_map[node]))

        results = []

        for output in self.nodes["outputs"]:
            results.append(get_result(output))
        
        return results
    
    def get_node_list(self) -> list[int]:
        return [y for x in self.nodes.values() for y in x]