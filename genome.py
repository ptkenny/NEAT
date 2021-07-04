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
        if len(connections) == 0:
            self.inputs = list(range(1, num_inputs + 1))
            self.outputs = list(range(len(self.inputs) + 1, num_outputs + num_inputs + 1))
            self.nodes = {"inputs": self.inputs, "hidden": [], "outputs": self.outputs}
            self.connections = set()
        else:
            all_nodes = list(
                set(con.first for con in connections).union(con.second for con in connections)
            )
            self.nodes = {
                "inputs": [x for x in all_nodes if x <= num_inputs],
                "hidden": [x for x in all_nodes if x > num_inputs and x < num_inputs + num_outputs],
                "outputs": [x for x in all_nodes if x >= num_inputs + num_outputs],
            }
            self.connections = connections
        self.fitness = 0


    def has_connection(self, innovation_number: int):
        return any(con.innovation_id == innovation_number for con in self.connections)


    def get_connection(self, id: int):
        for con in self.connections:
            if con.innovation_id == id:
                return con
        return None


    def feed_forward(self, inputs: list[float]) -> list[float]:
        all_nodes = [y for x in self.nodes.values() for y in x]
        outputs = {x: [] for x in all_nodes}
        for con in sorted(
            list(filter(lambda x: x.enabled, self.connections)), key=lambda c: c.first
        ):
            if con.first in range(len(inputs)):
                outputs[con.second].append(inputs[con.first - 1] * con.weight)
            else:
                result = activations.tanh(sum(outputs[con.first]))
                outputs[con.second].append(result)
        return [activations.tanh(sum(outputs[x])) for x in self.nodes["outputs"]]
    
    def get_node_list(self) -> list[int]:
        return [y for x in self.nodes.values() for y in x]