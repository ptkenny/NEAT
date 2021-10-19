import random


class RNG:

    AVERAGE_WEIGHT_INHERITANCE_PROBABILITY = 0.4
    DISABLED_CONNECTION_INHERITANCE_PROBABILITY = 0.75
    WEIGHT_CHANGE_PROBABILITY = 0.8
    NORMAL_WEIGHT_CHANGE_PROBABILITY = 0.9
    NEW_CONNECTION_PROBABILITY = 0.1
    NEW_NODE_PROBABILITY = 0.01
    SHOULD_MUTATE_WITHOUT_CROSSOVER = 0.25

    @staticmethod
    def set_probabilities(probabilities: dict[str, float]):
        # TODO: Define this function properly.
        pass


    def should_mutate_without_crossover() -> bool:
        return random.random() < RNG.should_mutate_without_crossover


    def should_inherit_average_weight() -> bool:
        return random.random() < RNG.AVERAGE_WEIGHT_INHERITANCE_PROBABILITY
    


    def should_disabled_connection_be_inherited() -> bool:
        return random.random() < RNG.DISABLED_CONNECTION_INHERITANCE_PROBABILITY
    


    def should_weights_change() -> bool:
        return random.random() < RNG.WEIGHT_CHANGE_PROBABILITY
    


    def should_weights_be_perturbed() -> bool:
        return random.random() < RNG.NORMAL_WEIGHT_CHANGE_PROBABILITY


    def should_connection_be_added() -> bool:
        return random.random() < RNG.NEW_CONNECTION_PROBABILITY


    def should_node_be_added() -> bool:
        return random.random() < RNG.NEW_CONNECTION_PROBABILITY