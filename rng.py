import random

class RNG:

    @staticmethod
    def set_probabilities(probabilities: dict[str, float]):
        # TODO: Define this function properly.
        pass

    AVERAGE_WEIGHT_INHERITANCE_PROBABILITY = 0.4
    DISABLED_CONNECTION_INHERITANCE_PROBABILITY = 0.75
    WEIGHT_CHANGE_PROBABILITY = 0.8
    NORMAL_WEIGHT_CHANGE_PROBABILITY = 0.9

    @property
    def should_inherit_average_weight() -> bool:
        return random.random() < RNG.AVERAGE_WEIGHT_INHERITANCE_PROBABILITY
    

    @property
    def should_disabled_connection_be_inherited() -> bool:
        return random.random() < RNG.DISABLED_CONNECTION_INHERITANCE_PROBABILITY
    

    @property
    def should_weights_change() -> bool:
        return random.random() < RNG.WEIGHT_CHANGE_PROBABILITY
    

    @property
    def should_weights_be_perturbed() -> bool:
        return random.random() < RNG.NORMAL_WEIGHT_CHANGE_PROBABILITY
