class Connection:
    def __init__(
        self, first: int, second: int, weight: float, enabled: bool, innovation_id: int
    ) -> None:
        self.first = first
        self.second = second
        self.weight = weight
        self.enabled = enabled
        # TODO: It might be a smarter idea to increment the global innovation number here.
        self.innovation_id = innovation_id

    def copy(self) -> "Connection":
        return Connection(self.first, self.second, self.weight, self.enabled, self.innovation_id)

    def __str__(self) -> str:
        return "{} --> {}, weight: {}, enabled: {}, innovation_id: {}".format(self.first, self.second, self.weight, self.enabled, self.innovation_id)

    def __hash__(self) -> int:
        return hash(self.innovation_id)

    def __eq__(self, other):
        return self.innovation_id == other.innovation_id