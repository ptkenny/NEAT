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
        con = Connection()
        con.first = self.first
        con.second = self.second
        con.weight = self.weight
        con.enabled = self.enabled
        con.innovation_id = self.innovation_id
        return con

    def __str__(self) -> str:
        return "{} --> {}, weight: {}".format(self.first, self.second, self.weight)

    def __hash__(self) -> int:
        return hash(self.innovation_id)
