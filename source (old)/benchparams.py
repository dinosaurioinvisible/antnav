
class Parameters:
    def __init__(self, combos, route_ids, routes_path, cid):
        assert isinstance(combos, list)
        self.combos = combos
        assert isinstance(route_ids, list)
        self.routes_ids = route_ids
        assert isinstance(routes_path, str)
        self.routes_path = routes_path
        self.cid = cid

    def get_params(self):
        return self.combos, self.routes_ids, self.routes_path
