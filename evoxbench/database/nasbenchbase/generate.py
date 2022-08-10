class Generate:
    def __init__(self, search_space):
        self.search_space = search_space

    def next(self):
        raise NotImplementedError
