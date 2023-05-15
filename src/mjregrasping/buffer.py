class Buffer:

    def __init__(self, n: int):
        self.n = n
        self.data = []

    def insert(self, x):
        self.data.append(x)
        if len(self.data) > self.n:
            self.data.pop(0)

    def full(self):
        return len(self.data) == self.n

    def get(self, i):
        return self.data[i]

    def most_recent(self):
        return self.data[-1]

    def reset(self):
        self.data = []

    def __len__(self):
        return len(self.data)