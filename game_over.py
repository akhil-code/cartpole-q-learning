class GameOver:
    def __init__(self):
        self.totalCount = {
            'training': 0,
            'testing': 0,
        }
        self.counts = {
            'training': [],
            'testing': [],
        }
    
    def add(self, count, is_training):
        key = 'training' if is_training else 'testing'
        self.totalCount[key] += count
        self.counts[key].append(self.totalCount[key])
    