"""Training infos."""

class TrainingInfo():

    def __init__(self):
        self.iteration = 0
    def set_iteration(self, iteration):
        self.iteration = iteration
    def get_iteration(self):
        return self.iteration

