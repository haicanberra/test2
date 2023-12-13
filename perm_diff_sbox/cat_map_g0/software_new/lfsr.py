class LFSR:
    def __init__(self, seed, taps):
        self.state = seed
        self.taps = taps

    def next(self):
        feedback = 0
        for tap in self.taps:
            feedback ^= int(self.state[tap-1])
        self.state = str(feedback) + self.state[:-1]

        return feedback

    def output_bit(self):
        return self.state[-1]

    def get_state(self):
        return self.state
    
    def get_recovered_bit(self):
        return self.state[0]