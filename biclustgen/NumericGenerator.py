from Generator import Generator


class NumericGenerator(Generator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.realval = bool(kwargs.get('realval', True))
        self.minval = int(kwargs.get('minval', -10))
        self.maxval = int(kwargs.get('maxval', 10))