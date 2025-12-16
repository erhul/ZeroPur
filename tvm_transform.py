from transformations.tvm import reconstruct as tvm

class Defense(object):

    def __init__(self, defense, defense_name):
        assert callable(defense)
        self.defense = defense
        self.defense_name = defense_name

    def __call__(self, im):
        return self.defense(im)

    def get_name(self):
        return self.defense_name
    
def defense(im):
    im = tvm(
        im,
        .0,
        'bregman',
        0.06 #0.11
    )
    return im
