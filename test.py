from utils.decorators import timed
from utils.augment import Cutout
from utils.augment import apply_augment
from utils.score import score2019



@timed
def test():
    print('test')


test()