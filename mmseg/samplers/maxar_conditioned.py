from torch.utils.data import Sampler
import random

class MaxarConditionedSampler(Sampler):
    def __init__(self, num_event_imgs, seed = None):
        self.num_event_imgs = num_event_imgs
        
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
    
    def __iter__(self):
        while True:
            rnd_ix = random.randrange(len(self.num_event_imgs))
            if rnd_ix == 0:
                start = 0
            else:
                start = self.num_event_imgs[rnd_ix - 1]
            end = self.num_event_imgs[rnd_ix]
            i = random.randint(start, end - 1)
            # print(i)
            yield i
    
    def __len__(self):
        return 40_000