from torch.utils.data import Sampler
import random
import numpy as np
import json

class UDAEntropyIJSampler(Sampler):
    def __init__(self, num_event_imgs, all_train_paths, seed = None):
        self.num_event_imgs = num_event_imgs
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
        #should have the 3 tensor of all entropies
        self.entropies = np.load('./metadata/glbl_high_res_entropies.npy')
        with open('./metadata/index_map.json', 'r') as file:
            self.index_map = json.load(file)
        keep_ix = [self.index_map[img_path] for img_path in all_train_paths]
        self.entropies = self.entropies[keep_ix]
        if self.num_event_imgs[-1] != self.entropies.shape[0]:
            raise ValueError("The number of event images should match the number of entropies")
        
    def __iter__(self):
        while True:
            res = []
            for _ in range(2):
                rnd_ix = random.randrange(len(self.num_event_imgs))
                if rnd_ix == 0:
                    start = 0
                else:
                    start = self.num_event_imgs[rnd_ix - 1]
                end = self.num_event_imgs[rnd_ix]
                ev_entropies = self.entropies[start:end]
                ev_patch_prob = ev_entropies / ev_entropies.sum()
                flat_probabilities = ev_patch_prob.flatten()
                # Sample a flat index based on the probabilities
                sampled_flat_index = np.random.choice(a=len(flat_probabilities), p=flat_probabilities)
                # Convert the flat index back to 3D index
                sampled_index_3d = np.unravel_index(sampled_flat_index, ev_patch_prob.shape)
                #print((start + sampled_index_3d[0], sampled_index_3d[1], sampled_index_3d[2]))
                #print('Entropy at position ^')
                #print(self.entropies[start + sampled_index_3d[0], sampled_index_3d[1], sampled_index_3d[2]])
                #print()
                res.append((start + sampled_index_3d[0], sampled_index_3d[1], sampled_index_3d[2]))
            yield res
    
    def __len__(self):
        return 40_000