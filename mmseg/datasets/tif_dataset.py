import os
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class TifDataset(CustomDataset):
    CLASSES = ('background', 'road', 'tree', 'building',)
    PALETTE = ([0,0,0], [128, 64, 128], [0, 128, 0], [70, 70, 70]) 

    def load_annotations(self, 
                         img_dir = 'data/maxar-open-data', 
                         img_suffix = '.tif', 
                         ann_dir = 'data/outputs/', 
                         seg_map_suffix = '.tif',
                         split = 'train',):
        img_infos = []
        folders = [f for f in os.listdir(ann_dir) if os.path.isdir(os.path.join(ann_dir, f))]
        for folder in folders:
            ann_folder = os.path.join(ann_dir, folder)
            img_folder = os.path.join(img_dir, folder)
            # append 'pre/' to both
            ann_folder = os.path.join(ann_folder, 'pre/')
            img_folder = os.path.join(img_folder, 'pre/')
            for mosaic in os.listdir(ann_folder):
                ann_mosaic_folder = os.path.join(ann_folder, mosaic)
                img_mosaic_folder = os.path.join(img_folder, mosaic)
                for tile in os.listdir(ann_mosaic_folder):
                    ann_tile = os.path.join(ann_mosaic_folder, tile)
                    img_tile = os.path.join(img_mosaic_folder, tile)
                    # remove img_dir and ann_dir
                    ann_tile = ann_tile.replace(ann_dir, '')
                    img_tile = img_tile.replace(img_dir, '')
                    if ann_tile.endswith(seg_map_suffix):
                        img_info = dict(
                            filename=img_tile,
                        )
                        img_info['ann'] = dict(seg_map=ann_tile)
                        img_infos.append(img_info)
        return img_infos