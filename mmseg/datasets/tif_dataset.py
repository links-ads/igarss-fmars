import os
from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp
import rasterio as rio


@DATASETS.register_module()
class TifDataset(CustomDataset):
    CLASSES = ('background', 'road', 'tree', 'building',)
    PALETTE = ([0, 0, 0], [128, 64, 128], [0, 128, 0], [70, 70, 70])

    def load_annotations(self,
                         img_dir='data/maxar-open-data',
                         img_suffix='.tif',
                         ann_dir='data/outputs/',
                         seg_map_suffix='.tif',
                         split='train',
                         rare_class_sampling=None,
                         ):
        self.rare_class_sampling = rare_class_sampling
        img_infos = []
        folders = [f for f in os.listdir(
            ann_dir) if os.path.isdir(os.path.join(ann_dir, f))]
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

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                with rio.open(seg_map) as src:
                    gt_seg_map = src.read()
            gt_seg_maps.append(gt_seg_map[0])
        return gt_seg_maps

@DATASETS.register_module()
class MaxarDataset(TifDataset):
    
    def load_annotations(self,
                         img_dir='data/maxar-open-data',
                         img_suffix='.tif',
                         ann_dir='data/outputs/',
                         seg_map_suffix='.tif',
                         split='train',
                         rare_class_sampling=None,
                         ):
        self.rare_class_sampling = rare_class_sampling
        img_infos = []
        glbl_img_count = 0
        folders = [f for f in os.listdir(
            ann_dir) if os.path.isdir(os.path.join(ann_dir, f))]
        folders = sorted(folders)
        self.num_event_imgs = [] #las
        for folder in folders: # iterate over each event
            ann_folder = os.path.join(ann_dir, folder)
            img_folder = os.path.join(img_dir, folder)
            # append 'pre/' to both
            ann_folder = os.path.join(ann_folder, 'pre/')
            img_folder = os.path.join(img_folder, 'pre/')
            
            for mosaic in sorted(os.listdir(ann_folder)): # iterate over each mosaic
                ann_mosaic_folder = os.path.join(ann_folder, mosaic)
                img_mosaic_folder = os.path.join(img_folder, mosaic)
                tif_files = [tif_file for tif_file in os.listdir(ann_mosaic_folder) if tif_file.endswith('.tif')]
                for tile in sorted(tif_files):
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
                    glbl_img_count += 1
            if len(os.listdir(ann_mosaic_folder)) != 0:
                self.num_event_imgs.append(glbl_img_count)
        self.all_train_paths = [img_info['filename'] for img_info in img_infos]
        return img_infos

@DATASETS.register_module()
class MaxarDsEntropy(MaxarDataset):
    def __getitem__(self, idx_3d: tuple):
        """Get training/test data after pipeline.

        Args:
            idx_3d (tuple): (img_idx, i, j)

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        if self.test_mode:
            return self.prepare_test_img(idx_3d)
        else:
            return self.prepare_train_img(idx_3d)
    
    def prepare_train_img(self, idx_3d):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        
        idx, i, j = idx_3d
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info, local_idx = (i, j))
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def prepare_test_img(self, idx_3d):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        idx, i, j = idx_3d
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info, local_idx = (i, j))
        self.pre_pipeline(results)
        return self.pipeline(results)