import os
from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp
import rasterio as rio
from rasterio.features import rasterize
from pathlib import Path
import glob
import json
import shapely


@DATASETS.register_module()
class TifDataset(CustomDataset):
    CLASSES = ('background', 'road', 'tree', 'building',)
    PALETTE = ([0, 0, 0], [245, 248, 80], [173, 51, 24], [78, 106, 220])
    
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
        folders = sorted(folders)
        for folder in folders:
            ann_folder = os.path.join(ann_dir, folder)
            img_folder = os.path.join(img_dir, folder)
            # append 'pre/' to both
            ann_folder = os.path.join(ann_folder, 'pre/')
            img_folder = os.path.join(img_folder, 'pre/')
            # for mosaic in os.listdir(ann_folder):
            for mosaic in sorted(os.listdir(ann_folder)):
                ann_mosaic_folder = os.path.join(ann_folder, mosaic)
                img_mosaic_folder = os.path.join(img_folder, mosaic)
                # for tile in os.listdir(ann_mosaic_folder):
                for tile in sorted(os.listdir(ann_mosaic_folder)):
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

    def get_gt_seg_maps(self, efficient_test=False, img_order=None):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        if img_order is not None:
            ori_filenames = img_order
        else:
            ori_filenames = [img_info['ann']['seg_map'] for img_info in self.img_infos]
        for ori_filename in ori_filenames:
            # print(ori_filename)
            seg_map = osp.join(self.ann_dir, ori_filename)
            if efficient_test:
                raise NotImplementedError
                # gt_seg_map = seg_map
            else:
                with rio.open(seg_map) as src:
                    gt_seg_map = src.read()                
            aoi_mask = path_to_aoi_mask(seg_map)
            gt_seg_map = gt_seg_map[0]
            gt_seg_map = gt_seg_map + 1
            gt_seg_map[aoi_mask == 0] = 255
            gt_seg_map[gt_seg_map == 256] = 255  # 255 is the class for 'out of bounds', set as ignore_index 
            gt_seg_maps.append(gt_seg_map)
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
class MaxarDatasetVal(TifDataset):
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
        files = [f for f in os.listdir(
            ann_dir) if os.path.isfile(os.path.join(ann_dir, f))]
        for tile in files:
            ann_tile = os.path.join(ann_dir, tile)
            img_tile = os.path.join(img_dir, tile)
            # remove img_dir and ann_dir
            ann_tile = ann_tile.replace(ann_dir, '')
            img_tile = img_tile.replace(img_dir, '')
            # replace 'gt' with 'img'
            img_tile = img_tile.replace('gt', 'img')
            if ann_tile.endswith(seg_map_suffix):
                img_info = dict(
                    filename=img_tile,
                )
                img_info['ann'] = dict(seg_map=ann_tile)
                img_infos.append(img_info)
        return img_infos
    
    def get_gt_seg_maps(self, efficient_test=False):
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                with rio.open(seg_map) as src:
                    gt_seg_map = src.read()
            aoi_mask = path_to_aoi_mask(seg_map)
            gt_seg_map = gt_seg_map[0]
            gt_seg_map = gt_seg_map + 1
            gt_seg_map[gt_seg_map == 256] = 0
            # set gt_seg_map to 255 where aoi_mask is 0
            gt_seg_map[aoi_mask == 0] = 255
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    

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


def path_2_tile_aoi(tile_path, root = './metadata/from_github_maxar_metadata/datasets' ):
    """
    Create a shapely Polygon from a tile_path
    Example of a tile_path: '../Gambia-flooding-8-11-2022/pre/10300100CFC9A500/033133031213.tif'
    """
    if isinstance(tile_path, str):
        event = tile_path.split('/')[-4]
        child = tile_path.split('/')[-2]
        tile = tile_path.split('/')[-1].replace(".tif", "")
    elif isinstance(tile_path, Path):
        event = tile_path.parts[-4]
        child = tile_path.parts[-2]
        tile = tile_path.parts[-1].replace(".tif", "")
    else:
        raise TypeError("tile_path must be a string or a Path object")
    
    try:
        path_2_child_geojson = os.path.join(root, event, child +'.geojson')
        with open(path_2_child_geojson, 'r') as f:
            child_geojson = json.load(f)
    except:
        file_pattern = str(os.path.join(root, event, child + '*inv.geojson'))
        file_list = glob.glob(f"{file_pattern}")
        assert len(file_list) == 1, f"Found {len(file_list)} files with pattern {file_pattern}. Expected 1 file."
        path_2_child_geojson = file_list[0]
        with open(path_2_child_geojson, 'r') as f:
            child_geojson = json.load(f)
    
    
    j = [el["properties"]["proj:geometry"] for el in child_geojson['features'] if el['properties']['quadkey'] == tile][0]
    tile_polyg = shapely.geometry.shape(j)
    return tile_polyg

def path_to_aoi_mask(tile_path):
    with rio.open(tile_path) as src:
        transform = src.transform
        tile_shape = (src.height, src.width)
        
    aoi_mask = rasterize([path_2_tile_aoi(tile_path)], out_shape = tile_shape, fill=False, default_value=True, transform = transform)
    return aoi_mask

import numpy as np
    
@DATASETS.register_module()
class MaxarNoTrees(MaxarDsEntropy):
    CLASSES = ('background', 'road', 'building',)
    PALETTE = ([0, 0, 0], [245, 248, 80], [78, 106, 220])
    # same as MaxarDataset but trees are set as background
    def get_gt_seg_maps(self, efficient_test=False, img_order=None):
        # get gt_seg_maps from super
        gt_seg_maps = super().get_gt_seg_maps(efficient_test, img_order)
        # set trees as background
        for gt_seg_map in gt_seg_maps:
            gt_seg_map[gt_seg_map == 2] = 0
            # shift class 3 to 2
            gt_seg_map[gt_seg_map == 3] = 2
        return gt_seg_maps
    
    
@DATASETS.register_module()
class MaxarNoTreesVal(MaxarDataset):
    CLASSES = ('background', 'road', 'building',)
    PALETTE = ([0, 0, 0], [245, 248, 80], [78, 106, 220])
    # same as MaxarDataset but trees are set as background
    def get_gt_seg_maps(self, efficient_test=False, img_order=None):
        # get gt_seg_maps from super
        gt_seg_maps = super().get_gt_seg_maps(efficient_test, img_order)
        # set trees as background
        for gt_seg_map in gt_seg_maps:
            gt_seg_map[gt_seg_map == 2] = 0
            # shift class 3 to 2
            gt_seg_map[gt_seg_map == 3] = 2
        return gt_seg_maps
    