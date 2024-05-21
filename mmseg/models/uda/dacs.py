# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.masking_consistency_module import MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import denorm, get_class_masks, get_mean_std, strong_transform
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg["max_iters"]
        self.source_only = cfg["source_only"]
        self.alpha = cfg["alpha"]
        self.pseudo_threshold = cfg["pseudo_threshold"]
        self.psweight_ignore_top = cfg["pseudo_weight_ignore_top"]
        self.psweight_ignore_bottom = cfg["pseudo_weight_ignore_bottom"]
        self.mix = cfg["mix"]
        self.blur = cfg["blur"]
        self.color_jitter_s = cfg["color_jitter_strength"]
        self.color_jitter_p = cfg["color_jitter_probability"]
        self.mask_mode = cfg["mask_mode"]
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg["print_grad_magnitude"]
        assert self.mix == "class"

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg["model"])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(), self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop("loss", None)  # remove the unnecessary 'loss'
        outputs = dict(log_vars=log_vars, num_samples=len(data_batch["img_metas"]))
        return outputs

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, : self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom :, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def forward_train(
        self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, rare_class=None, valid_pseudo_mask=None
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            "mix": None,
            "color_jitter": random.uniform(0, 1),
            "color_jitter_s": self.color_jitter_s,
            "color_jitter_p": self.color_jitter_p,
            "blur": random.uniform(0, 1) if self.blur else 0,
            "mean": means[0].unsqueeze(0),  # assume same normalization
            "std": stds[0].unsqueeze(0),
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=True)
        seg_debug["Source"] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=False)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [p.grad.detach().clone() for p in params if p.grad is not None]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f"Seg. Grad.: {grad_mag}", "mmseg")

        pseudo_label, pseudo_weight = None, None
        if not self.source_only:
            # Generate pseudo-label
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.get_ema_model().generate_pseudo_label(target_img, target_img_metas)
            seg_debug["Target"] = self.get_ema_model().debug_output

            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(ema_logits)
            del ema_logits

            pseudo_weight = self.filter_valid_pseudo_region(pseudo_weight, valid_pseudo_mask)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mixed_seg_weight = pseudo_weight.clone()
            mix_masks = get_class_masks(gt_semantic_seg)

            for i in range(batch_size):
                strong_parameters["mix"] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])),
                )
                _, mixed_seg_weight[i] = strong_transform(
                    strong_parameters, target=torch.stack((gt_pixel_weight[i], pseudo_weight[i]))
                )
            del gt_pixel_weight
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)

            # Train on mixed images
            mix_losses = self.get_model().forward_train(
                mixed_img,
                img_metas,
                mixed_lbl,
                seg_weight=mixed_seg_weight,
                return_feat=False,
            )
            seg_debug["Mix"] = self.get_model().debug_output
            mix_losses = add_prefix(mix_losses, "mix")
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()

        # Masked Training
        if self.enable_masking and self.mask_mode.startswith("separate"):
            masked_loss = self.mic(
                self.get_model(),
                img,
                img_metas,
                gt_semantic_seg,
                target_img,
                target_img_metas,
                valid_pseudo_mask,
                pseudo_label,
                pseudo_weight,
            )
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, "masked")
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()

        if self.local_iter % self.debug_img_interval == 0 and not self.source_only:
            out_dir = os.path.join(self.train_cfg["work_dir"], "debug")
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={"hspace": 0.1, "wspace": 0, "top": 0.95, "bottom": 0, "right": 1, "left": 0},
                )
                subplotimg(axs[0][0], vis_img[j], "Source Image")
                subplotimg(axs[1][0], vis_trg_img[j], "Target Image")
                subplotimg(axs[0][1], gt_semantic_seg[j], "Source Seg GT", cmap="cityscapes")
                subplotimg(axs[1][1], pseudo_label[j], "Target Seg (Pseudo) GT", cmap="cityscapes")
                subplotimg(axs[0][2], vis_mixed_img[j], "Mixed Image")
                subplotimg(axs[1][2], mix_masks[j][0], "Domain Mask", cmap="gray")
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                if mixed_lbl is not None:
                    subplotimg(axs[1][3], mixed_lbl[j], "Seg Targ", cmap="cityscapes")
                subplotimg(axs[0][3], mixed_seg_weight[j], "Pseudo W.", vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(axs[0][4], self.debug_fdist_mask[j][0], "FDist Mask", cmap="gray")
                if self.debug_gt_rescale is not None:
                    subplotimg(axs[1][4], self.debug_gt_rescale[j], "Scaled GT", cmap="cityscapes")
                for ax in axs.flat:
                    ax.axis("off")
                plt.savefig(os.path.join(out_dir, f"{(self.local_iter + 1):06d}_{j}.png"))
                plt.close()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg["work_dir"], "debug")
            os.makedirs(out_dir, exist_ok=True)
            if seg_debug["Source"] is not None and seg_debug:
                if "Target" in seg_debug:
                    seg_debug["Target"]["Pseudo W."] = mixed_seg_weight.cpu().numpy()
                for j in range(batch_size):
                    cols = len(seg_debug)
                    rows = max(len(seg_debug[k]) for k in seg_debug.keys())
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(5 * cols, 5 * rows),
                        gridspec_kw={"hspace": 0.1, "wspace": 0, "top": 0.95, "bottom": 0, "right": 1, "left": 0},
                        squeeze=False,
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            subplotimg(axs[k2][k1], **prepare_debug_out(f"{n1} {n2}", out[j], means, stds))
                    for ax in axs.flat:
                        ax.axis("off")
                    plt.savefig(os.path.join(out_dir, f"{(self.local_iter + 1):06d}_{j}_s.png"))
                    plt.close()
                del seg_debug
        self.local_iter += 1

        return log_vars
