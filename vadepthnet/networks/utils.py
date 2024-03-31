import warnings
import os
import os.path as osp
import pkgutil
import warnings
from collections import OrderedDict
from importlib import import_module

import torch
import torchvision
import torch.nn as nn
from torch.utils import model_zoo
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch import distributed as dist

TORCH_VERSION = torch.__version__


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def is_module_wrapper(module):
    module_wrappers = (DataParallel, DistributedDataParallel)
    return isinstance(module, module_wrappers)


def get_dist_info():
    if TORCH_VERSION < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    # 先注释掉了
    # if unexpected_keys:
    #     err_msg.append('unexpected key in source '
    #                    f'state_dict: {", ".join(unexpected_keys)}\n')
    # if missing_keys:
    #     err_msg.append(
    #         f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_url_dist(url, model_dir=None):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`. Default: None.

    Returns:
        dict | OrderedDict: The loaded checkpoint. It can be either an
            OrderedDict storing model weights or a dict containing other
            information, which depends on the checkpoint.
    """
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please '
                      'use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    # reshape absolute position embedding
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        N1, L, C1 = absolute_pos_embed.size()
        N2, C2, H, W = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H*W:
            logger.warning("Error in loading absolute_pos_embed, pass")
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)

    # interpolate position bias table if needed
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_key_no_backbone = table_key.replace('backbone.','')
        table_current = model.state_dict()[table_key_no_backbone]
        # # table_current_key = model.state_dict().get(table_key, None)
        # # table_current = model.state_dict()[table_pretrained]
        # # table_current = model.state_dict()[table_current_key]
        # table_pretrained = state_dict[table_key]
        # table_current = model.state_dict()[table_key]
        L1, nH1 = table_pretrained.size()
        L2, nH2 = table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {table_key}, pass")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                table_pretrained_resized = F.interpolate(
                     table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                     size=(S2, S2), mode='bicubic')
                state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def compute_sparse_map(depth_map_tensor, block_sizew=16, block_sizeh=16):
    batch_size, channels, height, width = depth_map_tensor.shape
    depth_map_resized = torch.zeros((batch_size, channels, height // block_sizeh, width // block_sizew), device=depth_map_tensor.device)
    # print(depth_map_tensor.shape)
    for i in range(0, height, block_sizeh):
        for j in range(0, width, block_sizew):
            block = depth_map_tensor[:, :, i:i+block_sizeh, j:j+block_sizew]
            valid_values = block[block != 0]

            valid_size = valid_values.size()[0] 
            if valid_size > 0:
                avg_depth = torch.mean(valid_values.float()) 
                depth_map_resized[:, :, i // block_sizeh, j // block_sizew] = avg_depth
   
    return depth_map_resized


def compute_sparse_gradient(depth):
    num, _, h, w = depth.shape

    gradient = torch.zeros(num, 4, h, w, device=depth.device)

    for i in range(num):
        for j in range(h):
            for k in range(w):
                if (k+1) % w != 0 and (k+1) < w:
                    gradient[i, 0, j, k] = depth[i, 0, j, k] - depth[i, 0, j, k+1]
                
                if j + 1 < h:
                    gradient[i, 1, j, k] = depth[i, 0, j, k] - depth[i, 0, j+1, k]
                
                if (k+2) % w != 0 and (k+2) < w:
                    gradient[i, 2, j, k] = depth[i, 0, j, k] - depth[i, 0, j, k+2]
                
                if j + 2 < h:
                    gradient[i, 3, j, k] = depth[i, 0, j, k] - depth[i, 0, j+2, k]

    return gradient


def fuse_gradient(actual_gradients, predicted_gradients):
    channel_num =  actual_gradients.shape[0] * 16 # 2為batch_size/GPU數量
    # print(channel_num)
    predicted_gradient = predicted_gradients.view(channel_num, 22, 76, 4)
    actual_gradient = actual_gradients.permute(2, 3, 1, 0)
    predicted_gradient = predicted_gradient.permute(1, 2, 3, 0)
    
    blended_gradient = torch.zeros(22, 76, 4, channel_num, device=predicted_gradient.device)
    for bsize in range(actual_gradient.shape[3]):
        pre_grad = predicted_gradient[:, :, :, bsize*16 :(bsize+1)*16]
        act_grad = actual_gradient[:, :, :, bsize]
        horizontal_channels = [0, 2]
        vertical_channels = [1, 3]

        # 初始化缩放因子张量
        scaling_factors = torch.zeros_like(pre_grad)

        # 逐个像素点计算缩放因子
        for channel in horizontal_channels:
            actual_values = act_grad[:, :, channel]
            predicted_values = pre_grad[:, :, channel, :]
            for cnum in range(16):
                actual_value = actual_values
                predicted_value = predicted_values[:, :, cnum]

                mask = actual_value != 0
                scaling_factors[:, :, channel, cnum][mask] = predicted_value[mask] / actual_value[mask]

        for channel in vertical_channels:
            actual_values = act_grad[:, :, channel]
            predicted_values = pre_grad[:, :, channel, :]
            for cnum in range(16):
                actual_value = actual_values
                predicted_value = predicted_values[:, :, cnum]

                mask = actual_value != 0
                scaling_factors[:, :, channel, cnum][mask] = predicted_value[mask] / actual_value[mask]

        # 计算每个像素点的平均缩放因子
        mean_scaling_factors = torch.mean(scaling_factors, dim=(0, 1))
        
        # 使用平均缩放因子对梯度进行缩放 ([22, 76, 16])
        scaled_gradient = act_grad.unsqueeze(-1).expand(22, 76, 4, 16).clone()
        for i in range(4):
            mask = act_grad[:, :, i] != 0
            scaled_gradient[:, :, i, :][mask] *= mean_scaling_factors[i,:]
            
        # 设置权重用于融合
        weight_actual = 0.5  
        weight_predicted = 0.5  

        filled_gradient = scaled_gradient.clone()
        filled_gradient[act_grad == 0] = pre_grad[act_grad == 0]

        # 计算融合后的gradient
        blended_gradient[:,:,:,bsize*16 :(bsize+1)*16] = (weight_actual * filled_gradient + weight_predicted * pre_grad)

    blended_gradient = blended_gradient.view(1672, 4, channel_num, 1)
    blended_gradient = blended_gradient.permute(2, 0, 1, 3)
#   應返回(16, 1672, 4, 1)
    return blended_gradient
