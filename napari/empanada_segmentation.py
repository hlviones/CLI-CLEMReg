
import os
import sys
import torch
import albumentations as A
import numpy as np
import torch.multiprocessing as mp
from urllib.parse import urlparse
from albumentations.pytorch import ToTensorV2
from empanada.config_loaders import load_config
from empanada.data import VolumeDataset
from empanada.inference import filters
from empanada.inference.engines import PanopticDeepLabRenderEngine3d
from empanada.inference.patterns import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

def load_model_to_device(fpath_or_url, device):
    """ Check whether to use a local version of the mitonet model
    or to download it from a given url

    Parameters
    ----------
    fpath_or_url : str
        String indicating whether the model is stored locally or will be downloaded
    device : str
        Device which the model will be loaded to
    Returns
    -------
        Loaded model
    """
    #

    model = torch.jit.load("/empanda_configs/" + fpath_or_url, map_location=device)


    return model





def _empanada_segmentation(args, volume, start_time):
    """ Function to apply empanada segmentation based on user args input
    and volume input


    Parameters
    ----------
    args : dict
        Dictionary of arguments for the parameters of the empanada segmentation
    volume : napari.layers.Image
        Volume to be segmented using empanada model

    Returns
    -------
        Segmented volume
    """
    # read the model config file
    config = load_config(args.config)

    # set device and determine model to load
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    if sys.platform != 'darwin':
        use_quantized = str(device) == 'cpu' and config.get('model_quantized') is not None
        model_key = 'MitoNet_v1_quantized.pth' if use_quantized else 'MitoNet_v1.pth'
    else:
        model_key = 'model'

    model = load_model_to_device(model_key, device)
    model = model.to(device)

    model.eval()

    zarr_store = None
    # load the volume
    # if '.zarr' in args.volume_path:
    #     zarr_store = zarr.open(args.volume_path, mode='r+')
    #     keys = args.data_key.split(',')
    #     volume = zarr_store[keys[0]]
    #     for key in  keys[1:]:
    #         volume = volume[key]
    # elif '.tif' in args.volume_path:
    #     zarr_store = None
    #     volume = io.imread(args.volume_path)
    # else:
    #     raise Exception(f'Unable to read file {args.volume_path}. Volume must be .tif or .zarr')

    shape = volume.shape
    if args.mode == 'orthoplane':
        axes = {'xy': 0, 'xz': 1, 'yz': 2}
    else:
        axes = {'xy': 0}

    eval_tfs = A.Compose([
        A.Normalize(**config['norms']),
        ToTensorV2()
    ])

    trackers = {}
    class_labels = config['class_names']
    thing_list = config['thing_list']
    label_divisor = args.label_divisor

    # create a separate tracker for
    # each prediction axis and each segmentation class
    trackers = create_axis_trackers(axes, class_labels, label_divisor, shape)

    for axis_name, axis in axes.items():
        print(f'Predicting {axis_name} stack')

        # create placeholder volume for stack results, if desired
        if args.save_panoptic and zarr_store is not None:
            # chunk in axis direction only
            chunks = [None, None, None]
            chunks[axis] = 1
            stack = zarr_store.create_dataset(
                f'panoptic_{axis_name}', shape=shape,
                dtype=np.uint32, chunks=tuple(chunks), overwrite=True
            )
        elif args.save_panoptic:
            stack = np.zeros(shape, dtype=np.uint32)
        else:
            stack = None

        # make axis-specific dataset
        dataset = VolumeDataset(volume, axis, eval_tfs, scale=args.downsample_f)

        num_workers = 8 if zarr_store is not None else 0
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=False,
            drop_last=False, num_workers=0
        )

        # create the inference engine
        inference_engine = PanopticDeepLabRenderEngine3d(
            model, thing_list=thing_list,
            median_kernel_size=args.qlen,
            label_divisor=label_divisor,
            nms_threshold=args.nms_thr,
            nms_kernel=args.nms_kernel,
            confidence_thr=args.seg_thr,
            padding_factor=config['padding_factor'],
            coarse_boundaries=not args.fine_boundaries
        )

        # create a separate matcher for each thing class
        matchers = create_matchers(thing_list, label_divisor, args.iou_thr, args.ioa_thr)

        if sys.platform == "darwin":
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass

        # setup matcher for multiprocessing
        queue = mp.Queue()
        rle_stack = []
        matcher_out, matcher_in = mp.Pipe()
        matcher_args = (
            matchers, queue, rle_stack, matcher_in,
            class_labels, label_divisor, thing_list
        )
        matcher_proc = mp.Process(target=forward_matching, args=matcher_args)
        matcher_proc.start()

        for batch in tqdm(dataloader, total=len(dataloader)):
            image = batch['image']
            size = batch['size']

            # pads and crops image in the engine
            # upsample output by same factor as downsampled input
            pan_seg = inference_engine(image, size, upsampling=args.downsample_f)

            if pan_seg is None:
                queue.put(None)
                continue
            else:
                pan_seg = pan_seg.squeeze().cpu().numpy()
                queue.put(pan_seg)

        final_segs = inference_engine.end(args.downsample_f)
        if final_segs:
            for i, pan_seg in enumerate(final_segs):
                pan_seg = pan_seg.squeeze().cpu().numpy()
                queue.put(pan_seg)

        # finish and close forward matching process
        queue.put('finish')
        rle_stack = matcher_out.recv()[0]
        matcher_proc.join()

        print(f'Propagating labels backward through the stack...')
        for index, rle_seg in tqdm(backward_matching(rle_stack, matchers, shape[axis]), total=shape[axis]):
            update_trackers(rle_seg, index, trackers[axis_name]) #, axis, stack) # Updated empanada only requires 3 arguments 

        finish_tracking(trackers[axis_name])
        for tracker in trackers[axis_name]:
            filters.remove_small_objects(tracker, min_size=args.min_size)
            filters.remove_pancakes(tracker, min_span=args.min_span)

    # create the final instance segmentations
    for class_id, class_name in config['class_names'].items():
        print(f'Creating consensus segmentation for class {class_name}...')
        class_trackers = get_axis_trackers_by_class(trackers, class_id)

        # merge instances from orthoplane inference if applicable
        if args.mode == 'orthoplane':
            if class_id in thing_list:
                consensus_tracker = create_instance_consensus(
                    class_trackers, args.pixel_vote_thr, args.cluster_iou_thr, args.one_view
                )
                filters.remove_small_objects(consensus_tracker, min_size=args.min_size)
                filters.remove_pancakes(consensus_tracker, min_span=args.min_span)
            else:
                consensus_tracker = create_semantic_consensus(class_trackers, args.pixel_vote_thr)
        else:
            consensus_tracker = class_trackers[0]

        dtype = np.uint32 if class_id in thing_list else np.uint8

        # decode and fill the instances
        # if zarr_store is not None:
        # consensus_vol = zarr_store.create_dataset(
        # f'{class_name}_pred', shape=shape, dtype=dtype,
        # overwrite=True, chunks=(1, None, None)
        # )
        # fill_volume(consensus_vol, consensus_tracker.instances, processes=4)
        # else:
        consensus_vol = np.zeros(shape, dtype=dtype)
        fill_volume(consensus_vol, consensus_tracker.instances)

        # volpath = os.path.dirname(args.volume_path)
        # volname = os.path.basename(args.volume_path).replace('.tif', f'_{class_name}.tif')
        # io.imsave(os.path.join(volpath, volname), consensus_vol)

    print(f'Finished EM segmentation after {time.time() - start_time}s!')
    print(consensus_vol)
    
    return consensus_vol


def empanada_segmentation(input, axis_prediction, args):
    """ Initialise empanada segmentation

    Parameters
    ----------
    input : napari.layers.Image
        Image volume to apply segmentation to
    axis_prediction : bool
        Determine whether to run prediction across orthogonal planes or not.

    Returns
    -------
        Segmentation of the user inputted image
    """
    start_time = time.time()

    config = os.path.abspath(
        os.path.join(os.path.realpath(__file__), '..', '..', 'empanada_configs', 'MitoNet_V1.yaml'))
    args.config = config

    if axis_prediction:
        args.mode = 'orthoplane'
    else:
        args.mode = 'stack'
    return _empanada_segmentation(args, input, start_time)