
from log_segmentation import log_segmentation
from empanada_segmentation import empanada_segmentation
from point_cloud_sampling import point_cloud_sampling
from napari.layers.utils._link_layers import link_layers
from napari.layers.utils.stack_utils import images_to_stack, stack_to_images
from point_cloud_registration import point_cloud_registration
import matplotlib.pyplot as plt
from warp_image_volume import warp_image_volume
import argparse
from tifffile import imwrite
# Create the parser
from os import mkdir, path
import numpy as np
import napari
import skimage
from napari.layers import Image, Shapes, Labels, Points




def main():
    if not path.exists(args.output_folder):
        mkdir(args.output_folder)
    viewer = napari.Viewer(show=False)
    print("Loading EM Data")
    em = skimage.io.imread(path.join(args.input_folder, args.em_input))   
    
    viewer.add_image(em, name=args.em_input.replace(".tif", ""))

    print("Loading LM Data")
    lm = skimage.io.imread(path.join(args.input_folder, args.lm_input))  


    

    # Assuming 'image' is your 3D numpy array
    slices = np.split(lm.data, lm.data.shape[0])


    # Add each 2D slice to the viewer

    if len(em.data.shape) != 3:
        return f"ERROR: EM Data is not 3D, shape is {len(em.data.shape)}"
    if len(lm.data.shape) <= 2:
        return f"ERROR: LM Data is not 3D, shape is {len(lm.data.shape)}"
    elif len(lm.data.shape) == 4:
        if args.mito_channel:
            slices = np.split(lm.data, lm.data.shape[0])
            linked_layers = []
            for i, slice in enumerate(slices):
                slice = np.squeeze(slice, axis=0)
                layer = viewer.add_image(slice, name=f'{args.lm_input.replace(".tif", "")}_{i}')
                linked_layers.append(layer)
            lm = viewer.layers[f'{args.lm_input.replace(".tif", "")}_{args.mito_channel}']
            link_layers(linked_layers)   
        else:
            return f"ERROR: Mito Channel has not been set and LM image has {lm.data.shape[0]}"
    elif len(lm.data.shape) == 3:
        viewer.add_image(lm, name=f'{args.lm_input.replace(".tif", "")}')
        lm = viewer.layers[args.lm_input.replace(".tif", "")]
    else:
        return f"ERROR: LM Data is not 3D, shape is {len(lm.data.shape)}"


    
    lm_seg = log_segmentation(lm, args.log_sigma, args.threshold)

    if args.save_intermediary == True:
        imwrite(path.join(args.output_folder, "Moving_Segmentation", ".tif"), lm_seg.data.astype(np.int64))
    viewer.add_labels(lm_seg.data.astype(np.int64),
                        name="Moving_Segmentation")
    


    em = viewer.layers[args.em_input.replace(".tif", "")]
    em_seg = empanada_segmentation(em.data, False, args)
    if args.save_intermediary == True:
        imwrite(path.join(args.output_folder, "Fixed_Segmentation", ".tif"), em_seg.astype(np.int64))

    viewer.add_labels(em_seg.astype(np.int64),
                        name="Fixed_Segmentation")
    
    mp = point_cloud_sampling(viewer.layers["Moving_Segmentation"], args.sampling_freq, args.sampling_sigma)
    fp = point_cloud_sampling(viewer.layers["Fixed_Segmentation"], args.sampling_freq, args.sampling_sigma)
    if args.save_intermediary == True:
        imwrite(path.join(args.output_folder, "Moving_Points", ".tif"), mp.data)
    viewer.add_points(mp.data,
                        name='moving_points',
                        face_color='red')
    if args.save_intermediary == True:
        imwrite(path.join(args.output_folder, "Fixed_Segmentation", ".tif"), fp.data)
    viewer.add_points(fp.data,
                        name='fixed_points',
                        face_color='blue')

    registration_algorithm = 'Rigid CPD'
    moving, fixed, transformed, kwargs = point_cloud_registration(viewer.layers["moving_points"].data, 
                                                                  viewer.layers["fixed_points"].data, 
                                                                  algorithm=args.registration_algorithm,
                                                                  voxel_size=args.voxel_size,
                                                                  every_k_points=args.subsampling,
                                                                  max_iterations=args.iterations)
    
    if registration_algorithm == 'Affine CPD' or registration_algorithm == 'Rigid CPD':
        transformed = Points(moving, **kwargs)
    else:
        transformed = Points(transformed)
    
    warp_img = warp_image_volume(
            lm,
            em.data,
            args.registration_algorithm, 
            moving,
            transformed,
            interpolation_order=args.interpolation_order,
            approximate_grid=args.approximate_grid,
            sub_division_factor=args.sub_division_factor)
    viewer.add_layer(transformed)
    if isinstance(warp_img, list):
        layers = []
        for image_data in warp_img:
            data, kwargs = image_data
            viewer.add_image(data, **kwargs)
            layers.append(viewer.layers[kwargs['name']])
            imwrite(path.join(args.output_folder, kwargs['name'], ".tif"), data)

    else:
        data, kwargs = warp_img
        imwrite(path.join(args.output_folder, kwargs['name'], ".tif"), data)
        viewer.add_image(data, **kwargs)
        



if __name__ == "__main__":
    import argparse
    # Create the parser
    parser = argparse.ArgumentParser(description='CLEM-reg is the combination of 5 main steps, MitoNet segmentation, LoG segmentation, point cloud sampling, point cloud registration and lastly image warping.')
    # Create an argument group for EM segmentation
    main_group = parser.add_argument_group('General', 'Inputs, Mito Channel and Algorithm Selection')
    main_group.add_argument('--lm_input',
                        type=str,
                        required=True,
                        help='Here you select your light microscopy (LM) data which will be warped to align with the fixed electron microscopy (EM) image.')

    main_group.add_argument('--em_input',
                        type=str,
                        required=True,
                        help='Here you select your EM data which will act as the reference point for the LM to be aligned to.')
    main_group.add_argument('--input_folder',
                        type=str,
                        required=True,
                        help='Here you select your folder for your input data')



    main_group.add_argument('--mito_channel',
                        type=int,
                        required=False,
                        help='Optional input channel for multi-channel LM data')
    
    main_group.add_argument('--registration_algorithm',
                        choices=['BCPD', 'Rigid CPD', 'Affine CPD'],
                        required=True,
                        default='Rigid CPD',
                        help='Here you can decide which type of registration algorith will be used for the registration of inputted LM and EM. In terms of speed of each algorithm the following is the generally true, Rigid CPD > Affine CPD > BCPD.')   
    main_group.add_argument('--output_folder',
                        type=str,
                        required=False,
                        default="./output/",
                        help='Output folder for warped images')
    main_group.add_argument('--save_intermediary',
                        type=bool,
                        required=False,
                        default=False,
                        help='Saves intermediary files from all steps')

    
    log_segmentation_group = parser.add_argument_group('LoG Segmentation Parameters','Here are the advanced options for the segmentation of the mitochondria in the LM data. ')
    log_segmentation_group.add_argument('--log_sigma',
                        type=int,
                        required=False,
                        default=3,
                        help='Sigma value for the Laplacian of Gaussian filter.')

    log_segmentation_group.add_argument('--threshold',
                        type=int,
                        required=False,
                        default=1.2,
                        help='Threshold value for the segmenting the LM data.')



    
    point_sampling_group = parser.add_argument_group('Point Cloud Sampling', 'Here are the advanced options for the point cloud sampling of the segmentations of the LM and EM data.')
    point_sampling_group.add_argument('--sampling_freq',
                        type=int,
                        required=False,
                        default=0.01,
                        help='Frequency of point sampling from the fixed and moving segmentation. The greater the value the more points in the point cloud.')

    point_sampling_group.add_argument('--sampling_sigma',
                        type=int,
                        required=False,
                        default=1.0,
                        help='Sigma value for the canny edge filter')



    registration_group = parser.add_argument_group('Point Cloud Registration', 'Here are the advanced options for the registration of the point clouds of both the LM and EM data.')
    registration_group.add_argument('--voxel_size',
                        type=int,
                        required=False,
                        default=5,
                        help='The size voxel size of each point. Smaller the size the less memory consumption.')

    registration_group.add_argument('--subsampling',
                        type=int,
                        required=False,
                        default=1,
                        help='Downsampling of the point clouds to reduce memory consumption. Greater the number the fewer points in the point cloud.'),


    registration_group.add_argument('--iterations',
                        type=int,
                        required=False,
                        default=50, 
                        help="The number of round of point cloud registration. If too small it won't converge on an opitmal registration.")
    
    warping_group = parser.add_argument_group('Point Cloud Registration', 'Here are the advanced options for the registration of the point clouds of both the LM and EM data.')
    warping_group.add_argument('--interpolation_order',
                        type=int,
                        required=False, 
                        default=1,
                        help='The order of the spline interpolation')

    warping_group.add_argument('--approximate_grid',
                        type=int,
                        required=False,
                        default=5,
                        help="""Controls the "resolution" of the grid onto which you're warping. A higher value reduces the step size between coordinates."""),


    warping_group.add_argument('--sub_division_factor',
                        type=int,
                        required=False,
                        default=1, 
                        help="Controls the size of the chunk when applying the warping")
    
    em_group = parser.add_argument_group('MitoNet Segmentation Parameters', 'Here are the advanced options for the segmentation of the mitochondria in the EM data.')
    
    
    em_group = parser.add_argument_group('MitoNet Segmentation Parameters', 'Here are the advanced options for the segmentation of the mitochondria in the EM data.')


    # Add the arguments to the EM group
    em_group.add_argument('--config', type=str, metavar='config', help='Path to a model config yaml file')
    em_group.add_argument('--data-key', type=str, metavar='data-key', default='em',
                        help='Key in zarr volume (if volume_path is a zarr). For multiple keys, separate with a comma.')
    em_group.add_argument('--mode', type=str, dest='mode', metavar='inference_mode', choices=['orthoplane', 'stack'],
                        default='stack', help='Pick orthoplane (xy, xz, yz) or stack (xy)')
    em_group.add_argument('--qlen', type=int, dest='qlen', metavar='qlen', choices=[1, 3, 5, 7, 9, 11],
                        default=3, help='Length of median filtering queue, an odd integer')
    em_group.add_argument('--nmax', type=int, dest='label_divisor', metavar='label_divisor',
                        default=20000, help='Maximum number of objects per instance class allowed in volume.')
    em_group.add_argument('--seg-thr', type=float, dest='seg_thr', metavar='seg_thr', default=0.3,
                        help='Segmentation confidence threshold (0-1)')
    em_group.add_argument('--nms-thr', type=float, dest='nms_thr', metavar='nms_thr', default=0.1,
                        help='Centroid confidence threshold (0-1)')
    em_group.add_argument('--nms-kernel', type=int, dest='nms_kernel', metavar='nms_kernel', default=3,
                        help='Minimum allowed distance, in pixels, between object centers')
    em_group.add_argument('--iou-thr', type=float, dest='iou_thr', metavar='iou_thr', default=0.25,
                        help='Minimum IoU score between objects in adjacent slices for label stiching')
    em_group.add_argument('--ioa-thr', type=float, dest='ioa_thr', metavar='ioa_thr', default=0.25,
                        help='Minimum IoA score between objects in adjacent slices for label merging')
    em_group.add_argument('--pixel-vote-thr', type=int, dest='pixel_vote_thr', metavar='pixel_vote_thr', default=2,
                        choices=[1, 2, 3], help='Votes necessary per voxel when using orthoplane inference')
    em_group.add_argument('--cluster-iou-thr', type=float, dest='cluster_iou_thr', metavar='cluster_iou_thr', default=0.75,
                        help='Minimum IoU to group together instances after orthoplane inference')
    em_group.add_argument('--min-size', type=int, dest='min_size', metavar='min_size', default=500,
                        help='Minimum object size, in voxels, in the final 3d segmentation')
    em_group.add_argument('--min-span', type=int, dest='min_span', metavar='min_span', default=4,
                        help='Minimum number of consecutive slices that object must appear on in final 3d segmentation')
    em_group.add_argument('--downsample-f', type=int, dest='downsample_f', metavar='dowsample_f', default=1,
                        help='Factor by which to downsample images before inference, must be log base 2.')
    em_group.add_argument('--one-view', action='store_true',
                        help='One to allow instances seen in just 1 stack through to orthoplane consensus.')
    em_group.add_argument('--fine-boundaries', action='store_true',
                        help='Whether to calculate cells on full resolution image.')
    em_group.add_argument('--use-cpu', action='store_true', help='Whether to force inference to run on CPU.')
    em_group.add_argument('--save-panoptic', action='store_true',
                        help='Whether to save raw panoptic segmentation for each stack.')

    args = parser.parse_args()
    main()
