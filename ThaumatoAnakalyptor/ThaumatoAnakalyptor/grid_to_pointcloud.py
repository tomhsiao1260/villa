### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius Challenge 2023

from .surface_detection import surface_detection
import torch
import numpy as np
import tifffile
import tqdm
from PIL import Image
import open3d as o3d
import time

import os
import numpy as np
import torch
torch.set_float32_matmul_precision('medium')
from scipy.interpolate import interp1d
from .add_random_colors_to_pointcloud import add_random_colors
# import torch.multiprocessing as multiprocessing
import multiprocessing
import glob
import argparse

CFG = {'num_threads': 4, 'GPUs': 1}

# Signal handler function
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

def cleanup_temp_files(directory):
    # Get a list of all .temp files in the specified directory
    temp_files = glob.glob(os.path.join(directory, '*_temp.ply'))

    # Report if any temp files are found
    if temp_files:
        print(f"Found {len(temp_files)} leftover .temp files. Cleaning up...")
        for file in temp_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
    else:
        print("No leftover .temp files found.")

def load_grid(path_template, cords, grid_block_size=500, cell_block_size=500, uint8=True):
        """
        path_template: Template for the path to load individual grid files
        cords: Tuple (x, y, z) representing the corner coordinates of the grid block
        grid_block_size: Size of the grid block
        cell_block_size: Size of the individual grid files
        """
        # make grid_block_size an array with 3 elements
        if isinstance(grid_block_size, int):
            grid_block_size = np.array([grid_block_size, grid_block_size, grid_block_size])
        
        # Convert corner coordinates to file indices and generate the file path
        # Starting indices
        file_x_start, file_y_start, file_z_start = cords[0]//cell_block_size, cords[1]//cell_block_size, cords[2]//cell_block_size
        # Ending indices
        file_x_end, file_y_end, file_z_end = (cords[0] + grid_block_size[0])//cell_block_size, (cords[1] + grid_block_size[1])//cell_block_size, (cords[2] + grid_block_size[2])//cell_block_size

        # Generate the grid block
        if uint8:
            grid_block = np.zeros((grid_block_size[2], grid_block_size[0], grid_block_size[1]), dtype=np.uint8)
        else:
            grid_block = np.zeros((grid_block_size[2], grid_block_size[0], grid_block_size[1]), dtype=np.uint16)

        # Load the grid block from the individual grid files and place it in the larger grid block
        for file_x in range(file_x_start, file_x_end + 1):
            for file_y in range(file_y_start, file_y_end + 1):
                for file_z in range(file_z_start, file_z_end + 1):
                    path = path_template.format(file_x, file_y, file_z)

                    # Check if the file exists
                    if not os.path.exists(path):
                        # print(f"File {path} does not exist.")
                        continue

                    # Read the image
                    with tifffile.TiffFile(path) as tif:
                        images = tif.asarray()

                    if uint8:
                        images = np.uint8(images//256)

                    # grid block slice position for the current file
                    x_start = max(file_x*cell_block_size, cords[0])
                    x_end = min((file_x + 1) * cell_block_size, cords[0] + grid_block_size[0])
                    y_start = max(file_y*cell_block_size, cords[1])
                    y_end = min((file_y + 1) * cell_block_size, cords[1] + grid_block_size[1])
                    z_start = max(file_z*cell_block_size, cords[2])
                    z_end = min((file_z + 1) * cell_block_size, cords[2] + grid_block_size[2])

                    # Place the current file in the grid block
                    try:
                        grid_block[z_start - cords[2]:z_end - cords[2], x_start - cords[0]:x_end - cords[0], y_start - cords[1]:y_end - cords[1]] = images[z_start - file_z*cell_block_size: z_end - file_z*cell_block_size, x_start - file_x*cell_block_size: x_end - file_x*cell_block_size, y_start - file_y*cell_block_size: y_end - file_y*cell_block_size]
                    except:
                        print(f"Error in grid block placement for grid block {cords} and file {file_x}, {file_y}, {file_z}")

        return grid_block

def save_surface_ply(surface_points, normals, filename, color=None):
    if (len(surface_points)  < 1):
        return
    # Create an Open3D point cloud object and populate it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_points.astype(np.float32))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float16))
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color.astype(np.float16))

    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save to a temporary file first to ensure data integrity
    temp_filename = filename.replace(".ply", "_temp.ply")
    o3d.io.write_point_cloud(temp_filename, pcd)

    # Rename the temp file to the original filename
    os.rename(temp_filename, filename)

def generate_line(start_point, direction_vector, filename, length=500, step=0.1):
    """
    Generate a line in 3D space.
    
    :param start_point: Starting point of the line as a 1x3 numpy array.
    :param direction_vector: Direction of the line as a 1x3 numpy array. This will be normalized.
    :param length: Length of the line.
    :param step: Distance between consecutive points.
    :return: A 2D numpy array containing the points of the line.
    """
    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    
    # Create the points
    num_points = int(length / step)
    points = np.array([start_point + i * step * direction_vector for i in range(num_points)])
    
    line = np.array(points)

    yellow = np.array([1, 1, 0])  # RGB for yellow
    colors = np.tile(yellow, (points.shape[0], 1))
    save_surface_ply(line, np.zeros_like(line), filename, color=colors)

def load_xyz_from_file(filename='umbilicus.txt'):
    """
    Load a file with comma-separated xyz coordinates into a 2D numpy array.
    
    :param filename: The path to the file.
    :return: A 2D numpy array of shape (n, 3) where n is the number of lines/coordinates in the file.
    """
    return np.loadtxt(filename, delimiter=',')


def umbilicus(points_array):
    """
    Interpolate between points in the provided 2D array based on y values.

    :param points_array: A 2D numpy array of shape (n, 3) with x, y, and z coordinates.
    :return: A 2D numpy array with interpolated points for each 0.1 step in the y direction.
    """

    # Separate the coordinates
    x, y, z = points_array.T

    # Create interpolation functions for x and y based on z
    fx = interp1d(y, x, kind='linear', fill_value="extrapolate")
    fz = interp1d(y, z, kind='linear', fill_value="extrapolate")

    # Define new y values for interpolation
    y_new = np.arange(y.min(), y.max(), 1)

    # Calculate interpolated x and y values
    x_new = fx(y_new)
    z_new = fz(y_new)

    # Return the combined x, y, and z values as a 2D array
    return np.column_stack((x_new, y_new, z_new))

def umbilicus_xz_at_y(points_array, y_new):
    """
    Interpolate between points in the provided 2D array based on y values.

    :param points_array: A 2D numpy array of shape (n, 3) with x, y, and z coordinates.
    :return: A 2D numpy array with interpolated points for each 0.1 step in the y direction.
    """

    # Separate the coordinates
    x, y, z = points_array.T

    # Create interpolation functions for x and z based on y
    fx = interp1d(y, x, kind='linear', fill_value="extrapolate")
    fz = interp1d(y, z, kind='linear', fill_value="extrapolate")

    # Calculate interpolated x and z values
    x_new = fx(y_new)
    z_new = fz(y_new)

    # Return the combined x, y, and z values as a 2D array
    res = np.array([x_new, y_new, z_new])
    return res

# Picks block closest to the edge of the complete volume to more accurately calculate global reference vectors (less curvature outside the scroll)
def pick_block(blocks, xy_min=2000, xy_max=6000, z_min=500, z_max=5000):

    min_border_distance = float("inf")
    min_border_block = None
    for block in blocks:
        x, y, z = block
        if abs(x - xy_min) < min_border_distance:
            min_border_distance = x - xy_min
            min_border_block = block
        if abs(xy_max - x) < min_border_distance:
            min_border_distance = xy_max - x
            min_border_block = block
        if abs(y - xy_min) < min_border_distance:
            min_border_distance = y - xy_min
            min_border_block = block
        if abs(xy_max - y) < min_border_distance:
            min_border_distance = xy_max - y
            min_border_block = block

    print("Picked block:", min_border_block, "Distance to border:", min_border_distance)
    return min_border_block

# Picks block radius away from center of the scroll
def pick_block_away_from_center(blocks, umbilicus_points, radius=1000, grid_block_size=500):
    min_border_distance = float("inf")
    min_border_block = None
    for block in blocks:
        y, x, z = block
        umbilicus_point = umbilicus_xz_at_y(umbilicus_points, z-grid_block_size//2)
        umbilicus_point = umbilicus_point[[0, 2, 1]] - grid_block_size//2 # cell/corner coord
        cost = abs(abs((y-(umbilicus_point[0]))**2 + (x-(umbilicus_point[1]))**2) - radius**2)
        if cost < min_border_distance:
            min_border_distance = cost
            min_border_block = block

    print("Picked block:", min_border_block, "Distance to border:", min_border_distance)
    return min_border_block

def extract_size(points, normals, grid_block_position_min, grid_block_position_max):
    """
    Extract points and corresponding normals that lie within the given size range.

    Parameters:
        points (numpy.ndarray): The point coordinates, shape (n, 3).
        normals (numpy.ndarray): The point normals, shape (n, 3).
        grid_block_position_min (numpy.ndarray): The minimum block size, shape (3,).
        grid_block_position_max (numpy.ndarray): The maximum block size, shape (3,).

    Returns:
        filtered_points (numpy.ndarray): The filtered points, shape (m, 3).
        filtered_normals (numpy.ndarray): The corresponding filtered normals, shape (m, 3).
    """
    
    # Create a mask to filter points within the specified range
    mask_min = np.all(points >= grid_block_position_min, axis=-1)
    mask_max = np.all(points <= grid_block_position_max, axis=-1)
    
    # Combine the masks to get the final mask
    mask = np.logical_and(mask_min, mask_max)
    
    # Apply the mask to filter points and corresponding normals
    filtered_points = points[mask]
    filtered_normals = normals[mask]

    # Reposition the points to be relative to the grid block
    filtered_points -= grid_block_position_min
    
    return filtered_points, filtered_normals

def process_block(args):
    corner_coords, blocks_to_process, blocks_processed, umbilicus_points, umbilicus_points_old, lock, path_template, save_template_v, save_template_r, grid_block_size, recompute, fix_umbilicus, computed_block, maximum_distance, gpu_num = args
    if fix_umbilicus:
        fix_umbilicus_indicator = fix_umbilicus_recompute(corner_coords, grid_block_size, umbilicus_points, umbilicus_points_old)
    else:
        fix_umbilicus_indicator = False
    recompute = recompute or fix_umbilicus_indicator

    skip_computation_flag = skip_computation_block(corner_coords, grid_block_size, umbilicus_points, maximum_distance=maximum_distance)
    # Pick a block to process
    blocks_to_process.remove(corner_coords)
    blocks_processed[corner_coords] = True

    # Load the grid block from corner_coords and grid size
    padding = 50
    corner_coords_padded = np.array(corner_coords) - padding
    grid_block_size_padded = grid_block_size + 2 * padding

    # Save the surface points and normals as a PLY file
    file_x, file_y, file_z = corner_coords[0]//grid_block_size, corner_coords[1]//grid_block_size, corner_coords[2]//grid_block_size
    surface_ply_filename_v = save_template_v.format(file_x, file_y, file_z)
    surface_ply_filename_r = save_template_r.format(file_x, file_y, file_z)

    if (not skip_computation_flag) and (recompute or not (os.path.exists(surface_ply_filename_r) and os.path.exists(surface_ply_filename_v))) and (not computed_block): # Recompute if file doesn't exist or recompute flag is set
        print(f"Skip computation flag: {skip_computation_flag}, Recompute flag: {recompute}, path verso exists: {os.path.exists(surface_ply_filename_v)}, path recto exists: {os.path.exists(surface_ply_filename_r)}")
        # Load padded grid block
        block = load_grid(path_template, corner_coords_padded, grid_block_size=grid_block_size_padded)
        # Check if the block is empty
        if np.all(block == 0):
            return False
        
        device = torch.device("cuda:" + str(gpu_num))
        block = torch.tensor(np.array(block), device=device, dtype=torch.float16)
    
        block_point = np.array(corner_coords) + grid_block_size//2
        umbilicus_point = umbilicus_xz_at_y(umbilicus_points, block_point[2])
        umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords
        umbilicus_normal = block_point - umbilicus_point
        umbilicus_normal = umbilicus_normal[[2, 0, 1]] # corner coords to tif
        unit_umbilicus_normal = umbilicus_normal / np.linalg.norm(umbilicus_normal)
        # dtype float 32
        reference_vector = torch.tensor(unit_umbilicus_normal, device=device, dtype=torch.float32)
        # Perform surface detection for this block
        recto, verso = surface_detection(block, reference_vector, blur_size=11, window_size=9, stride=1, threshold_der=0.075, threshold_der2=0.002)
        points_r, normals_r = recto
        points_v, normals_v = verso
        # Extract actual volume size from the oversized input block
        points_r, normals_r = extract_size(points_r, normals_r, padding, grid_block_size+padding) # 0, 0, 0 is the minimum corner of the grid block
        points_v, normals_v = extract_size(points_v, normals_v, padding, grid_block_size+padding) # 0, 0, 0 is the minimum corner of the grid block

        ### Adjust the 3D coordinates of the points based on their position in the larger volume

        # permute the axes to match the original volume
        points_r = points_r[:, [1, 0, 2]]
        normals_r = normals_r[:, [1, 0, 2]]
        points_v = points_v[:, [1, 0, 2]]
        normals_v = normals_v[:, [1, 0, 2]]

        y_d, x_d, z_d = corner_coords
        points_r += np.array([y_d, z_d, x_d])
        points_v += np.array([y_d, z_d, x_d])

        save_surface_ply(points_r, normals_r, surface_ply_filename_r)
        save_surface_ply(points_v, normals_v, surface_ply_filename_v)

    if not skip_computation_flag:
        # Compute neighboring blocks
        for dx in [-grid_block_size, 0, grid_block_size]:
            for dy in [-grid_block_size, 0, grid_block_size]:
                for dz in [-grid_block_size, 0, grid_block_size]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if abs(dx) + abs(dy) + abs(dz) > grid_block_size:
                        continue
                    neighbor_coords = (corner_coords[0] + dx, corner_coords[1] + dy, corner_coords[2] + dz)
                    
                    # Add the neighbor to the list of blocks to process if it hasn't been processed yet
                    with lock:
                        # Add the neighbor to the list of blocks to process if it hasn't been processed yet
                        if neighbor_coords not in blocks_processed and neighbor_coords not in blocks_to_process:
                            blocks_to_process.append(neighbor_coords)

    return True

# fixing the pointcloud because of computation with too short umbilicus
def fix_umbilicus_recompute(corner_coords, grid_block_size, umbilicus_points, umbilicus_points_old, additional_distance=300):
    block_point = np.array(corner_coords) + grid_block_size//2
    umbilicus_point = umbilicus_xz_at_y(umbilicus_points, block_point[2])
    umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords
    umbilicus_point_old = umbilicus_xz_at_y(umbilicus_points_old, block_point[2])
    umbilicus_point_old = umbilicus_point_old[[0, 2, 1]] # ply to corner coords
    umbilicus_point_middle = (umbilicus_point + umbilicus_point_old) / 2
    umbilicus_point_dist = umbilicus_point_middle - umbilicus_point
    umbilicus_point_dist = np.linalg.norm(umbilicus_point_dist)
    # block dist to disaster
    umbilicus_dist_middle = block_point - umbilicus_point_middle
    umbilicus_dist_middle = np.linalg.norm(umbilicus_dist_middle)
    if umbilicus_dist_middle <= umbilicus_point_dist * 1.5 + additional_distance:
        return True
    return False

# fixing the pointcloud because of computation with too short umbilicus
def skip_computation_block(corner_coords, grid_block_size, umbilicus_points, maximum_distance=2500):
    if maximum_distance <= 0:
        return False
    
    block_point = np.array(corner_coords) + grid_block_size//2
    umbilicus_point = umbilicus_xz_at_y(umbilicus_points, block_point[2])
    umbilicus_point = umbilicus_point[[0, 2, 1]] # ply to corner coords

    umbilicus_point_dist = umbilicus_point - block_point
    umbilicus_point_dist = np.linalg.norm(umbilicus_point_dist)
    return umbilicus_point_dist > maximum_distance


def compute_surface_for_block_multiprocessing(corner_coords, pointcloud_base, path_template, save_template_v, save_template_r, umbilicus_points, grid_block_size=500, recompute=False, fix_umbilicus=False, umbilicus_points_old=None, maximum_distance=2500):
    # Create a manager for shared data
    manager = multiprocessing.Manager()
    blocks_to_process = manager.list([corner_coords])
    blocks_processed = manager.dict()
    lock = manager.Lock()

    computed_blocks = set()
    # Try to load the list of computed blocks
    try:
        with open(os.path.join(pointcloud_base, "computed_blocks.txt"), "r") as f:
            # load saved tuples with 3 elements
            computed_blocks = set([eval(line.strip()) for line in f])
    except Exception as e:
        print(f"Error loading computed blocks: {e}")

    # Limit the number of concurrent jobs to, for instance, 4. You can change this value as desired.
    # for 2 threads:
    # Blocks processed: 0 Blocks to process: 1 Time per block: Unknown
    # Blocks processed: 1 Blocks to process: 6 Time per block: 5.990274429321289
    # Blocks processed: 5 Blocks to process: 16 Time per block: 3.1694461345672607
    # Blocks processed: 13 Blocks to process: 28 Time per block: 2.626911860245925
    # Blocks processed: 25 Blocks to process: 40 Time per block: 2.517346658706665
    # Blocks processed: 41 Blocks to process: 52 Time per block: 2.382093807546104
    # for 3 threads:
    # Blocks processed: 0 Blocks to process: 1 Time per block: Unknown
    # Blocks processed: 1 Blocks to process: 6 Time per block: 5.592769622802734
    # Blocks processed: 5 Blocks to process: 16 Time per block: 3.340130424499512
    # Blocks processed: 13 Blocks to process: 28 Time per block: 2.6134009177868185
    # Blocks processed: 25 Blocks to process: 40 Time per block: 2.4300418853759767
    # Blocks processed: 41 Blocks to process: 52 Time per block: 2.366847770970042
    # for 4 threads: SLOWER

    pool = multiprocessing.Pool(processes=CFG['num_threads'])

    # Timing
    processed_nr = 0
    # This loop ensures that all tasks in the list are completed.
    while len(blocks_to_process) > 0:
        start_time = time.time()
        current_blocks = list(set(list(blocks_to_process)))  # Take a snapshot of current blocks

        current_block_batches = [current_blocks[i:min(len(current_blocks), i+3*CFG['num_threads'])] for i in range(0, len(current_blocks), 3*CFG['num_threads'])]
        # Initialize the tqdm progress bar
        with tqdm.tqdm(total=len(current_blocks)) as pbar:
            for current_block_batch in current_block_batches:
                if len(current_block_batch) == 0:
                    continue
                # Process each block and update the progress bar upon completion of each block
                for _ in pool.imap(process_block, [(block, blocks_to_process, blocks_processed, umbilicus_points, umbilicus_points_old, lock, path_template, save_template_v, save_template_r, grid_block_size, recompute, fix_umbilicus, block in computed_blocks, maximum_distance, proc_nr % CFG['GPUs']) for proc_nr, block in enumerate(current_block_batch)]):
                    pbar.update(1)

                computed_blocks.update(current_block_batch)
                # Save the computed blocks
                with open(os.path.join(pointcloud_base, "computed_blocks.txt"), "w") as f:
                    for block in computed_blocks:
                        f.write(str(block) + "\n")


                torch.cuda.empty_cache()

        # results = list(tqdm.tqdm(pool.imap(process_block, [(block, blocks_to_process, blocks_processed, umbilicus_points, umbilicus_points_old, lock, path_template, save_template_v, save_template_r, grid_block_size, recompute, fix_umbilicus, maximum_distance, proc_nr % CFG['GPUs']) for proc_nr, block in enumerate(current_blocks)]), total=len(current_blocks)))
        current_time = time.time()
        print("Blocks total processed:", len(blocks_processed), "Blocks to process:", len(blocks_to_process), "Time per block:", f"{(current_time - start_time) / (len(blocks_processed) - processed_nr):.3f}" if len(blocks_processed)-processed_nr > 0 else "Unknown")
        processed_nr = len(blocks_processed)

def compute(disk_load_save, base_path, volume_subpath, pointcloud_subpath, maximum_distance, recompute, fix_umbilicus, start_block, num_threads, gpus):
    # Initialize CUDA context
    # _ = torch.tensor([0.0]).cuda()
    try:
        multiprocessing.set_start_method('spawn')
    except Exception as e:
        print(f"Error setting multiprocessing start method: {e}")

    CFG['num_threads'] = num_threads
    CFG['GPUs'] = gpus

    pointcloud_base = os.path.dirname(pointcloud_subpath)
    pointcloud_subpath_recto = pointcloud_subpath + "_recto"
    pointcloud_subpath_verso = pointcloud_subpath + "_verso"
    pointcloud_subpath_colorized = pointcloud_subpath + "_colorized"
    src_dir = base_path + "/" + volume_subpath + "/"
    dest_dir_r = src_dir.replace(volume_subpath, pointcloud_subpath_recto).replace(disk_load_save[0], disk_load_save[1])
    dest_dir_v = src_dir.replace(volume_subpath, pointcloud_subpath_verso).replace(disk_load_save[0], disk_load_save[1])

    # Cleanup temp files
    cleanup_temp_files(dest_dir_r)
    cleanup_temp_files(dest_dir_v)
    path_template = src_dir + "cell_yxz_{:03}_{:03}_{:03}.tif"
    save_template_r = path_template.replace(".tif", ".ply").replace(volume_subpath, pointcloud_subpath_recto).replace(disk_load_save[0], disk_load_save[1])
    save_template_v = path_template.replace(".tif", ".ply").replace(volume_subpath, pointcloud_subpath_verso).replace(disk_load_save[0], disk_load_save[1])

    umbilicus_path = src_dir + "umbilicus.txt"
    save_umbilicus_path = umbilicus_path.replace(".txt", ".ply").replace(disk_load_save[0], disk_load_save[1])
    save_umbilicus_path = save_umbilicus_path.replace(volume_subpath, pointcloud_subpath)

    # Copy umbilicus.txt to pointcloud_subpath dir
    umbilicus_copy_path = os.path.dirname(save_umbilicus_path)
    #directory of umbilicus_copy_path directory
    umbilicus_copy_path = os.path.dirname(umbilicus_copy_path) + "/umbilicus.txt"
    os.makedirs(os.path.dirname(umbilicus_copy_path), exist_ok=True)
    os.system("cp " + umbilicus_path + " " + umbilicus_copy_path)

    # Usage
    umbilicus_raw_points = load_xyz_from_file(umbilicus_path)
    umbilicus_points = umbilicus(umbilicus_raw_points)
    # Red color for umbilicus
    colors = np.zeros_like(umbilicus_points)
    colors[:,0] = 1.0
    # Save umbilicus as a PLY file, for visualization (CloudCompare)
    save_surface_ply(umbilicus_points, np.zeros_like(umbilicus_points), save_umbilicus_path, color=colors)

    if fix_umbilicus:
        # Load old umbilicus
        umbilicus_path_old = src_dir + "umbilicus_old.txt"
        # Usage
        umbilicus_raw_points_old = load_xyz_from_file(umbilicus_path_old)
        umbilicus_points_old = umbilicus(umbilicus_raw_points_old)
    else:
        umbilicus_points_old = None

    # Starting grid block at corner (3000, 4000, 2000) to match cell_yxz_006_008_004
    # (2600, 2200, 5000)
    compute_surface_for_block_multiprocessing(start_block, pointcloud_base, path_template, save_template_v, save_template_r, umbilicus_points, grid_block_size=200, recompute=recompute, fix_umbilicus=fix_umbilicus, umbilicus_points_old=umbilicus_points_old, maximum_distance=maximum_distance)

    # Sample usage:
    # src is folder of save_umbilicus_path
    dest_folder_v = dest_dir_v.replace(pointcloud_subpath, pointcloud_subpath_colorized).replace(disk_load_save[0], disk_load_save[1])
    add_random_colors(dest_dir_v, dest_folder_v)

    dest_folder_r = dest_dir_r.replace(pointcloud_subpath, pointcloud_subpath_colorized).replace(disk_load_save[0], disk_load_save[1])
    add_random_colors(dest_dir_r, dest_folder_r)

def main():
    # Parse arguments defaults
    maximum_distance = 2300 # scroll 3 all
    maximum_distance= -1 #1750 # maximum distance between blocks to compute and the umbilicus (speed up pointcloud generation if only interested in inner part of scrolls)
    recompute=False # whether to completely recompute all already processed blocks or continue (recompute=False). 
    fix_umbilicus = False
    disk_load_save = ["SSD4TB", "SSD2"] # Disk that contains input data, and disk that should be used to save output data
    base_path = "/media/julian/SSD4TB"
    volume_subpath = "PHerc0332.volpkg/volumes/2dtifs_8um_grids"
    pointcloud_subpath = "scroll3_surface_points/point_cloud"
    # start_block = (3000, 4000, 2000) # scroll1
    start_block = (500, 500, 500)

    parser = argparse.ArgumentParser(description="Extract papyrus sheet surface points from a 3D volume of a scroll CT scan")
    parser.add_argument("--base_path", type=str, help="Base path to the data", default=base_path)
    parser.add_argument("--disk_load_save", type=str, nargs=2, help="Disk that contains input data, and dist that should be used to save output data", default=disk_load_save)
    parser.add_argument("--volume_subpath", type=str, help="Subpath to the volume data", default=volume_subpath)
    parser.add_argument("--pointcloud_subpath", type=str, help="Subpath to the pointcloud data", default=pointcloud_subpath)
    parser.add_argument("--max_umbilicus_dist", type=float, help="Maximum distance between the umbilicus and blocks that should be computed. -1.0 for no distance restriction", default=-1.0)
    parser.add_argument("--recompute", action='store_true', help="Flag, recompute all blocks, even if they already exist")
    parser.add_argument("--fix_umbilicus", action='store_true', help="Flag, recompute all close to the updated umbilicus (make sure to also save the old umbilicus.txt as umbilicus_old.txt)")
    parser.add_argument("--start_block", type=int, nargs=3, help="Starting block to compute", default=start_block)
    parser.add_argument("--num_threads", type=int, help="Number of threads to use", default=CFG['num_threads'])
    parser.add_argument("--gpus", type=int, help="Number of GPUs to use", default=CFG['GPUs'])

    args = parser.parse_args()

    # Print arguments
    print(f"Arguments: {args}")

    # update variables from arguments
    disk_load_save = args.disk_load_save
    base_path = args.base_path
    volume_subpath = args.volume_subpath
    pointcloud_subpath = args.pointcloud_subpath
    maximum_distance = args.max_umbilicus_dist
    recompute = args.recompute
    fix_umbilicus = args.fix_umbilicus
    start_block = tuple(args.start_block)
    
    # Compute the surface points
    compute(disk_load_save, base_path, volume_subpath, pointcloud_subpath, maximum_distance, recompute, fix_umbilicus, start_block, args.num_threads, args.gpus)

if __name__ == "__main__":
    main()