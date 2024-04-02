from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import imageio
import cv2

def load_colmap_data():
    r"""
    After using colmap2nerf.py to convert the colmap intrinsics and extrinsics,
    read in the transform_colmap.json file

    Expected Returns:
      An array of resized imgs, normalized to [0, 1]
      An array of poses, essentially the transform matrix
      Camera parameters: H, W, focal length

    NOTES:
      We recommend you resize the original images from 800x800 to lower resolution,
      i.e. 200x200 so it's easier for training. Change camera parameters accordingly
    """
    ################### YOUR CODE START ###################
    # Load the data from transform.json
    with open('/home/griffin/Documents/ese6500-hw3/gaddison_hw3_problem3/data/data/transforms.json') as f:
        data = json.load(f)


    # Set desired width and height
    original_width = data['w']
    original_height = data['h']
    new_width = 200
    new_height = 200


    # Load the images
    imgs = []
    # For each image

    # print("len(data['frames'])", len(data['frames']))
    for i in range(len(data['frames'])):
        # Read the image
        image_path_prefix = "/home/griffin/Documents/ese6500-hw3/gaddison_hw3_problem3/data/data/images/"
        # print(data['frames'][i]['file_path'][0])
        image_path = image_path_prefix + data['frames'][i]['file_path'][0]
        img = cv2.imread(image_path)
        # Resize the image to 200x200
        img = cv2.resize(img, (new_width, new_height))
        # MAYBE: Normalize the image color to [0, 1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        # Append the image to our list
        imgs.append(img)
    imgs = torch.tensor(imgs, dtype=torch.float32)


    # Load the poses
    poses = []
    for i in range(len(data['frames'])):
        poses.append(data['frames'][i]['transform_matrix'])
    poses = torch.tensor(poses)


    H = new_height
    W = new_width
    focal_length = (data['fl_x'] + data['fl_y'] / 2) * (new_width / original_width)

    # Create dictionary of cam params so i can call each by name
    # cam_params = {'H': H, 'W': W, 'focal_length': focal_length} 
    cam_params = [H, W, focal_length]

    return imgs, poses, cam_params

    ################### YOUR CODE END ###################


def get_rays(H, W, focal_length, pose):
    r"""Compute rays passing through each pixels

    Expected Returns:
      ray_origins: A tensor of shape (H, W, 3) denoting the centers of each ray.
      ray_directions: A tensor of shape (H, W, 3) denoting the direction of each 
        ray. ray_directions[i][j] denotes the direction (x, y, z) of the ray 
        passing through the pixel at row index `i` and column index `j`.
    """
    ################### YOUR CODE START ###################
       # Create meshgrid for pixel coordinates (H, W)
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    pixels = torch.stack([j, i, torch.ones_like(i)], dim=-1)  # Shape: (H, W, 3)
    
    # Calculate K inverse once since it's constant
    K_inv = torch.linalg.inv(torch.tensor([[focal_length, 0, W / 2],
                                           [0, focal_length, H / 2],
                                           [0, 0, 1]], dtype=torch.float32))
    
    # Transform pixel coordinates to world coordinates
    pixels_world = torch.einsum('ij,hwj->hwi', K_inv, pixels.float())
    
    # Calculate ray directions
    ray_directions = torch.einsum('ij,hwj->hwi', pose[:3, :3], pixels_world)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)
    
    # Calculate ray origins
    ray_origins = pose[:3, 3].expand(H, W, 3)  # The origin is the same for all rays
    
    return ray_origins, ray_directions
    # ray_origins = torch.zeros((H, W, 3))
    # ray_directions = torch.zeros((H, W, 3))
    #
    # K_inv = torch.linalg.inv(torch.tensor([[focal_length, 0, W/2],\
    #               [0, focal_length, H/2],\
    #               [0, 0, 1]]))
    # # For each pixel
    # for i in range(H):
    #     for j in range(W):
    #         # Calculate ray origin
    #         ray_origins[i][j] = pose[:3, 3]
    #
    #         # Calculate ray direction
    #         pixel_c = K_inv @ torch.tensor([j, i, 1], dtype=torch.float32)
    #         ray_direction_w = pose[:3, :3] @ pixel_c
    #         ray_directions[i][j] = ray_direction_w / torch.linalg.norm(ray_direction_w)
    #
    #
    # return ray_origins, ray_directions
    ################### YOUR CODE END ###################


def sample_points_from_rays(ray_origins, ray_directions, snear, sfar, Nsample):
    r"""Compute a set of 3D points given the bundle of rays

    Expected Returns:
      sampled_points: axis of the sampled points along each ray, shape (H, W, num_samples, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)
    """
    ################### YOUR CODE START ###################
   #  H, W, _ = ray_origins.shape
   #  sampled_points = torch.zeros((H, W, Nsample, 3))
   #  depth_values = torch.zeros((H, W, Nsample))
   # 
   #  # Calculate segment length for jittering
   #  segment_lengths = (sfar - snear) / Nsample
   #
   #  # for i in range(H):
   #  #     for j in range(W):
   #  #         # Sample depth values deterministically
   #  #         depth_values_k = torch.linspace(snear, sfar, Nsample)
   #  #         # Add uniform noise to jitter the samples for this ray
   #  #         # sample_jitters = np.random.uniform(-segment_lengths / 2, segment_lengths / 2, Nsample)
   #  #         # Torch equivalent:
   #  #         # sample_jitters = (torch.rand(Nsample) - 0.5) * segment_lengths
   #  #         # depth_values += sample_jitters
   #  #
   #  #         for k in range(Nsample):
   #  #             # Calculate the point at depth k along the ray (i, j)
   #  #             sampled_points[i, j, k] = ray_origins[i, j] + ray_directions[i, j] * depth_values_k[k]
   #  #             depth_values[i, j, k] = depth_values_k[k]
   #  # # print("sampled_point.shape: ", sampled_points.shape)
   #  #
   #
   #  ray_origins = ray_origins[..., None, :]
   #  ray_directions = ray_directions[..., None, :]
   #  for k in range(Nsample):
   #      depth_values[..., k] = snear + (sfar - snear) * (k + 0.5) / Nsample
   #      print("ray_origins.shape: ", ray_origins.shape, "ray_directions.shape: ", ray_directions.shape)
   #      print("depth_values[..., k].unsqueeze(-1).shape: ", depth_values[..., k].unsqueeze(-1).shape)
   #      print("ray_directions * depth_values[..., k].unsqueeze(-1).shape",(ray_directions * depth_values[..., k].unsqueeze(-1)).shape)
   #      sampled_points[..., k, :] = ray_origins + ray_directions * depth_values[..., k].unsqueeze(-1)
   #
    H, W, _ = ray_origins.shape
    # Shape: (H, W, Nsample)
    depth_values = torch.linspace(snear, sfar, Nsample).view(1, 1, Nsample).expand(H, W, Nsample)

    # Optional: Add jitter to depth values to sample points in between segments
    # segment_lengths = (sfar - snear) / Nsample
    # sample_jitters = (torch.rand(H, W, Nsample) - 0.5) * segment_lengths
    # depth_values += sample_jitters

    # Calculate the 3D points
    # Expand ray_origins and ray_directions to match depth_values shape for broadcasting
    # Shape: (H, W, Nsample, 3)
    ray_origins_exp = ray_origins.unsqueeze(2).expand(-1, -1, Nsample, -1)
    ray_directions_exp = ray_directions.unsqueeze(2).expand(-1, -1, Nsample, -1)
    # Calculate the sampled points
    sampled_points = ray_origins_exp + ray_directions_exp * depth_values.unsqueeze(-1)

    return sampled_points, depth_values    ################### YOUR CODE END ###################


def positional_encoding(pos_in, max_freq_power=10, include_input=True):
    r"""Apply positional encoding to the input. (Section 5.1 of original paper)
    We use positional encoding to map continuous input coordinates into a 
    higher dimensional space to enable our MLP to more easily approximate a 
    higher frequency function.

    Expected Returns:
      pos_out: positional encoding of the input tensor. 
               (H*W*num_samples, (include_input + 2*freq) * 3)
    """
    ################### YOUR CODE START ###################
    frequencies = 2 ** torch.arange(max_freq_power + 1).to(pos_in.device, pos_in.dtype)
    sin_encodings = []
    cos_encodings = []

    for dim in range(3):
        dim_freqs = pos_in[:, dim:dim+1] * frequencies * torch.pi
        sin_encodings.append(torch.sin(dim_freqs))
        cos_encodings.append(torch.cos(dim_freqs))

    sin_encodings = torch.cat(sin_encodings, dim=-1)
    cos_encodings = torch.cat(cos_encodings, dim=-1)
    encoded = torch.cat([sin_encodings, cos_encodings], dim=-1)

    if include_input:
        encoded = torch.cat([pos_in, encoded], dim=-1)

    return encoded
    # frequencies = 2 ** torch.arange(max_freq_power + 1)
    # sin_encodings = []
    # cos_encodings = []
    #
    # # Using broadcasting to apply sine and cosine functions to all frequencies and dimensions at once
    # for dim in range(3):  # Assuming pos_in is [N, 3]
    #     # Expand dimension for broadcasting: [N, 1] to [N, 1, 1] and frequencies to [1, 1, F]
    #     dim_freqs = (pos_in[:, dim:dim+1].unsqueeze(-1) * frequencies.to(pos_in.device).float()).numpy() * np.pi
    #     sin_encodings.append(torch.sin(torch.tensor(dim_freqs)))
    #     cos_encodings.append(torch.cos(torch.tensor(dim_freqs)))
    #
    # # Concatenate all sine and cosine encodings along the last dimension, then reshape to [N, F*3*2]
    # sin_encodings = torch.cat(sin_encodings, dim=2).view(pos_in.shape[0], -1)
    # cos_encodings = torch.cat(cos_encodings, dim=2).view(pos_in.shape[0], -1)
    # encoded = torch.cat([sin_encodings, cos_encodings], dim=1)
    #
    # # Include the input if specified
    # if include_input:
    #     encoded = torch.cat([pos_in, encoded], dim=1)
    #
    # return encoded
#num_points = pos_in.shape[0]
    #pos_out = torch.tensor([])
    #frequencies = [2**i for i in range(max_freq_power+1)]

    #if include_input:
    #    pos_out = pos_in
    # print("pos_out.shape: ", pos_out.shape)
    # For each frequency, each of xyz, each of sin and cos, append this col to pos_out
    #for freq in frequencies:
        #    for dim in range(3):
            # print("pos_out.shape: ", pos_out.shape)
    #        pos_out = torch.cat([pos_out, torch.sin(freq * np.pi * pos_in[:, dim]).unsqueeze(1)], dim=1)
    #        pos_out = torch.cat([pos_out, torch.cos(freq * np.pi * pos_in[:, dim]).unsqueeze(1)], dim=1) # dim:dim+1 = .reshape(-1, 1) = .unsqueeze(1)

    # print("pos_out.shape should be (H*W*num_samples, (include_input + 2*freq) * 3):", pos_out.shape)

    #return pos_out


    


    # N = pos_in.shape[0]
    # pos_out = []
    #
    # if include_input:
    #     pos_out.append(pos_in)
    #
    # frequencies = [2**i for i in range(max_freq_power+1)]
    # 
    # # Apply sine functions for each frequency
    # for f in frequencies:
    #     for dim in range(3):  # Apply encoding for each dimension (x, y, z)
    #         pos_out.append(torch.sin(f * np.pi * pos_in[:, dim:dim+1]))
    # 
    # # Concatenate along the feature dimension
    # pos_out = torch.cat(pos_out, axis=1)
    # 
    # print("pos_out.shape should be (H*W*num_smaples, (include_input + 2*freq) * 3):", pos_out.shape)
    # # print("H: ", H, " W: ", W, " num_samples: ", num_samples, " pos_in.shape: ", pos_in.shape, " freqeuncies: ", frequencies)
    # return pos_out


    ################### YOUR CODE END ###################


def volume_rendering(
    radiance_field: torch.Tensor,
    ray_origins: torch.Tensor,
    depth_values: torch.Tensor
) -> Tuple[torch.Tensor]:
    r"""Differentiably renders a radiance field, given the origin of each ray in the
    bundle, and the sampled depth values along them.

    Args:
      radiance_field: at each query location (X, Y, Z), our model predict 
        RGB color and a volume density (sigma), shape (H, W, num_samples, 4)
      ray_origins: origin of each ray, shape (H, W, 3)
      depth_values: sampled depth values along each ray, shape (H, W, num_samples)
    
    Expected Returns:
      rgb_map: rendered RGB image, shape (H, W, 3)
    """
    ################### YOUR CODE START ###################

    H, W, Nsamples, _ = radiance_field.shape

    rgb_map = torch.zeros(H, W, 3, dtype=torch.float32)

    # Calculate deltas via staggered subtraction
    deltas = depth_values[..., 1:] - depth_values[..., :-1]
    # print("deltas.shape", deltas.shape, "deltas[..., -1].unsqueeze(2).shape", deltas[..., -1].unsqueeze(2).shape)
    deltas = torch.cat([deltas, deltas[..., -1].unsqueeze(2)], dim=-1)

    densities = radiance_field[..., 3]
    colors = radiance_field[..., 0:3]

    opacities = 1 - torch.exp(-densities * deltas)

    # Compute transmittance using the cumulative product of (1-opacity)
    transmittance = torch.cat([torch.ones_like(opacities[:, :, :1]), 1 - opacities], dim=-1)
    transmittance = torch.cumprod(transmittance, dim=-1)[:, :, :-1]
    
    # Weighted sum of colors along each ray
    weighted_colors = colors * opacities.unsqueeze(-1) * transmittance.unsqueeze(-1)
    
    # Integrate along the ray
    rgb_map = torch.sum(weighted_colors, dim=-2)

    # print("rgb_map.shape should be (H, W, 3):", rgb_map.shape)
    # print("rgb_map.type():", rgb_map.type())
    return rgb_map

    # # For each ray
    # for y in range(H):
    #     for x in range(W):
    #         densities = radiance_field[y, x, :, 0]
    #         colors = radiance_field[y, x, :, 1:4]
    #         ray_color = torch.zeros(3)
    #         for k in range(Nsamples-1):
    #             sample_color = colors[k]
    #             sample_density = densities[k]
    #             step_size = depth_values[y, x, k+1] - depth_values[y, x, k]
    #             sample_opacity = 1 - torch.exp(-sample_density * step_size)
    #
    #             transmittance = 1
    #             for j in range(k-1):
    #                 sample_opacity_j = 1 - torch.exp(-densities[j] * (depth_values[y, x, j+1] - depth_values[y, x, j]))
    #                 transmittance *= 1 - sample_opacity_j
    #
    #             ray_color += sample_color * sample_opacity * transmittance
    #                 
    #         rgb_map[y, x] = ray_color / Nsamples

    ################### YOUR CODE END ###################


class TinyNeRF(torch.nn.Module):
    def __init__(self, pos_dim, fc_dim=128):
      r"""Initialize a tiny nerf network, which composed of linear layers and
      ReLU activation. More specifically: linear - relu - linear - relu - linear
      - relu -linear. The module is intentionally made small so that we could 
      achieve reasonable training time

      Args:
        pos_dim: dimension of the positional encoding output
        fc_dim: dimension of the fully connected layer
      """
      super().__init__()

      self.nerf = nn.Sequential(
                    nn.Linear(pos_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, fc_dim),
                    nn.ReLU(),
                    nn.Linear(fc_dim, 4)
                  )
    
    def forward(self, x):
      r"""Output volume density and RGB color (4 dimensions), given a set of 
      positional encoded points sampled from the rays
      """
      x = self.nerf(x)
      return x


def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def nerf_step_forward(height, width, focal_length, trans_matrix,
                            near_point, far_point, num_depth_samples_per_ray,
                            get_minibatches_function, model):
    r"""Perform one iteration of training, which take information of one of the
    training images, and try to predict its rgb values

    Args:
      height: height of the image
      width: width of the image
      focal_length: focal length of the camera
      trans_matrix: transformation matrix, which is also the camera pose
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      get_minibatches_function: function to cut the ray bundles into several chunks
        to avoid out-of-memory issue

    Expected Returns:
      rgb_predicted: predicted rgb values of the training image
    """
    ################### YOUR CODE START ###################
    # TODO: Get the "bundle" of rays through all image pixels
    # imgs, poses, cam_params = load_colmap_data()
    print("get_rays")
    ray_origins, ray_directions = get_rays(height, width, focal_length, trans_matrix) 
    # TODO: Sample points along each ray
    print("sample_points_from_rays")
    sampled_points, depth_values = sample_points_from_rays(ray_origins, ray_directions, near_point, far_point, num_depth_samples_per_ray)
    # TODO: positional encoding, shape of return [H*W*num_samples, (include_input + 2*freq) * 3]
    print("positional_encoding")
    positional_encoded_points = positional_encoding(sampled_points.reshape(-1, 3), max_freq_power=10)
    # print("positional_encoded_points.shape should be (H*W*num_samples, (include_input + 2*freq) * 3):", positional_encoded_points.shape)
    ################### YOUR CODE END ###################

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    print("doing minibatches stuff")
    batches = get_minibatches_function(positional_encoded_points, chunksize=16384)
    predictions = []
    for batch in batches:
      predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0) # (H*W*num_samples, 4)

    # "Unflatten" the radiance field.
    unflattened_shape = [height, width, num_depth_samples_per_ray, 4] # (H, W, num_samples, 4)
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape) # (H, W, num_samples, 4)

    ################### YOUR CODE START ###################
    # TODO: Perform differentiable volume rendering to re-synthesize the RGB image. # (H, W, 3)
    # print("radiance_field_flattened.view")
    # radiance_field = radiance_field_flattened.view(height, width, num_depth_samples_per_ray, 4)
    
    print("volume rendering")
    rgb_predicted = volume_rendering(radiance_field, ray_origins, depth_values)

    print("nerf_step_forward end")
    return rgb_predicted
    
    ################### YOUR CODE END ###################


def train(images, poses, hwf, near_point, 
          far_point, num_depth_samples_per_ray,
          num_iters, model, DEVICE="cuda"):
    r"""Training a tiny nerf model

    Args:
      images: all the images extracted from dataset (including train, val, test)
      poses: poses of the camera, which are used as transformation matrix
      hwf: [height, width, focal_length]
      near_point: threshhold of nearest point
      far_point: threshold of farthest point
      num_depth_samples_per_ray: number of sampled depth from each rays in the ray bundle
      num_iters: number of training iterations
      model: predefined tiny NeRF model
    """
    H, W, focal_length = hwf
    H = int(H)
    W = int(W)
    n_train = images.shape[0]

    # Optimizer parameters
    lr = 5e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Seed RNG, for repeatability
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)
    plt.figure()

    for i in tqdm(range(num_iters)):
      print("i", i)
      # Randomly pick a training image as the target, get rgb value and camera pose
      train_idx = np.random.randint(n_train)
      train_img_rgb = images[train_idx, ..., :3]
      train_pose = poses[train_idx]

      # Run one iteration of TinyNeRF and get the rendered RGB image.
      print("train: nerf_step_forward start")
      rgb_predicted = nerf_step_forward(H, W, focal_length,
                                              train_pose, near_point,
                                              far_point, num_depth_samples_per_ray,
                                              get_minibatches, model)
      print("train: nerf_step_forward end")

      if i % 50 == 0:
        plt.imshow(rgb_predicted.detach().numpy())
        plt.show()

    
      # Compute mean-squared error between the predicted and target images
      loss = torch.nn.functional.mse_loss(rgb_predicted, train_img_rgb)
      print("loss", loss)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    print('Finish training')


if __name__ == "__main__":
    # TODO
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print("device: ", device)
    #torch.set_default_device('cuda')
    # Load the data from transform.json
    print("load_colmap_data start")
    imgs, poses, cam_params = load_colmap_data()
    print("load_colmap_data end")

    # Set camera parameters
    H = cam_params[0]
    W = cam_params[1]
    focal_length = cam_params[2]

    # Set near and far clip planes
    # near_point = 0.1
    near_point = 2.0 
    far_point = 5.0 
        
    # Set number of depth samples per ray
    num_depth_samples_per_ray = 32 

    # Initialize model
    print("Initialize model start")
    model = TinyNeRF(pos_dim=69, fc_dim=32)
    #model = model.to(device)
    print("Initialize model end")
    #
    # Train the model
    print("Train the model start")
    print("imgs.type(): ", imgs.type(), "poses.type(): ", poses.type(), "type(cam_params): ", type(cam_params))
    train(imgs, poses, cam_params, near_point, far_point, num_depth_samples_per_ray, 1000, model)
    print("Train the model end")
    #
    # # Save the model 
    torch.save(model.state_dict(), '/home/griffin/Documents/ese6500-hw3/gaddison_hw3_problem3/model.pt')

    # Load the model
    # model = TinyNeRF(pos_dim=69, fc_dim=32)
    model.load_state_dict(torch.load('/home/griffin/Documents/ese6500-hw3/gaddison_hw3_problem3/model.pt'))

    # Test the model
    test_idx = 80       
    test_img_rgb = imgs[test_idx, ..., :3]
    test_pose = poses[test_idx]
    print("nerf_step_forward start")
    rgb_predicted = nerf_step_forward(H, W, focal_length, test_pose, near_point, far_point, num_depth_samples_per_ray, get_minibatches, model)
    print("nerf_step_forward end")
    print("rgb_predicted.shape", rgb_predicted.shape)
    print("rgb_predicted[:5, :5]", rgb_predicted[:5, :5])
    plt.figure(1)
    print("torch.max", torch.max(rgb_predicted))
    plt.imshow(rgb_predicted.detach())
    # plt.show()
    plt.figure(2)   
    plt.imshow(test_img_rgb)
    plt.show()
    print('done')
