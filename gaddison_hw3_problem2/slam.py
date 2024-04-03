# Pratik Chaudhari (pratikac@seas.upenn.edu)

import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

import time

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        
        # # clip to within map bounds
        # x = np.clip(x, s.xmin, s.xmax)
        # y = np.clip(y, s.ymin, s.ymax)
        #
        # # convert to cell indices
        # x = np.floor((x - s.xmin) / s.resolution).astype(int) # maybe round instead of floor
        # y = np.floor((y - s.ymin) / s.resolution).astype(int)
        #
        # return np.array([x, y])
        rows = np.clip((x - s.xmin)// s.resolution, 0, s.szx-1)
        cols = np.clip((y - s.ymin) // s.resolution, 0, s.szy-1)
        
        return np.vstack((rows,cols))


class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        # s.Q = 1e-8*np.eye(3)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX

        # # compute cumulative sum of weights
        # cumsum = np.cumsum(w)
        #
        # # create n equally spaced samples
        # samples = np.linspace(0, 1, len(w), endpoint=False) + np.random.uniform(0, 1/len(w))
        #
        # # find the indices of the particles that are closest to the samples
        # new_indices = np.zeros(len(w), dtype=int)
        # j = 0
        #
        # for i in range(len(w)):
        #     while samples[i] > cumsum[j]:
        #         j += 1
        #         indices[i] = j
        #         
        # # create new particles and weights
        # new_particles = p[:, new_indices]
        # new_weights = np.ones(len(w)) / len(w)
        #
        # return new_particles, new_weights
        
        n = len(w)
        r = np.random.uniform(0, 1/n)
        i = 0
        c = w[0]
        new_particles = np.zeros((3, n))
        
        for m in range(n):
            u = r + m/n
            while u > c:
                i = i + 1
                c =  c + w[i]
            new_particles[:,m] = p[:,i]
       
        new_weights = np.ones(n) / n
        return new_particles, new_weights 


    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """
        #### TODO: XXXXXXXXXXX

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data
        # d = np.clip(d, s.lidar_dmin, s.lidar_dmax)
        d = d[np.where((d >= s.lidar_dmin) & (d <= s.lidar_dmax))]
        angles = angles[np.where((d >= s.lidar_dmin) & (d <= s.lidar_dmax))]
        # print("d.shape", d.shape)
        # 1. from lidar distances to points in the LiDAR frame
        P_lidar = np.vstack((d * np.cos(angles), \
                             d * np.sin(angles), \
                             np.zeros(len(d)), \
                             np.ones(len(d))))

        # 2. from LiDAR frame to the body frame
        body_T_lidar = euler_to_se3(0, head_angle, neck_angle, np.array([0, 0, s.lidar_height])) # order of frames might be wrong naming-wise
        P_body = body_T_lidar @ P_lidar

        # 3. from body frame to world frame
        x, y, yaw = p
        # print("p: ", p)
        # print("x, y, yaw: ", x, y, yaw)
        world_T_body = euler_to_se3(0, 0, yaw, np.array([x, y, s.head_height]))
        P_world = world_T_body @ P_body

        return P_world


    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        #### TODO: XXXXXXXXXXX

        # TODO: check that i am using the correct data
        control = smart_minus_2d(s.lidar[t]['xyth'], s.lidar[t-1]['xyth'])

        return control


    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### TODO: XXXXXXXXXXX

        # Overview: dynamics is just the control, plus noise
      
        # for each particle
        control = s.get_control(t)
        for i in range(s.n):
            # create noise (normal, mean 0, var Q, for xyz for each particle)
            mean = np.zeros(3)
            noise = np.random.multivariate_normal(mean, s.Q)
            # print("noise.shape: ", noise.shape)
            # print("control.shape: ", control.shape)
            # compose control and noise
            # dynamics = control + noise
            #
            # # transform each particle with dynamics
            # # print("particles before dynamics: ", s.p[:, i])
            # s.p[:, i] = smart_plus_2d(s.p[:, i], dynamics)
            # print("particles after dynamics: ", s.p[:, i])

            s.p[:, i] = smart_plus_2d(s.p[:, i], control)
            s.p[:, i] = smart_plus_2d(s.p[:, i], noise)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        # print("obs_logp: ", obs_logp)
        # w = np.exp(obs_logp) * w
        #
        # # normalize weights
        # w = w / np.sum(w)
        #
        # return w
        return np.exp(np.log(w) + obs_logp - slam_t.log_sum_exp(np.log(w) + obs_logp))

    def bresenham_line(s, start, end):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end

        :param start: (x0, y0) starting position
        :param end: (x1, y1) ending position
        :returns: List of tuples where each tuple is a cell coordinate on the line
        """
        # Setup initial conditions
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0

        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)

        # Rotate line
        if is_steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        # Swap start and end points if necessary and store swap state
        swapped = False
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            swapped = True

        # Recalculate differentials
        dx = x1 - x0
        dy = y1 - y0

        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y0 < y1 else -1

        # Iterate over bounding box generating points between start and end
        y = y0
        points = []
        for x in range(x0, x1 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return points

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """
        #### TODO: XXXXXXXXXXX
        # obs_start_time = time.time() 
        # 1. a. Find head, neck angle at t
        # print("s.joint['head_angles'].shape", s.joint['head_angles'].shape)
        t_idx = s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])
        # neck, head = s.joint['head_angles'][:, t_idx]
        neck = s.joint['head_angles'][0][t_idx]
        head = s.joint['head_angles'][1][t_idx]

        # 1. b. Project lidar scan into world frame
        scan = s.lidar[t]['scan'] # I assume this is the depths of the lidar scans
        obs_cells = np.zeros((s.n, s.map.szx, s.map.szy))
        obs_logodds = np.zeros(s.n)
        # for each particle
        # print("observation particles loop: start")
        # obs_loop_start_time = time.time()
        # print("s.n: ", s.n)
        occ_grid_per_particle = np.zeros((s.n, s.map.szx, s.map.szy))
        pose = s.p.T
        for i in range(s.n):

            ## For one particle

            # convert lidar points to world frame
            # loop_start_time = time.time()
            lidar_hits_W = s.rays2world(pose[i].T, scan, head, neck, s.lidar_angles)
            # lidar_hits_W_time = time.time()
            lidar_hits_cellIdx = s.map.grid_cell_from_xy(lidar_hits_W[0], lidar_hits_W[1]).astype(int)
            # lidar_hits_cellIdx_time = time.time()

            occ_grid_per_particle[i, lidar_hits_cellIdx[0], lidar_hits_cellIdx[1]] = 1
            # if i % 25 == 0:
            #     plt.figure()
            #     plt.imshow(occ_grid_per_particle[i], cmap='binary')
            #     plt.show()
            # 1. c. Calculate which cells are obstacles according to this particle
            # Every time we hit a cell already in the map, increase log odds of this particle
            obs_logodds[i] = s.map.cells[lidar_hits_cellIdx[0], lidar_hits_cellIdx[1]].sum() 
            # obs_logodds_time = time.time()
            # obs_cells[i, cell_indices_all[0], cell_indices_all[1]] = 1
            # for lidar_point in lidar_points:
            #     cell_indices = s.map.grid_cell_from_xy(lidar_point[0], lidar_point[1])
            #     obs_cells[i, cell_indices[0], cell_indices[1]] = 1
            #     # print("s.map.cells: ", s.map.cells)
            #     obs_logodds[i] += s.map.cells[cell_indices[0], cell_indices[1]]  
            # print("lidar_hits_w_time: ", lidar_hits_W_time - loop_start_time)
            # print("lidar_hits_cellIdx_time: ", lidar_hits_cellIdx_time - lidar_hits_W_time)
            # print("obs_logodds_time: ", obs_logodds_time - lidar_hits_cellIdx_time)
        # obs_loop_end_time = time.time()
        # print("observation particles loop: end")
        # 2. Update particle weights using observation log_probability
        # print("s.w before: ", s.w)
        s.w = s.update_weights(s.w, obs_logodds)
        # print("s.w after: ", s.w)

        # 3. Find particle with largest weight, use its occupied cells to update map.log_odds and map.cells
        # most_likely_particle = s.p[:, np.argmax(s.w)]
        # mlp_lidar_hits = s.rays2world(most_likely_particle.T, scan, head, neck, s.lidar_angles)
        # mlp_lidar_hits_cellIdx = s.map.grid_cell_from_xy(mlp_lidar_hits[0], mlp_lidar_hits[1]).astype(int)
        # s.map.log_odds += s.lidar_log_odds_free
        # s.map.log_odds[mlp_lidar_hits_cellIdx[0], mlp_lidar_hits_cellIdx[1]] -= s.lidar_log_odds_free 
        # s.map.log_odds[mlp_lidar_hits_cellIdx[0], mlp_lidar_hits_cellIdx[1]] += s.lidar_log_odds_occ 
        
        most_likely_occ_grid = occ_grid_per_particle[np.argmax(s.w)]
        # plt.figure()
        # plt.imshow(most_likely_occ_grid, cmap='binary')
        # plt.title("most likely occ grid")
        # plt.show()
           
        s.map.num_obs_per_cell += most_likely_occ_grid.astype(np.uint64)

        if t % 1000 == 0:
            plt.figure()
            plt.imshow(s.map.log_odds, cmap='binary')
            plt.title('log odds before')
            plt.show()
        s.map.log_odds += s.lidar_log_odds_occ * most_likely_occ_grid

        if t % 1000 == 0:
            plt.figure()
            plt.imshow(s.map.log_odds, cmap='binary')
            plt.title('log odds after occ')
            plt.show()

        # For each particle in path of hit, decrease log odds
        # for i in range(s.n):
        #     free_cells = s.bresenham_line((s.p[0, i].astype(int), s.p[1, i].astype(int)), (lidar_hits_cellIdx[0][i].astype(int), lidar_hits_cellIdx[1][i].astype(int)))
        #     s.map.log_odds[free_cells] += s.lidar_log_odds_free
        if t % 1000 == 0:
            plt.figure()
            plt.imshow(s.map.log_odds, cmap='binary')
            plt.title('log odds after free')
            plt.show()




# Now decide whether to update log odds for free space based on the observation count
# Only decrease log odds for cells that have been observed more than a certain number of times
        # obs_threshold = 5  # Set this based on your typical scan density
        # mask = (s.map.num_obs_per_cell >= obs_threshold) & (most_likely_occ_grid == 0)

        # s.map.log_odds += s.lidar_log_odds_free * (1 - most_likely_occ_grid)
        # s.map.log_odds[mask] -= s.lidar_log_odds_free

        # if t % 1000 == 0:
        #     plt.figure()
        #     plt.imshow(s.map.log_odds, cmap='binary')
        #     plt.title('log odds after free')
        #     plt.show()
        # s.map.log_odds += s.lidar_log_odds_occ * obs_cells[max_weight_idx] 
        # s.map.log_odds -= s.lidar_log_odds_free * (1- obs_cells[max_weight_idx])
        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max)
        # print("s.map.log_odds: ", s.map.log_odds)
        # print("s.map.log_odds_thresh: ", s.map.log_odds_thresh)
        # print("s.map.cells: ", s.map.cells)
        s.map.cells = np.zeros(s.map.cells.shape)
        # s.map.cells = np.where(s.map.log_odds > s.map.log_odds_thresh, 1, 0)
        s.map.cells[s.map.log_odds >= s.map.log_odds_thresh] = 1
        # plt.figure()
        # plt.imshow(s.map.cells, cmap='binary')
        # plt.title("map cells")
        # plt.show()

        # print("s.map.cells: ", s.map.cells)
        # obs_post_loop_end_time = time.time()
        # obs_end_time = time.time()


        # print("observation_step_time: ", obs_end_time - obs_start_time)
        # print("obs_loop_time: ", obs_loop_end_time - obs_loop_start_time)
        # print("obs_post_loop_time: ", obs_post_loop_end_time - obs_loop_end_time)
        # print("obs_resample_time: ", obs_end_time - obs_post_loop_end_time)
        # print("\n\n")
    

                    
    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
