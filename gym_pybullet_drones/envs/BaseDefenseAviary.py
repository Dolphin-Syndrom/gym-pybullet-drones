import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.drone_constants import HEAVY_DRONE_MASS, HEAVY_DRONE_KF, HEAVY_DRONE_KM, HEAVY_DRONE_HOVER_RPM

class BaseDefenseAviary(CtrlAviary):
    """
    Base class for the Drone Swarm Defense environment.
    Inherits from CtrlAviary to leverage physics control.
    """

    def __init__(self,
                 drone_model: DroneModel=DroneModel.HEAVY,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results'
                 ):
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )

        # ----------------------------------------------------------------------
        # 1. Override Constants for Heavy Drone Physics
        # ----------------------------------------------------------------------
        if self.DRONE_MODEL == DroneModel.HEAVY:
            self.M = HEAVY_DRONE_MASS
            self.KF = HEAVY_DRONE_KF
            self.KM = HEAVY_DRONE_KM
            
            # Recalculate derived constants
            self.GRAVITY = self.G * self.M
            self.HOVER_RPM = HEAVY_DRONE_HOVER_RPM
            self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
            self.MAX_THRUST = (4 * self.KF * self.MAX_RPM**2)
            
            # Recalculate Torque limits based on new Mass/KF/KM and Arm Length (self.L is loaded from URDF)
            # Assuming X configuration for Heavy Drone as per URDF
            self.MAX_XY_TORQUE = (2 * self.L * self.KF * self.MAX_RPM**2) / np.sqrt(2)
            self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM**2)

        # ----------------------------------------------------------------------
        # 2. Environment Specific Parameters
        # ----------------------------------------------------------------------
        self.MAX_ENDURANCE_MINUTES = 90
        self.MAX_RANGE_KM = 40
        self.MAX_SPEED_MS = 10.0
        self.MIN_ALTITUDE = 60.0
        self.MAX_ALTITUDE = 120.0

        # Initialize Endurance (Battery)
        # Total steps = 90 mins * 60 sec/min * ctrl_freq steps/sec
        self.battery_steps = int(self.MAX_ENDURANCE_MINUTES * 60 * self.CTRL_FREQ)
        self.current_battery_steps = np.full(self.NUM_DRONES, self.battery_steps)

        # Initialize Range Tracking
        self.cumulative_distance = np.zeros(self.NUM_DRONES)
        self.prev_pos = np.copy(initial_xyzs) if initial_xyzs is not None else np.zeros((self.NUM_DRONES, 3))

        # ----------------------------------------------------------------------
        # 3. Threat & Base Parameters
        # ----------------------------------------------------------------------
        self.BASE_POS = np.array([0, 0, 0])
        self.THREAT_SPEED = 5.0 # m/s
        self.THREAT_ID = None
        self.threat_pos = np.array([0, 0, 200]) # Placeholder
        self.threat_vel = np.zeros(3)

    def _observationSpace(self):
        """
        Returns the observation space of the environment.
        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES, 12) containing:
            - Self State (6): Pos (3), Vel (3)
            - Threat State (3): Relative Pos
            - Base State (3): Relative Pos
        """
        # 12 features per drone
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DRONES, 12), dtype=np.float32)

    def _computeObs(self):
        """
        Returns the current observation of the environment.
        """
        obs = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            vel = state[10:13]
            
            # Normalization Factor (Max Range = 40km = 40000m)
            # Using 40000 as normalization constant for positions
            NORM_DIST = self.MAX_RANGE_KM * 1000

            # 1. Self State (Normalized)
            obs[i, 0:3] = pos / NORM_DIST
            obs[i, 3:6] = vel / self.MAX_SPEED_MS # Normalize velocity by max speed

            # 2. Threat State (Relative Position)
            # Vector from Drone to Threat
            if self.THREAT_ID is not None:
                rel_threat = self.threat_pos - pos
                obs[i, 6:9] = rel_threat / NORM_DIST
            else:
                obs[i, 6:9] = np.zeros(3)

            # 3. Base State (Relative Position)
            # Vector from Drone to Base
            rel_base = self.BASE_POS - pos
            obs[i, 9:12] = rel_base / NORM_DIST
            
        return obs

    def reset(self, seed=None, options=None):
        """
        Resets the environment.
        """
        # 1. Reset PyBullet and Drones
        obs, info = super().reset(seed=seed, options=options)
        
        # 2. Spawn Threat
        # Remove old threat if exists (PyBullet reset clears it, but good to be safe in logic)
        # p.resetSimulation() in super().reset() clears everything.
        
        # Load Threat Visual (Red Sphere)
        # Using sphere2.urdf from assets
        import pkg_resources
        sphere_path = pkg_resources.resource_filename('gym_pybullet_drones', 'assets/sphere2.urdf')
        self.THREAT_ID = p.loadURDF(sphere_path, [0,0,200], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.CLIENT)
        p.changeVisualShape(self.THREAT_ID, -1, rgbaColor=[1, 0, 0, 1], physicsClientId=self.CLIENT) # Red color

        # 3. Position Threat 200m away
        # Random position on a hemisphere (z > 0) or just 3D sphere?
        # "Incoming aerial threats" -> Z > 0
        # "200m away from Base"
        
        # Random spherical coordinates
        r = 200.0
        theta = np.random.uniform(0, 2*np.pi) # Azimuth
        phi = np.random.uniform(0, np.pi/2)   # Elevation (0 to 90 deg) - Upper Hemisphere
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # Ensure min altitude for threat? 
        # If z is too low, it might hit ground immediately. 
        # Let's enforce z >= 50m
        z = max(z, 50.0)
        
        self.threat_pos = np.array([x, y, z])
        
        # 4. Set Threat Velocity (Towards Base)
        # Vector from Threat to Base (Base is at 0,0,0)
        direction = self.BASE_POS - self.threat_pos
        direction_norm = direction / np.linalg.norm(direction)
        self.threat_vel = direction_norm * self.THREAT_SPEED
        
        # Move Threat to Start Position
        p.resetBasePositionAndOrientation(self.THREAT_ID, self.threat_pos, [0,0,0,1], physicsClientId=self.CLIENT)
        
        # 5. Reset Environment State Variables
        self.current_battery_steps = np.full(self.NUM_DRONES, self.battery_steps)
        self.cumulative_distance = np.zeros(self.NUM_DRONES)
        self.prev_pos = np.copy(self.INIT_XYZS)
        
        # Initialize previous distance to threat for shaping reward
        self.prev_dist_to_threat = np.zeros(self.NUM_DRONES)
        for i in range(self.NUM_DRONES):
            # Use initial position from INIT_XYZS
            pos = self.INIT_XYZS[i]
            self.prev_dist_to_threat[i] = np.linalg.norm(self.threat_pos - pos)

        # 6. Recompute Observation with new Threat
        return self._computeObs(), info

    def step(self, action):
        """
        Advances the environment by one simulation step.
        """
        # ----------------------------------------------------------------------
        # 1. Update Threat Position
        # ----------------------------------------------------------------------
        if self.THREAT_ID is not None:
            # Move threat
            self.threat_pos += self.threat_vel * self.CTRL_TIMESTEP
            
            # Update Visuals
            p.resetBasePositionAndOrientation(self.THREAT_ID, self.threat_pos, [0,0,0,1], physicsClientId=self.CLIENT)
            
            # Check if Threat hit Base (Simple check)
            dist_to_base = np.linalg.norm(self.threat_pos - self.BASE_POS)
            if dist_to_base < 1.0: # Hit radius
                # Logic for base hit (handled in reward usually, but here we just move it)
                pass

        # ----------------------------------------------------------------------
        # 2. Speed Cap (Hard Clip)
        # ----------------------------------------------------------------------
        # Check and clamp velocity before physics update
        for i in range(self.NUM_DRONES):
            vel = self._getDroneStateVector(i)[10:13]
            speed = np.linalg.norm(vel)
            if speed > self.MAX_SPEED_MS:
                new_vel = (vel / speed) * self.MAX_SPEED_MS
                p.resetBaseVelocity(self.DRONE_IDS[i], linearVelocity=new_vel, physicsClientId=self.CLIENT)

        # ----------------------------------------------------------------------
        # 2. Step Simulation (Parent Method)
        # ----------------------------------------------------------------------
        obs, reward, terminated, truncated, info = super().step(action)

        # ----------------------------------------------------------------------
        # 3. Update Endurance & Range
        # ----------------------------------------------------------------------
        self.current_battery_steps -= 1
        
        current_pos = np.zeros((self.NUM_DRONES, 3))
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            current_pos[i] = pos
            
            # Update cumulative distance
            dist = np.linalg.norm(pos - self.prev_pos[i])
            self.cumulative_distance[i] += dist
            self.prev_pos[i] = pos

        # ----------------------------------------------------------------------
        # 4. Check Termination Conditions
        # ----------------------------------------------------------------------
        if np.any(self.current_battery_steps <= 0):
            terminated = True
            info["termination_reason"] = "endurance_depleted"
        
        if np.any(self.cumulative_distance > self.MAX_RANGE_KM * 1000):
            terminated = True
            info["termination_reason"] = "range_exceeded"

        return obs, reward, terminated, truncated, info

    def _computeReward(self):
        """
        Computes the current reward value.
        Returns a dictionary {agent_id: reward_value}.
        """
        rewards = {}
        
        # Parameters
        INTERCEPT_RADIUS = 5.0
        BASE_SAFETY_RADIUS = 10.0
        
        # Check Threat Status
        threat_dist_to_base = np.linalg.norm(self.threat_pos - self.BASE_POS)
        base_hit = threat_dist_to_base < BASE_SAFETY_RADIUS
        
        for i in range(self.NUM_DRONES):
            reward = 0.0
            
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            z = state[2]
            
            # 1. Team Reward (Win/Loss)
            dist_to_threat = np.linalg.norm(self.threat_pos - pos)
            
            if dist_to_threat < INTERCEPT_RADIUS:
                reward += 100.0 # Threat Neutralized
            
            if base_hit:
                reward -= 500.0 # Mission Failed
                
            # 2. Shaping Reward (Potential-Based)
            # (prev_dist - curr_dist) * 10
            # Positive if getting closer, Negative if moving away
            shaping = (self.prev_dist_to_threat[i] - dist_to_threat) * 10.0
            reward += shaping
            
            # Update previous distance for next step
            self.prev_dist_to_threat[i] = dist_to_threat
            
            # 3. Penalties
            # Crash (Ground)
            if z < 0.1:
                reward -= 100.0
            
            # Too High
            if z > 120.0:
                reward -= 10.0
                
            # Energy (Constant step penalty)
            reward -= 0.05
            
            rewards[i] = reward

        return rewards

    def _computeInfo(self):
        """
        Computes the current info dict(s).
        """
        # Default info from BaseAviary is usually empty or basic
        info = {}
        
        # Calculate endurance remaining percentage
        # We return a list or a single value? Gym expects info to be a dict.
        # If Multi-Agent, it might be complex, but standard Gym API returns one info dict.
        # We'll store the average or list.
        info['endurance_remaining'] = (self.current_battery_steps / self.battery_steps) * 100
        info['cumulative_distance'] = self.cumulative_distance
        
        return info
