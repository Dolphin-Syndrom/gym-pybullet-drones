import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class HeavyDSLPIDControl(DSLPIDControl):
    """
    PID control class specifically tuned for the Heavy Tactical Drone (8kg).
    Inherits from DSLPIDControl.
    """

    def __init__(self,
                 drone_model: DroneModel=DroneModel.HEAVY,
                 g: float=9.8
                 ):
        
        super().__init__(drone_model=drone_model, g=g)
        
        # ----------------------------------------------------------------------
        # 1. Override PWM -> RPM Constants
        # ----------------------------------------------------------------------
        # The heavy drone simulation expects RPMs directly.
        # We bypass the Crazyflie-specific PWM mapping by setting 1:1 mapping.
        self.PWM2RPM_SCALE = 1.0
        self.PWM2RPM_CONST = 0.0
        self.MIN_PWM = 0
        self.MAX_PWM = 10000 # Arbitrary high limit, actual clipping happens in Env

        # ----------------------------------------------------------------------
        # 2. Retune Position Gains (Force)
        # ----------------------------------------------------------------------
        # User requested P=0.4, D=0.8. 
        # Assuming these are normalized acceleration gains (m/s^2 / m),
        # we scale them by Mass (8.0kg) to get Force gains (N / m).
        # P_FOR: 0.4 * 8.0 = 3.2
        # D_FOR: 0.8 * 8.0 = 6.4
        # Z-axis usually needs higher P gain. Default was 1.25 (normalized ~46).
        # We'll scale Z similarly: 1.25 * 8 = 10.0
        
        self.P_COEFF_FOR = np.array([3.2, 3.2, 10.0])
        self.I_COEFF_FOR = np.array([0.1, 0.1, 0.1]) # Small integral gain
        self.D_COEFF_FOR = np.array([6.4, 6.4, 4.0])

        # ----------------------------------------------------------------------
        # 3. Retune Attitude Gains (Torque -> RPM)
        # ----------------------------------------------------------------------
        # Since we output RPM directly (via PWM2RPM_SCALE=1), these gains 
        # map Angle Error (rad) -> RPM difference.
        # Inertia is high, so we need significant RPM differential.
        # P=30000 means 0.1 rad error -> 3000 RPM difference.
        
        self.P_COEFF_TOR = np.array([30000., 30000., 30000.])
        self.I_COEFF_TOR = np.array([100., 100., 100.])
        self.D_COEFF_TOR = np.array([10000., 10000., 10000.])
        
        # Mixer Matrix for X-Configuration (Same as CF2X but can be customized)
        # Maps [Thrust, Roll, Pitch, Yaw] -> [M1, M2, M3, M4]
        self.MIXER_MATRIX = np.array([ 
                                [-.5, -.5, -1],
                                [-.5,  .5,  1],
                                [.5,  .5, -1],
                                [.5, -.5,  1]
                                ])
