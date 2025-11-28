"""Script to visualize the Base Defense scenario with a moving threat."""
import time
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseDefenseAviary import BaseDefenseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync

def run():
    """
    Run the simulation loop to visualize the threat moving towards the base.
    """
    # Initialize the environment
    env = BaseDefenseAviary(drone_model=DroneModel.HEAVY,
                            num_drones=1,
                            initial_xyzs=np.array([[0, 0, 1.0]]), # Drone starts near base
                            initial_rpys=np.array([[0, 0, 0]]),
                            physics=Physics.PYB,
                            gui=True,
                            record=False,
                            obstacles=False,
                            user_debug_gui=True
                            )

    # Reset the environment (spawns the threat)
    obs, info = env.reset()
    
    print("\n[INFO] Simulation started.")
    print(f"[INFO] Threat spawned at: {env.threat_pos}")
    print(f"[INFO] Threat velocity: {env.threat_vel}")
    print("[INFO] Watch the Red Sphere move towards the Base (0,0,0).")
    print("[INFO] Press 'q' in the GUI or Ctrl+C in terminal to exit.\n")

    START = time.time()
    action = np.zeros((1, 4)) # No action (drone will fall, but we care about the threat)

    # Run indefinitely until user exits
    i = 0
    while True:
        
        # Step the simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Sync the simulation
        sync(i, START, env.CTRL_TIMESTEP)
        
        # Optional: Print distance every second
        if i % env.CTRL_FREQ == 0:
            dist = np.linalg.norm(env.threat_pos - env.BASE_POS)
            print(f"Time: {i/env.CTRL_FREQ:.1f}s | Threat Dist: {dist:.2f}m")
            
        # Check for 'q' key press to exit
        keys = p.getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            print("[INFO] 'q' pressed. Exiting simulation.")
            break
            
        i += 1

    # Close the environment
    env.close()

if __name__ == "__main__":
    run()
