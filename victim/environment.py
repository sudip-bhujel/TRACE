import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


class AI2THORNavEnv:
    """
    Gym-like wrapper around AI2-THOR for point navigation tasks.

    Observations: RGB image (C, H, W) resized to (3, 84, 84)
    Actions: 0=MoveAhead, 1=RotateLeft(90), 2=RotateRight(90), 3=LookDown(30), 4=LookUp(30)
    Reward: -0.01 per step, +1.0 if agent reaches target position
    """

    ACTIONS = [
        {"action": "MoveAhead"},
        # {"action": "RotateLeft", "degrees": 90},
        # {"action": "RotateRight", "degrees": 90},
        # {"action": "LookDown", "degrees": 30},
        # {"action": "LookUp", "degrees": 30},
        {"action": "RotateLeft", "degrees": 15},
        {"action": "RotateRight", "degrees": 15},
        {"action": "LookDown", "degrees": 15},
        {"action": "LookUp", "degrees": 15},
    ]

    def __init__(
        self,
        scene: str = "FloorPlan1",
        image_size: Tuple[int, int] = (84, 84),
        max_steps: int = 200,
        headless: bool = True,
        grid_size: float = 0.25,
    ):
        self.scene = scene
        self.image_size = image_size
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.step_count = 0

        self.controller = Controller(
            scene=scene,
            gridSize=grid_size,
            # headless=headless,
            platform=CloudRendering if headless else None,
        )

        # Try to initialize
        try:
            self.controller.step({"action": "Initialize", "gridSize": grid_size})
        except Exception:
            pass

        # Action and observation spaces
        self.action_space_n = len(self.ACTIONS)
        self.observation_shape = (3, image_size[0], image_size[1])

        self.last_event = None
        self.target_position = None

    def _get_observation(self) -> np.ndarray:
        """Get RGB observation as (C, H, W) uint8 array."""
        frame = self.last_event.frame  # (H, W, 3)
        img = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
        img = np.transpose(img, (2, 0, 1)).astype(np.uint8)  # (C, H, W)
        return img

    def _get_agent_position(self) -> Dict[str, float]:
        """Get agent position."""
        return self.last_event.metadata["agent"]["position"]

    def _get_reachable_positions(self) -> List[Dict[str, float]]:
        """Get all reachable positions."""
        event = self.controller.step({"action": "GetReachablePositions"})
        return event.metadata.get("actionReturn", [])

    def _distance(self, p1: Dict, p2: Dict) -> float:
        """Euclidean distance between positions."""
        return np.sqrt(
            (p1["x"] - p2["x"]) ** 2
            + (p1["y"] - p2["y"]) ** 2
            + (p1["z"] - p2["z"]) ** 2
        )

    def set_scene(self, scene: str, max_retries: int = 3):
        """Change the current scene with retry logic for timeouts."""
        if scene != self.scene:
            self.scene = scene
            for attempt in range(max_retries):
                try:
                    self.controller.reset(scene)
                    return
                except (TimeoutError, RuntimeError) as e:
                    if attempt < max_retries - 1:
                        print(
                            f"Scene switch timeout, retrying ({attempt + 1}/{max_retries})..."
                        )
                        # Try to restart controller
                        self._restart_controller()
                    else:
                        raise e

    def reset(self, scene: str = None) -> np.ndarray:
        """Reset environment and return initial observation.

        Args:
            scene: Optional scene to switch to before reset.
        """
        if scene is not None:
            self.set_scene(scene)

        self.step_count = 0
        self.controller.reset(self.scene)

        # Try to initialize
        try:
            self.controller.step({"action": "Initialize", "gridSize": self.grid_size})
        except Exception:
            pass

        # Get reachable positions
        reachable = self._get_reachable_positions()

        # Teleport to random position with random rotation
        if reachable:
            pos = random.choice(reachable)
            rot = random.choice([0, 90, 180, 270])
            try:
                self.controller.step(
                    {
                        "action": "TeleportFull",
                        "x": pos["x"],
                        "y": pos["y"],
                        "z": pos["z"],
                        "rotation": {"x": 0, "y": rot, "z": 0},
                        "horizon": 0.0,
                    }
                )
            except Exception:
                pass

        # Get first frame
        self.last_event = self.controller.step({"action": "Pass"})

        # Set random target position (different from agent)
        if reachable:
            agent_pos = self._get_agent_position()
            valid_targets = [p for p in reachable if self._distance(p, agent_pos) > 1.0]
            if valid_targets:
                self.target_position = random.choice(valid_targets)
            else:
                self.target_position = random.choice(reachable)
        else:
            self.target_position = self._get_agent_position()

        return self._get_observation()

    def _restart_controller(self):
        """Restart the AI2-THOR controller."""
        print(f"Restarting AI2-THOR controller for scene {self.scene}...")
        try:
            self.controller.stop()
        except Exception:
            pass

        headless = False
        if hasattr(self.controller, "headless"):
            headless = self.controller.headless

        import time

        time.sleep(1.0)  # Give it a moment to close ports

        self.controller = Controller(
            scene=self.scene,
            gridSize=self.grid_size,
            # headless=headless,
            platform=CloudRendering if headless else None,
        )
        print("Controller restarted.")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take action and return (obs, reward, done, info)."""
        self.step_count += 1

        # Execute action
        try:
            self.last_event = self.controller.step(self.ACTIONS[action])
        except Exception as e:
            print(f"Controller error in step: {e}")
            self._restart_controller()
            # Retry step once
            try:
                self.last_event = self.controller.step(self.ACTIONS[action])
            except Exception as e2:
                print(f"Controller error in retry step: {e2}")
                # If retry fails, just pass to avoid crashing if possible, or re-raise
                self.last_event = self.controller.step({"action": "Pass"})

        obs = self._get_observation()

        # Compute reward
        agent_pos = self._get_agent_position()
        dist = self._distance(agent_pos, self.target_position)

        reward = -0.01  # Step penalty
        done = False

        # Success if close to target
        if dist < 1.0:
            reward += 1.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        info = {"distance": dist}
        return obs, reward, done, info

    def close(self):
        """Close the environment."""
        try:
            self.controller.stop()
        except Exception:
            pass
