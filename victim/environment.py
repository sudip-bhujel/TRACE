import random
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


class AI2THORNavEnv:
    """
    Gym-like AI2-THOR point-navigation environment.

    Observations: RGB image (3, 84, 84), uint8.
    The default ``nav5`` action set preserves the original five action indices.
    ``nav8`` appends MoveBack, MoveLeft, and MoveRight.
    Reward: -0.01 per step, +1.0 if the agent reaches the target.
    """

    ACTION_SETS = {
        "nav5": [
            {"action": "MoveAhead"},
            {"action": "RotateLeft", "degrees": 15},
            {"action": "RotateRight", "degrees": 15},
            {"action": "LookDown", "degrees": 15},
            {"action": "LookUp", "degrees": 15},
        ],
        "nav8": [
            {"action": "MoveAhead"},
            {"action": "RotateLeft", "degrees": 15},
            {"action": "RotateRight", "degrees": 15},
            {"action": "LookDown", "degrees": 15},
            {"action": "LookUp", "degrees": 15},
            {"action": "MoveBack"},
            {"action": "MoveLeft"},
            {"action": "MoveRight"},
        ],
    }
    # Backward-compatible alias for code that reads the original class attribute.
    ACTIONS = ACTION_SETS["nav5"]

    def __init__(
        self,
        scene: str = "FloorPlan1",
        image_size: Tuple[int, int] = (84, 84),
        max_steps: int = 200,
        headless: bool = True,
        grid_size: float = 0.25,
        action_set: str = "nav5",
    ):
        if action_set not in self.ACTION_SETS:
            choices = ", ".join(sorted(self.ACTION_SETS))
            raise ValueError(f"Unknown action_set '{action_set}'. Choose: {choices}")

        self.scene = scene
        self.image_size = image_size
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.step_count = 0
        self.action_set = action_set
        self.actions = [dict(action) for action in self.ACTION_SETS[action_set]]
        self.action_names = [action["action"] for action in self.actions]

        self.controller = Controller(
            scene=scene,
            gridSize=grid_size,
            platform=CloudRendering if headless else None,
        )

        try:
            self.controller.step({"action": "Initialize", "gridSize": grid_size})
        except Exception:
            pass

        self.action_space_n = len(self.actions)
        self.observation_shape = (3, image_size[0], image_size[1])

        self.last_event = None
        self.target_position = None

    def _get_observation(self) -> np.ndarray:
        frame = self.last_event.frame
        img = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
        return np.transpose(img, (2, 0, 1)).astype(np.uint8)

    def _get_agent_position(self) -> Dict[str, float]:
        return self.last_event.metadata["agent"]["position"]

    def _get_reachable_positions(self) -> List[Dict[str, float]]:
        event = self.controller.step({"action": "GetReachablePositions"})
        return event.metadata.get("actionReturn", [])

    def _distance(self, p1: Dict, p2: Dict) -> float:
        return np.sqrt(
            (p1["x"] - p2["x"]) ** 2
            + (p1["y"] - p2["y"]) ** 2
            + (p1["z"] - p2["z"]) ** 2
        )

    def set_scene(self, scene: str, max_retries: int = 3):
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
                        self._restart_controller()
                    else:
                        raise e

    def reset(self, scene: str = None) -> np.ndarray:
        if scene is not None:
            self.set_scene(scene)

        self.step_count = 0
        self.controller.reset(self.scene)

        try:
            self.controller.step({"action": "Initialize", "gridSize": self.grid_size})
        except Exception:
            pass

        reachable = self._get_reachable_positions()

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

        self.last_event = self.controller.step({"action": "Pass"})

        if reachable:
            agent_pos = self._get_agent_position()
            valid_targets = [p for p in reachable if self._distance(p, agent_pos) > 1.0]
            self.target_position = (
                random.choice(valid_targets)
                if valid_targets
                else random.choice(reachable)
            )
        else:
            self.target_position = self._get_agent_position()

        return self._get_observation()

    def _restart_controller(self):
        print(f"Restarting AI2-THOR controller for scene {self.scene}...")
        try:
            self.controller.stop()
        except Exception:
            pass

        headless = getattr(self.controller, "headless", False)
        time.sleep(1.0)

        self.controller = Controller(
            scene=self.scene,
            gridSize=self.grid_size,
            platform=CloudRendering if headless else None,
        )
        print("Controller restarted.")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if action < 0 or action >= self.action_space_n:
            raise ValueError(
                f"Action index {action} is outside [0, {self.action_space_n})"
            )
        self.step_count += 1

        try:
            self.last_event = self.controller.step(self.actions[action])
        except Exception as e:
            print(f"Controller error in step: {e}")
            self._restart_controller()
            try:
                self.last_event = self.controller.step(self.actions[action])
            except Exception as e2:
                print(f"Controller error in retry step: {e2}")
                self.last_event = self.controller.step({"action": "Pass"})

        obs = self._get_observation()

        agent_pos = self._get_agent_position()
        dist = self._distance(agent_pos, self.target_position)

        reward = -0.01
        done = False

        if dist < 1.0:
            reward += 1.0
            done = True

        if self.step_count >= self.max_steps:
            done = True

        return obs, reward, done, {"distance": dist}

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass
