import json
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Tuple, Any, List
from utils import *
from PIL import Image
import plotly.express as px
import logging
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio

ROBOT_NAME = "ROSbot"
PERSON_NAME = "WorldHeadset"
TRAJECTORY_NAME = "Trajectory"
OBSTACLES_NAME = "Obstacles"

import logging
logger = logging.getLogger(__name__)

class ScenePlotter:
    def __init__(self) -> None:
        self.robot_pose:Tuple[np.ndarray, np.ndarray] = None
        self.user_pose:Tuple[np.ndarray, np.ndarray] = None
        self.ctrl_pts:np.ndarray = None 
        self.obstacles_params:Dict[str, np.ndarray] = None #amp, sigma,
        self.obstacles_positions:Dict[str, np.ndarray] = None 
        self.obstacles_rotations:Dict[str, np.ndarray] = None
        self.x_max: float
        self.x_min: float
        self.y_max: float
        self.y_min: float

    def extractObstacles(self, json_object: Dict[str, Any]) -> None:
        """
        Populate obstacle parameters and positions.

        Args:
            json_object: Dict mapping obstacle groups (e.g. "chairs") to
                        their parameters and positions.
        """
        self.obstacles_params = {}
        self.obstacles_positions = {}
        self.obstacles_rotations = {}

        for group_name, group_data in json_object.items():
            params = group_data["params"]
            amp = params["amp"]
            sigma = params["sigma"]

            # store [amp, sigma] as a 1D np.array
            self.obstacles_params[group_name] = np.array([amp, sigma], dtype=np.float64)

            # store positions as 2D np.array of shape (N, 3): [radius, x, y]
            self.obstacles_positions[group_name] = np.array(group_data["positions"], dtype=np.float64)
            self.obstacles_rotations[group_name] = np.array(group_data["rotations"], dtype=np.float64)

    def _collect_points_xy(self) -> np.ndarray:
        """Return an (N,2) array of all XZ points we care about."""
        chunks: list[np.ndarray] = []

        # robot/user are (rot, pos) where pos = [x, y, z] — we project to (x, z)
        if self.robot_pose is not None and self.robot_pose[1].size >= 3:
            rx, _, rz = np.asarray(self.robot_pose[1], dtype=float)[:3]
            chunks.append(np.atleast_2d([rx, rz]))
            logger.debug("added robot position (x,z) = (%.2f, %.2f) to chunks", rx, rz)

        if self.user_pose is not None and self.user_pose[1].size >= 3:
            ux, _, uz = np.asarray(self.user_pose[1], dtype=float)[:3]
            chunks.append(np.atleast_2d([ux, uz]))
            logger.debug("added user position  (x,z) = (%.2f, %.2f) to chunks", ux, uz)

        # Control points can be (N,3)->(x,y,z) or (N,2)->already (x,z)
        if self.ctrl_pts is not None and len(self.ctrl_pts) > 0:
            cp = np.asarray(self.ctrl_pts, dtype=float)
            if cp.ndim != 2 or cp.shape[0] == 0:
                pass  # nothing to add
            else:
                if cp.shape[1] >= 3:
                    pts_xz = np.column_stack((cp[:, 0], cp[:, 2]))  # use x,z
                elif cp.shape[1] == 2:
                    pts_xz = cp[:, :2]  # assume already (x,z)
                else:
                    pts_xz = np.empty((0, 2), dtype=float)

                if pts_xz.size:
                    chunks.append(np.atleast_2d(pts_xz))
                    # Optional verbose logging
                    # xs = ", ".join(f"{x:.2f}" for x in pts_xz[:, 0])
                    # zs = ", ".join(f"{z:.2f}" for z in pts_xz[:, 1])
                    # logger.debug("added %d control points: X=[%s], Z=[%s]", len(pts_xz), xs, zs)

        if not chunks:
            logger.critical("No XZ chunks added; returning tiny fallback box")
            return np.array([[0.0, 0.0], [1e-6, 1e-6]], dtype=float)

        return np.vstack(chunks)


    def compute_bounds(
        self,
        padding_frac: float = 0.10,   # 10% padding of span
        min_pad: float = 0.25,        # at least 0.25 units padding
        clamp: tuple | None = None,   # e.g. (0, 5, 0, 5) to keep inside 5x5
        square: bool = True
    ) -> tuple[float, float, float, float]:
        P = self._collect_points_xy()
        xy_min = np.nanmin(P, axis=0)
        xy_max = np.nanmax(P, axis=0)
        span = np.maximum(xy_max - xy_min, 1e-9)

        # padding per-axis
        pad = np.maximum(padding_frac * span, min_pad)
        x_min, x_max = xy_min[0] - pad[0], xy_max[0] + pad[0]
        y_min, y_max = xy_min[1] - pad[1], xy_max[1] + pad[1]

        # make square view (nice for spatial maps)
        if square:
            cx = 0.5 * (x_min + x_max)
            cy = 0.5 * (y_min + y_max)
            half = 0.5 * max(x_max - x_min, y_max - y_min)
            x_min, x_max = cx - half, cx + half
            y_min, y_max = cy - half, cy + half

        # clamp to fixed world bounds if provided (e.g., 0..5)
        if clamp is not None:
            cxmin, cxmax, cymin, cymax = clamp
            x_min, x_max = max(cxmin, x_min), min(cxmax, x_max)
            y_min, y_max = max(cymin, y_min), min(cymax, y_max)

            # if clamping collapses range, reopen minimally
            if not (x_max > x_min):
                mid = 0.5 * (cxmin + cxmax)
                eps = 1e-3
                x_min, x_max = mid - eps, mid + eps
            if not (y_max > y_min):
                mid = 0.5 * (cymin + cymax)
                eps = 1e-3
                y_min, y_max = mid - eps, mid + eps

        return x_min, x_max+5, y_min, y_max+5

    def parseJson(self, json_data:Dict[str, Any]) -> None:
        
        #Populate current global variables
        robot = json_data[ROBOT_NAME]
        person = json_data[PERSON_NAME]
        trajectory = json_data[TRAJECTORY_NAME]
        Obstacle_map = json_data[OBSTACLES_NAME]

        #Get robot and human pose
        robot_rotation = extractRotation(robot)
        person_rotation = extractRotation(person)
        robot_position = extractPosition(robot)
        person_position = extractPosition(person)

        self.robot_pose = (robot_rotation, robot_position)
        self.user_pose = (person_rotation, person_position)

        #Retrieve and store the trajectory as a np array
        self.ctrl_pts = np.array(trajectory["Control Points"])

        #Retrieve and store the obstacles
        self.extractObstacles(Obstacle_map)

        print("Robot pose:", self.robot_pose)
        print("User pose:", self.user_pose)
        print("Control points (ctrl_pts):", self.ctrl_pts)
        print("Obstacles parameters (amp, sigma):", self.obstacles_params)
        print("Obstacles positions:", self.obstacles_positions)

    #Returns the whole scene as a plotly image for upload to the gemini api 
    def buildScene(self, json_data:Dict[str, Any]) -> bytes:

        self.parseJson(json_data)
        print("new vals \n")
        bounds = self.compute_bounds(padding_frac=0.10, min_pad=0.25, square=True)

        fig = go.Figure()
        apply_bounds(fig, bounds)
        add_repulsive_potential_heatmap(fig,self.obstacles_positions, self.obstacles_params, bounds,nx=250,ny=250,opacity=0.7)
        plotConfig(fig,bounds)
        plotArrow(fig,self.robot_pose,"Robot")
        plotArrow(fig,self.user_pose,"User")
        plotTrajectory(fig, self.ctrl_pts)
        plotObstacles(fig, self.obstacles_positions, self.obstacles_params, self.obstacles_rotations)
        
        return pio.to_image(fig, format="png"), fig
    
