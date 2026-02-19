from typing import Dict
import numpy as np
import math
from typing import Tuple, Any, List
from PIL import Image
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
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

import numpy as np
import plotly.graph_objects as go

def quaternion_to_yaw(q) -> float:
    """
    Yaw (heading) around Unity Y-axis, returned in *radians*.
    We negate the result so that +yaw (clockwise in Unity LH)
    becomes +CCW in screen coordinates.
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    yaw_unity = math.atan2(2 * y * w - 2 * x * z, 1 - 2 * y * y - 2 * z * z)
    print("Quaternion: ", x, y, z, w)
    print("degrees", np.degrees(yaw_unity))
    return -np.degrees(yaw_unity)

def extractRotation(json_object:Dict[str, Any]) -> np.ndarray:
    rot = json_object.get("rotation", {})
    return np.array([
        rot.get("x", 0.0),
        rot.get("y", 0.0),
        rot.get("z", 0.0),
        rot.get("w", 1.0),
    ], dtype=float)

def extractPosition(json_object:Dict[str, Any]) -> np.ndarray:
    pos = json_object.get("position", {})
    x = pos.get("x", 0.0)
    y = pos.get("y", 0.0)   
    z = pos.get("z", 0.0)
    return np.array([x, y, z], dtype=float)


def plotArrow(fig, object, object_name):
    print("ARROW ROTATION")
    print(object[0])
    robot_yaw = quaternion_to_yaw(object[0])
    robot_img = Image.open("assets/arrow.png").rotate(robot_yaw, expand=True)
    print("Adding object ", object_name)
    
    # Plot the robot image
    fig.add_layout_image(
        dict(
            source=robot_img,
            x=object[1][0],
            y=object[1][2], 
            sizex=0.3,
            sizey=0.3,
            xref="x",
            yref="y",
            xanchor="center",
            yanchor="middle",
            layer="above"
        )
    )

    # Plot direction line
    yaw_rad = np.radians(robot_yaw)
    line_length = 0.5  # Length of direction line
    end_x = object[1][0] + line_length * np.sin(yaw_rad)
    end_y = object[1][2] + line_length * np.cos(yaw_rad)
    
    #fig.add_trace(go.Scatter(
    #    x=[object[1][0], end_x],
    #    y=[object[1][2], end_y],
    #    mode='lines',
    #    line=dict(color='red', width=2),
    #    showlegend=False
    #))

    # Add label
    fig.add_annotation(
        x=object[1][0],
        y=object[1][2]+0.2,
        text=object_name,
        showarrow=False,
        font=dict(color="black", size=6),
        align="center"
    )
    

def plotConfig(fig, bounds, *, dtick=0.5, show_grid=True):
    x_min, x_max, y_min, y_max = bounds

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=260, t=40, b=60),  # room for Scene Key on the right
        xaxis=dict(
            title="x (m)",
            range=[x_min, x_max],
            showgrid=show_grid,
            dtick=dtick,
            gridcolor="rgba(0,0,0,0.08)",
            tickfont=dict(size=10),
            showline=True, mirror=True, linewidth=1, linecolor="rgba(0,0,0,0.4)",
            zeroline=False,
        ),
        yaxis=dict(
            title="z (m)",
            range=[y_min, y_max],
            showgrid=show_grid,
            dtick=dtick,
            gridcolor="rgba(0,0,0,0.08)",
            tickfont=dict(size=10),
            showline=True, mirror=True, linewidth=1, linecolor="rgba(0,0,0,0.4)",
            zeroline=False,
            scaleanchor="x", scaleratio=1,   # equal units
        ),
    )


def apply_bounds(fig: go.Figure, bounds: tuple[float, float, float, float]) -> None:
    x_min, x_max, y_min, y_max = bounds
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max], scaleanchor="x", scaleratio=1)  # equal units



def generate_clamped_knots(n_ctrl: int, degree: int) -> list[float]:
    m = n_ctrl + degree + 1
    K = [0.0] * m
    for i in range(degree+1, n_ctrl):
        K[i] = float(i - degree)
    last = float(n_ctrl - degree)
    for i in range(n_ctrl, m):
        K[i] = last
    return K


# --------- 2) de Boor: identical flow to your C++  ---------
def de_boor(t: float, degree: int, P: np.ndarray, K: list[float]) -> np.ndarray:
    """
    Python port of your deBoor (Cox–de Boor) for 2D points.
    P: (n_ctrl, 2)
    """
    n = P.shape[0] - 1

    # locate knot span k with K[k] <= t < K[k+1]
    k = degree
    for i in range(degree, n + 1):
        if K[i] <= t < K[i + 1]:
            k = i
            break
    # special-case right end
    if t >= K[n + 1]:
        k = n

    # local working copy d[0..degree]
    d = [None] * (degree + 1)
    for j in range(degree + 1):
        d[j] = P[k - degree + j].astype(float).copy()

    # Cox–de Boor recursion
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            denom = K[j + 1 + k - r] - K[j + k - degree]
            alpha = 0.0 if denom == 0.0 else (t - K[j + k - degree]) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[degree]

# --------- 3) Sampler: identical t-range mapping  ---------
def eval_clamped_bspline_like_cpp(ctrl_xy: np.ndarray,
                                  total_samples: int = 400,
                                  degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Matches your generateClampedBSpline:
      - same knot vector
      - t in [K[degree], K[m-1-degree]]  (== [K[p], K[n+1]])
      - includes the right endpoint so it reaches the last control point
    """
    pts = np.asarray(ctrl_xy, dtype=float)
    n_ctrl = pts.shape[0]

    if n_ctrl == 0 or total_samples < 2:
        if n_ctrl:
            return np.array([pts[0,0]]), np.array([pts[0,1]])
        return np.array([]), np.array([])

    if n_ctrl <= degree:
        # mirror your early-return behavior
        xs, ys = pts[:,0], pts[:,1]
        return xs, ys

    K = generate_clamped_knots(n_ctrl, degree)
    t_min = K[degree]
    t_max = K[len(K) - 1 - degree]  # == K[n+1]

    xs = np.empty(total_samples, float)
    ys = np.empty(total_samples, float)
    for i in range(total_samples):
        u = 0.0 if total_samples == 1 else i / (total_samples - 1)  # [0,1]
        t = t_min + u * (t_max - t_min)
        p = de_boor(t, degree, pts, K)
        xs[i], ys[i] = p[0], p[1]
        #print("XS: " + str(xs[i]))
        #print("YS: " + str(ys[i]))
    return xs, ys

# --------- 4) Plot function using the above  ---------
def plotTrajectory(fig: go.Figure,
                   ctrl_pts: np.ndarray,
                   r: float = 0.1,
                   samples: int = 500,
                   degree: int = 3):
    """
    ctrl_pts: (N,2) in plot coordinates (e.g., Unity XZ mapped to (x,y)).
    Keeps the given order (no sorting).
    """
    ctrl_xy = np.asarray(ctrl_pts, float)[1:, :2]

    # control points
    for idx, (x, y) in enumerate(ctrl_xy):
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                      line=dict(color="blue", width=2),
                      fillcolor="rgba(0,0,255,1)", name = "ctrl_pt")
        fig.add_annotation(x=x, y=y + r + 0.1, text=f"CTRL_PT {idx}", showarrow=False, font=dict(color="black", size=6))
    print(ctrl_xy)
    # spline sample (matches your C++)
    x_new, y_new = eval_clamped_bspline_like_cpp(ctrl_xy, total_samples=samples, degree=degree)
    fig.add_trace(go.Scatter(x=x_new, y=y_new, mode="lines",
                             name="Trajectory", line=dict(color="blue")))


    x_new[-1], y_new[-1] = ctrl_xy[-1]
    # sanity: ensure end hits last control point (within FP fuzz)
    if x_new.size:
        assert np.allclose([x_new[0],  y_new[0]],  ctrl_xy[0],  atol=1e-9)
        assert np.allclose([x_new[-1], y_new[-1]], ctrl_xy[-1], atol=1e-9)

def add_repulsive_potential_heatmap(fig, obstacle_positions, obstacle_params, bounds,
                                    nx=200, ny=200, opacity=0.7):
    """
    Add a heatmap of ONLY the repulsive potential, summed over all obstacle groups.

    Args:
        fig: plotly.graph_objects.Figure to add the heatmap to.
        obstacle_positions (dict[str, np.ndarray]): each value is shape (Ni, 3): [radius, x, y]
        obstacle_params (dict[str, tuple[float, float] or list[float]]): group -> (amp, sigma)
        bounds (tuple[float, float, float, float]): (x_min, x_max, y_min, y_max)
        nx (int): number of samples along x
        ny (int): number of samples along y
        opacity (float): heatmap opacity (so icons/annotations remain visible)

    Returns:
        fig (the same figure, with a heatmap trace added)
    """
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(xs, ys)     # shapes: (ny, nx)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (ny*nx, 2)

    # Accumulate repulsive potential U over all groups
    U = np.zeros(grid_pts.shape[0], dtype=float)

    for group_name, arr in obstacle_positions.items():
        if arr is None or len(arr) == 0:
            continue

        # Centers for this group: take (x, y) columns; rows are [radius, x, y]
        centers = np.asarray(arr, dtype=float)[:, 1:3]  # (Ni, 2)

        # Group params
        amp_sigma = obstacle_params.get(group_name, None)
        if amp_sigma is None:
            continue
        amp, sigma = float(amp_sigma[0]), float(amp_sigma[1])
        if sigma <= 0:
            continue

        # Vectorized U_group(p) = sum_j amp * exp( -||p - o_j||^2 / (2*sigma^2) )
        diff = grid_pts[:, None, :] - centers[None, :, :]        # (P, Ni, 2)
        d2 = np.einsum('ijk,ijk->ij', diff, diff)                # (P, Ni)
        U += np.sum(amp * np.exp(-0.5 * d2 / (sigma ** 2)), axis=1)

    Z = U.reshape(ny, nx)

    # Add heatmap (ONLY repulsive potential)
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=Z,
        colorbar=dict(title="Repulsive U"),
        opacity=opacity,
        showscale=True,
    ))

    # Lock to bounds and keep aspect ratio square in data units
    fig.update_xaxes(range=[x_min, x_max])
    fig.update_yaxes(range=[y_min, y_max], scaleanchor="x", scaleratio=1)
    fig.update_traces(selector=dict(type="heatmap"), colorscale="Inferno")
    return fig

def plotObstacles(fig, obstacle_positions, obstacle_params, obstacle_rotations):
    # obstacle_positions[group] -> array of shape (N, 3): [radius, x, y]
    # obstacle_rotations[group] -> (N,4) quaternions OR (N,) yaw angles (radians)

    for group_name, arr in obstacle_positions.items():
        amp, sigma = obstacle_params[group_name]

        # ensure rotations align with this group's obstacles
        rotations = np.asarray(obstacle_rotations[group_name])
        if len(rotations) != len(arr):
            raise ValueError(f"{group_name}: rotations length {len(rotations)} != positions length {len(arr)}")

        for i, (r, x, y) in enumerate(arr):
            rot_i = rotations[i]

            # compute yaw from quaternion or use scalar yaw directly
            if np.size(rot_i) == 4:
                yaw = quaternion_to_yaw(rot_i)
            else:
                yaw = float(np.asarray(rot_i).reshape(()))  # scalar

            # obstacle circle
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x - r, y0=y - r, x1=x + r, y1=y + r,
                line=dict(color="red", width=2),
                layer="above",
            )

            # heading line (nose) as a vector of length r in yaw direction
            hx = x + r * math.cos(yaw)
            hy = y + r * math.sin(yaw)
            fig.add_annotation(
                x=hx, y=hy,
                ax=x, ay=y,
                xref="x", yref="y",
                axref="x", ayref="y",   # <- add these
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="black",
            )

            hx = x + r * math.cos(yaw)
            hy = y + r * math.sin(yaw)
            
            # Front vector (same as original)
            fig.add_annotation(
                x=x + r * math.cos(yaw), 
                y=y + r * math.sin(yaw),
                ax=x, ay=y,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="black",
            )

            # Left vector (yaw + 90 degrees)
            fig.add_annotation(
                x=x + r * math.cos(yaw + math.pi/2),
                y=y + r * math.sin(yaw + math.pi/2),
                ax=x, ay=y,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="blue",
            )

            # Right vector (yaw - 90 degrees) 
            fig.add_annotation(
                x=x + r * math.cos(yaw - math.pi/2),
                y=y + r * math.sin(yaw - math.pi/2),
                ax=x, ay=y,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="red",
            )

             # back vector (yaw - 90 degrees) 
            fig.add_annotation(
                x=x + r * math.cos(yaw - math.pi),
                y=y + r * math.sin(yaw - math.pi),
                ax=x, ay=y,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="green",
            )


            # label
            fig.add_annotation(
                x=x, y=y + 0.2,
                text=group_name,
                showarrow=False,
                font=dict(color="black", size=6),
                align="center",
            )

    return fig


def calculateAttractiveGradient(k_att, goalPosition, point):
    return -k_att * (point - goalPosition)

def calculateAttractivePotential(k_att, goalPosition, point):
    return -k_att * np.linalg.norm(point - goalPosition)**2

def calculateRepulsiveGradient(point, obstacles, goal, amp, sigma):
    """
    Calculate the repulsive gradient from a group of obstacles.

    Args:
        point (np.ndarray): Shape (2,), current position [x, y].
        obstacles (list of np.ndarray): Each obstacle is (2,) array (position).
        goal (np.ndarray): Shape (2,), goal position.
        amp (float): Amplitude parameter.
        sigma (float): Sigma parameter.

    Returns:
        np.ndarray: Shape (2,), the total repulsive gradient.
    """
    result = np.zeros(2)
    k_rot = 0.8

    for obs in obstacles:
        # --- signed distance ---
        diff = point - obs
        signed_dist = max(1e-6, np.linalg.norm(diff))

        # --- normal vector (normalized diff) ---
        normal_vec = diff / signed_dist

        # --- potential magnitude ---
        ptl_mag = amp * np.exp(-0.5 * signed_dist**2 / (sigma**2))

        # --- repulsive gradient ---
        grad_repulsive = (ptl_mag / (sigma**2)) * normal_vec

        # --- tangent vector (90° rotation of normal) ---
        tangent_vec = np.array([-normal_vec[1], normal_vec[0]])

        # --- direction toward goal ---
        goal_dir = goal - point
        goal_dir /= np.linalg.norm(goal_dir)

        # --- rotation sign (cross product in 2D) ---
        rot_sign = 1.0 if (normal_vec[0] * goal_dir[1] - normal_vec[1] * goal_dir[0]) > 0 else -1.0

        # --- rotational gradient ---
        grad_rot = k_rot * rot_sign * ptl_mag * tangent_vec

        # --- accumulate ---
        result += grad_repulsive + grad_rot

    return result

def calculate_repulsive_potential(point, obstacles, amp, sigma):
    """
    Repulsive potential at a 2D point from a group of obstacles.

    U(p) = sum_i  amp * exp( - ||p - o_i||^2 / (2*sigma^2) )

    Args:
        point (array-like): shape (2,), [x, y].
        obstacles (array-like): shape (N, 2) or list of (2,) arrays for obstacle centers.
        amp (float): amplitude A.
        sigma (float): spread σ.

    Returns:
        float: scalar potential U(p).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    p = np.asarray(point, dtype=float).reshape(2,)
    obs = np.asarray(obstacles, dtype=float).reshape(-1, 2)

    if obs.size == 0:
        return 0.0

    # Squared distances ||p - o_i||^2 (no need to take sqrt since it’s squared in the exponent)
    diff = obs - p  # (N, 2)
    d2 = np.einsum('ij,ij->i', diff, diff)  # faster & stable sum of squares per row

    return float(np.sum(amp * np.exp(-0.5 * d2 / (sigma ** 2))))
