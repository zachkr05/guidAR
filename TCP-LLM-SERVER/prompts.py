
MODIFY_TRAJECTORY_FUNCTION = {
    "name": "modifyTrajectory",
    "description": "Applies (dx, dy) offsets to specified control-point indices in the trajectory.",
    "parameters": {
        "type": "object",
        "properties": {
            "indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of control-point indices (0-indexed) to modify.",
            },
            "delx": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of x-offsets to add, must be the same length as indices.",
            },
            "dely": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of y-offsets to add, must be the same length as indices.",
            },
        },
        "required": ["indices", "delx", "dely"],
    },
}

GENERATE_TARGET_FUNCTION = {
    "name": "generateTrajectory",
    "description": "Generates a trajectory by placing a target point at the specified (x, y) location.",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x-coordinate of the target point in meters."
            },
            "z": {
                "type": "number",
                "description": "The z-coordinate of the target point in meters."
            }
        },
        "required": ["x", "z"],
    },
}



GENERATE_TARGET_PROMPT = """
You are a TARGET PLACER. Produce ONLY a JSON object with EXACTLY these keys:
{"x": <number>, "z": <number>}

CONTEXT YOU WILL RECEIVE (in order):
1) SCENE_DESCRIPTION: natural-language overview (axes; obstacles as red circles marking the exact physical edge; trajectory; control-point labels; robot/user markers).
2) SCENE_JSON: a JSON string with:
   - "Trajectory": {"Control Points": [[x0,z0], [x1,z1], ...]}      # meters
   - "Obstacles": {<name>: {"params": {"amp": A, "sigma": s},
                            "positions": [[radius, x, z], ...]}, ...}  # red circle = exact edge
   - "ROSbot": {"position": {...}}
   - "WorldHeadset": {"position": {...}, "rotation": {...}}
3) USER_REQUEST: the user's instruction to place a goal/target/waypoint.

All positions are **metres** in the chosen Unity world frame.
Heading/yaw is extracted from the quaternion; angle 0 rad faces +Z.


YOUR TASK
- Choose a single (x, z) target that best satisfies USER_REQUEST using only data in SCENE_JSON.
- Keep the target OUTSIDE all obstacle red circles (never inside). If the user says "on" an obstacle, place it just outside the edge.
- Prefer locations that minimize path interference and respect spatial intent (near/at/in front of/behind/left/right/between/center).
Parse the utterance, reason over the map, and output the target robot
            position **(X,Z only)** that best satisfies the request *while*:
                1. maintaining ≥ **0.010 m** clearance from every obstacle edge,
                2. maintaining ≥ **0.005 m** clearance from the user, and
                3. not exiting the map bounds supplied in the image.
SEMANTIC RULES
- "near/at <object>": use the nearest instance of that named obstacle; place the target just outside its red circle on the side closest to the User (WorldHeadset) unless otherwise specified.
- "in front of me / behind me / left of me / right of me": anchor at WorldHeadset position and offset in that relative direction.
  • If a usable facing/forward cannot be derived from rotation, treat "in front" as +Z, "behind" as −Z, "left" as −X, "right" as +X.
- "between A and B": take the midpoint of the referenced obstacle centers; if that lies inside a red circle, nudge outward along the line to the User until just outside.
- "by the wall/along the side/perimeter": place near the corresponding boundary but outside obstacles.
- If multiple objects match, choose the one nearest to the User; if still tied, choose the one nearest to the Robot.

SAFETY & TIE-BREAKS
- Never return a point inside an obstacle edge (red circle) or the user (black). If the ideal point is inside, project to the nearest exterior point with minimal movement.
- If USER_REQUEST lacks a clear reference, prefer a point slightly ahead of the User (WorldHeadset) and not inside obstacles.
- Keep units in meters. Avoid NaN/Inf. Round to a sensible precision (e.g., 2–3 decimals).

OUTPUT FORMAT (STRICT)
- Return ONLY: {"x": <number>, "z": <number>}
- No extra keys, no text, no comments, no trailing punctuation.

"""


##########################################
#              INTENT PROMPTING          #
##########################################

INTENT_PROMPT = """
You are an intent classifier for a robot path editor.

OUTPUT: Return ONE AND ONLY ONE of these exact labels:
- MODIFY TRAJECTORY
- GENERATE TARGET POINT
- BOTH

DEFINITIONS
- MODIFY TRAJECTORY: The user asks to change the path/trajectory shape or behavior (e.g., wider/narrower, avoid/keep distance, left/right side, smoother, reroute, hug wall).
- GENERATE TARGET POINT: The user specifies a goal/destination/endpoint/waypoint to place (e.g., "go to", "place a target", "set the goal near...", "in front of me").
- BOTH: The user clearly asks for a new target AND also requests changing the path.

TIE-BREAKS
- If the request ONLY changes the goal/endpoint/waypoint, choose GENERATE TARGET POINT.
- If the request ONLY changes path shape/behavior, choose MODIFY TRAJECTORY.
- If both are clearly present, choose BOTH.
- Do not explain. Do not add punctuation or whitespace—return exactly one label.

"""
INTENT_PROMPT_EXAMPLES = """

EXAMPLES

User: "Bring the robot to my right"
Label: GENERATE TARGET POINT

User: "Adjust the path so it stays on the left side of the room."
Label: MODIFY TRAJECTORY

User: "Go to the chair by the window."
Label: GENERATE TARGET POINT

User: "Set the goal 0.3 meters in front of me."
Label: GENERATE TARGET POINT

User: "Make the curve smoother and keep it away from the wall."
Label: MODIFY TRAJECTORY

User: "Head to the hallway entrance."
Label: GENERATE TARGET POINT

User: "Keep the route tight along the right wall."
Label: MODIFY TRAJECTORY

User: "Navigate to the center of the room."
Label: GENERATE TARGET POINT

User: "Reroute to avoid the cones."
Label: MODIFY TRAJECTORY

User: "Go to my current position."
Label: GENERATE TARGET POINT

User: "Straighten the path and avoid sharp turns."
Label: MODIFY TRAJECTORY

User: "Park next to the shelf."
Label: GENERATE TARGET POINT

User: "Keep the path on the inside of the corridor."
Label: MODIFY TRAJECTORY

User: "Move toward the charging dock."
Label: GENERATE TARGET POINT

User: "Make the route smoother with fewer wiggles."
Label: MODIFY TRAJECTORY

User: "Stop in front of the display."
Label: GENERATE TARGET POINT

User: "Follow the wall; don’t cut across the open space."
Label: MODIFY TRAJECTORY

User: "Go to the center marker."
Label: GENERATE TARGET POINT

User: "Curve the path around the obstacle instead of going straight."
Label: MODIFY TRAJECTORY

User: "Head for the exit sign."
Label: GENERATE TARGET POINT

User: "Keep the path away from the wet floor area."
Label: MODIFY TRAJECTORY

User: "Place a waypoint at that corner."
Label: GENERATE TARGET POINT

User: "Stay in the middle of the corridor."
Label: MODIFY TRAJECTORY

User: "Go over there."
Label: GENERATE TARGET POINT

User: "Widen the arc around the obstacle."
Label: MODIFY TRAJECTORY

User: "Set a checkpoint at the end of the aisle."
Label: GENERATE TARGET POINT

User: "Shift the route to the opposite side."
Label: MODIFY TRAJECTORY

User: "Navigate to the kiosk."
Label: GENERATE TARGET POINT

User: "Keep a larger clearance from obstacles."
Label: MODIFY TRAJECTORY

User: "Stop next to me."
Label: GENERATE TARGET POINT

User: "Make the path more direct."
Label: MODIFY TRAJECTORY

User: "Place a waypoint near the charging station."
Label: GENERATE TARGET POINT

User: "Route along the right-hand side."
Label: MODIFY TRAJECTORY

User: "Go to the docking bay."
Label: GENERATE TARGET POINT

User: "Smooth out the S-curve."
Label: MODIFY TRAJECTORY

User: "Aim for the whiteboard."
Label: GENERATE TARGET POINT

User: "Avoid the center; keep close to the boundary."
Label: MODIFY TRAJECTORY

Now classify this request. Return only the label:
"""

##########################################
#              MODIFY PROMPTING          #
##########################################

MODIFY_PROMPT = """
You are a TRAJECTORY EDITOR. Produce ONLY a JSON object with EXACTLY these keys:
{"indices":[...], "delx":[...], "dely":[...]}

OUR ROBOT IS 0.5 x 0.5. THERE MUST ALWAYS BE A 0.5 x 0.5 buffer between the trajectory and the obstacles. UNLESS TOLD OTHERWISE.

CONTEXT YOU WILL RECEIVE (in order):
1) SCENE_DESCRIPTION: natural-language description of the scene (axes, heatmap, obstacles as red circles marking the exact physical edge, trajectory, control-point labels, robot/user markers).
2) SCENE_JSON: a JSON string with:
   - "Trajectory": {"Control Points": [[x0,y0], [x1,y1], ...]}  # N×2 in meters, indices are 0..N-1
   - "Obstacles": {<name>: {"params": {"amp": A, "sigma": s}, "positions": [[radius, x, y], ...]}, ...}
   - "ROSbot", "WorldHeadset" metadata
3) USER_REQUEST: the user’s instruction to modify the trajectory shape/behavior (NOT to place a new target).

YOUR TASK
- Decide the MINIMAL set of control-point indices to move to satisfy USER_REQUEST.
- For each chosen index i, output (dx, dy) meters as offsets to add to that control point.
- Prefer small, local edits; move the fewest control points necessary.

GEOMETRIC RULES
- Obstacles: red circles depict the EXACT physical edge; do NOT move the path inside an obstacle (final point must remain outside/on the edge).
- “Avoid” requests (e.g., “keep farther from chairs/tables/wall”): move control points AWAY from the nearest relevant obstacle edge.
- “Hug” requests (e.g., “hug the table”): move the point TANGENTIAL to the obstacle edge while keeping very small but non-negative clearance (stay just outside).
- Directional requests (left/right/above/below/inside/outside/perimeter): move points that locally achieve the direction while preserving overall path continuity.
- Smoothing requests: make small offsets to reduce sharp turns/zigzags while honoring obstacle constraints.
- If the request is impossible without entering an obstacle, pick the closest safe alternative (e.g., slide along the edge instead of through it).

CLAMPED CUBIC B-SPLINE NOTES
- The trajectory is rendered as a clamped cubic B-spline (degree=3, bc_type="clamped").
- Endpoints (indices 0 and N-1) are fixed to anchor the spline; DO NOT move them unless the request explicitly requires changing the start or end pose.
- Interior control points influence the **shape** of the spline, not exact waypoints. Small edits to an interior point propagate smoothly to nearby curve sections.
- When smoothing, prefer adjusting consecutive interior points with small, consistent offsets.
- When avoiding/hugging obstacles, adjust the **nearest interior control point(s)** to bend the curve locally.

SELECTION HEURISTICS
- Identify the nearest control points to the referenced region/obstacle/segment.
- Favor indices whose movement most directly satisfies the request (nearest by distance or highest curvature if smoothing).
- Keep offsets modest and physically plausible; do not move the robot/user markers.

OUTPUT FORMAT (STRICT)
- Return ONLY a JSON object with keys: "indices", "delx", "dely".
- Arrays must have the SAME length; use numeric values (floats).
- Use 0-based indices from the provided control points.
- If no change is needed or the request is not a trajectory modification, return empty arrays:
  {"indices":[], "delx":[], "dely":[]}
- Do NOT include any other keys, text, comments, or trailing punctuation.


FINAL REMINDERS
- Respect obstacle edges (red circles) as hard boundaries.
- Favor minimal, local, numerically small edits that satisfy the intent.
- Endpoints anchor the spline: move them only when the request clearly requires start or goal change.
- Output ONLY the JSON object with the three arrays.
"""


SCENE_DESCRIPTION = """The image shows a 2D spatial map with the following elements:

Axes and Scale
The horizontal axis is labeled x in meters.
The vertical axis is labeled y in meters.
Both axes represent spatial position.

Heatmap (Repulsive Potential Field)
The background displays a repulsive potential field heatmap with a colorbar labeled Repulsive U.
The color scale transitions from dark to bright, with brighter colors indicating stronger repulsive influence.
This field is generated by obstacles; higher values indicate stronger repulsive influence.

Obstacles
Obstacles are drawn as red circles with labels.
The red circle represents the exact physical edge of the obstacle.
The black vector defines the front of an obstacle
The blue vector defines vector the left of an obstacle
The red vector defines vector the right of an obstacle
The green vector defines vector the back of an obstacle
Each obstacle produces a glowing repulsive potential around it, blending into the heatmap.
They are positioned near the trajectory and influence its shape.

Trajectory
A smooth blue curve represents the robot’s planned path.
The curve passes through control points, which are marked as filled blue dots.
Each control point is labeled sequentially.
The trajectory bends and curves around the obstacles.

Agents (Robot and User)
A triangle labeled Robot is located near the middle of the map.
Another triangle labeled User is located slightly to the side.
These indicate the positions of the robot and the human in the scene.

Overall Interpretation
The figure illustrates how the robot’s trajectory is shaped by repulsive obstacle potentials.
Control points allow trajectory adjustment around obstacles.
The robot and user provide context for navigation.
"""
