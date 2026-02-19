# Standard library
import os
import json
import time
import socket
import struct
from io import BytesIO
import logging
from google import genai
import enum
# Third-party libraries
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from scenePlotter import ScenePlotter
from google.genai import types
from prompts import *
from enums import *
import base64

import logging
logger = logging.getLogger(__name__)


class LLMReasoner:

    """
    Handles all reasoning with the LLM:
    - Decides intention (MODIFY_TRAJECTORY, GENERATE_TARGET_POINT, BOTH).
    - Builds prompts or function-calls.
    - Sends scene + user request to the model.
    - Parses, validates, and projects outputs (JSON -> dx, dy edits or target point).
    - Applies guardrails (schema check, feasibility projector).
    - Returns clean structured results to the planner.
    """

    def __init__(self,):
        # Load environment variables
        load_dotenv()
        # Initialize the client with the API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")

        # Centralized, step-by-step requests your lightweight model can loop through
        self.thought_requests = [
            "Analyze user request to extract target characteristics and spatial relationships.",
            "Resolve ambiguous phrases into explicit spatial directives and numeric parameters (e.g., distance in meters).",
            "Parse SCENE_JSON to get user pose (position + quaternion), obstacles (centers + radii/safety), and map bounds.",
            "Convert user quaternion to yaw using atan2 with the scene’s coordinate system; assume +Z is forward at yaw=0 unless overridden.",
            "Compute forward/right vectors from yaw; support directives: in_front_of, behind, left_of, right_of.",
            "Set placement distance (default 1.0m if unspecified) and compute initial target = user_position + forward * distance (x,z).",
            "Clamp/adjust initial target to stay within map bounds.",
            "Check collision of target against all obstacles using safety margins.",
            "If collision: project target minimally along the normalized vector from obstacle center to target to clear safety radius; recheck.",
            "If still colliding: run a low-cost radial search around the initial heading (small angular and radial steps) with early exit on first valid point.",
            "Validate final target against all obstacles and bounds; ensure minimal deviation from requested directive.",
            "Support heading-sensitive directives by re-evaluating if user facing changes between steps.",
            "Generate output JSON with target coordinates (x,z), chosen distance, directive, and flags (within_bounds, collision_free, adjustments_applied).",
            "Log coordinate-frame assumptions (axis conventions, yaw definition) for reproducibility.",
            "Run quick tests on representative edge cases (yaw=0/±π/±π/2, near-boundary starts, dense obstacles).",
            "Return final JSON result."
        ]
        self.client = genai.Client(api_key=api_key)

        self.modify_traj_tools = types.Tool(function_declarations=[MODIFY_TRAJECTORY_FUNCTION])
        self.gen_target_tools = types.Tool(function_declarations=[GENERATE_TARGET_FUNCTION])
        self.modify_traj_config = types.GenerateContentConfig(
            tools=[self.modify_traj_tools],
            system_instruction=MODIFY_PROMPT,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=["modifyTrajectory"]
                )
            ),
            temperature=0,
        )
        self.gen_target_config = types.GenerateContentConfig(
            tools=[self.gen_target_tools],
            system_instruction=GENERATE_TARGET_PROMPT,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", allowed_function_names=["generateTrajectory"]
                )
            ),
            temperature=0,
        )
        self.intentionModel = "gemini-2.5-flash-lite"
        self.trajectory = None
        
    def _count_tokens(self, contents, model: str) -> int:
        """
        Count total tokens for a Gemini request (text + images).
        Uses Gemini's native tokenizer.
        """
        try:
            resp = self.client.models.count_tokens(
                model=model,
                contents=contents
            )
            return resp.total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return -1

    def getUserIntention(self, transcript: str) -> Intention:
        response = self.client.models.generate_content (

                model=self.intentionModel,
                contents = [ 
                            INTENT_PROMPT,
                            INTENT_PROMPT_EXAMPLES,
                            transcript
                    ],
                config = {
                    'response_mime_type': 'text/x.enum',
                    'response_schema': Intention,
                    },
                )
        tokens = self._count_tokens(
            [INTENT_PROMPT, INTENT_PROMPT_EXAMPLES, transcript],
            self.intentionModel
        )
        logger.info(f"Intent input tokens: {tokens}")
        return response.text

    def _compact_scene(self, json_data: dict) -> dict:
        cps = json_data["Trajectory"]["Control Points"]

        circles = []
        for name, obj in json_data.get("Obstacles", {}).items():
            for r, x, z in obj.get("positions", []):
                circles.append([float(r), float(x), float(z), name])

        return {
            "control_points": cps,
            "obstacle_circles": circles,
            "user": json_data.get("WorldHeadset", {}),
            "robot": json_data.get("ROSbot", {}),
            "bounds": json_data.get("Bounds", None),
        }

    def _build_scene_contents(self, user_request: str, json_data: dict, pil_img) -> None:
        """Build Content with user request, JSON, and inline image."""
        compact = self._compact_scene(json_data)
        scene_json_str = json.dumps(compact, separators=(",", ":"), ensure_ascii=False)

        # Encode PIL image into PNG bytes
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        self.transcript = user_request
        self.requestContent = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=user_request),
                    types.Part(text=scene_json_str),
                    types.Part.from_bytes(
                         data=img_bytes,
                         mime_type="image/png"
                     ),
                ],
            )
        ]


    def reasonTargetPoint(self,):
        model_name = "gemini-2.5-flash-lite"
        input_tokens = self._count_tokens(self.requestContent, model_name)
        logger.info(f"Input tokens for {model_name}: {input_tokens}")

        response = self.client.models.generate_content(
                        model=model_name,
                        config=self.gen_target_config,
                        contents=self.requestContent,
                    )
        # count tokens in response
        output_tokens = self._count_tokens([response.candidates[0].content], model_name)
        logger.info(f"Output tokens for {model_name}: {output_tokens}")
        logger.info(f"Total tokens for {model_name}: {input_tokens + output_tokens}")

        fc = self._extract_function_call(response)
         
        if fc:
            name, args = fc
            logger.info(f"Function to call: {name} | Args: {args}")
            if name == "generateTrajectory":
                print("\033[1;31mFUNCTION WAS CALLED IN THIS BRANCH\033[0m")
                prefix, msg = self.generateTrajectory(**args)
            else:
                print("\033[1;31mFUNCTION WAS CALLED IN THIS BRANCH\033[0m")
                logger.warning(f"Unexpected function name: {name}")
        else:
            args = self._try_parse_json_args(response.text.strip())
            if args:
                print("\033[1;31mFUNCTION WAS CALLED IN THIS BRANCH\033[0m")
                prefix, msg = self.generateTrajectory(**args)
                logger.debug("Called generate traj function")
            else:
                print("\033[1;31mFUNCTION WAS CALLED IN THIS BRANCH\033[0m")
                logger.warning("No function call found and text is not valid JSON.")
                logger.warning(response.text)

        return prefix, msg


    def _extract_function_call(self,resp) -> tuple[str, dict] | None:
        """Scan all candidates/parts; return (name, args) if any function_call exists."""
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    # fc.args is already a dict-like
                    return fc.name, dict(fc.args)
        return None

    def _try_parse_json_args(self,resp_text: str) -> dict | None:
        """Fallback when model returns raw JSON text instead of function_call."""
        try:
            return json.loads(resp_text)
        except Exception:
            return None


    def setTrajectory(self, json_data) -> None:
        # Extract control points
        ctrl_pts = json_data["Trajectory"]["Control Points"]
        self.trajectory = np.array(ctrl_pts, dtype=np.float64)


    def reasonTrajectoryEdits(self) -> np.ndarray:
        chain_of_thought = [
        "Analyze user request to extract target characteristics and spatial relationships.",
        "Resolve ambiguous phrases into explicit spatial directives and numeric parameters (e.g., distance in meters).",
        "Parse SCENE_JSON to get user pose (position + quaternion), obstacles (centers + radii/safety), and map bounds.",
        "Convert user quaternion to yaw using atan2 with the scene’s coordinate system; assume +Z is forward at yaw=0 unless overridden.",
        "Compute forward/right vectors from yaw; support directives: in_front_of, behind, left_of, right_of.",
        "Set placement distance (default 1.0m if unspecified) and compute initial target = user_position + forward * distance (x,z).",
        "Clamp/adjust initial target to stay within map bounds.",
        "Check collision of target against all obstacles using safety margins.",
        "If collision: project target minimally along the normalized vector from obstacle center to target to clear safety radius; recheck.",
        "If still colliding: run a low-cost radial search around the initial heading (small angular and radial steps) with early exit on first valid point.",
        "Validate final target against all obstacles and bounds; ensure minimal deviation from requested directive.",
        "Support heading-sensitive directives by re-evaluating if user facing changes between steps.",
        "Generate output JSON with target coordinates (x,z), chosen distance, directive, and flags (within_bounds, collision_free, adjustments_applied).",
        "Log coordinate-frame assumptions (axis conventions, yaw definition) for reproducibility.",
        "Run quick tests on representative edge cases (yaw=0/±π/±π/2, near-boundary starts, dense obstacles).",
        "Return final JSON result."
        ]

        model_name = "gemini-2.5-flash-lite"    
        input_tokens = self._count_tokens(self.requestContent, model_name)
        logger.info(f"Input tokens for {model_name}: {input_tokens}")
        before = time.time()
        response = self.client.models.generate_content(
                        model=model_name,
                        config=self.modify_traj_config,
                        contents=self.requestContent,
                    )
        
        after = time.time()
        output_tokens = self._count_tokens([response.candidates[0].content], model_name)
        logger.info(f"Output tokens for {model_name}: {output_tokens}")
        logger.info(f"Total tokens for {model_name}: {input_tokens + output_tokens}")
        logger.info(f"LLM latency: {after - before:.3f} seconds")

        logger.info(f"Time taken to receive response: {after - before:.3f} seconds")
        fc = self._extract_function_call(response)
        
        if fc:
            name, args = fc
            logger.info(f"Function to call: {name} | Args: {args}")
            if name == "modifyTrajectory":
                self.applyTrajectoryEdits(**args)
            else:
                logger.warning(f"Unexpected function name: {name}")
        else:
            args = self._try_parse_json_args(response.text.strip())
            if args:
                self.applyTrajectoryEdits(**args)
            else:
                logger.warning("No function call found and text is not valid JSON.")
                logger.warning(response.text)
        
        return self.trajectory

        
    def applyTrajectoryEdits(self, indices: np.ndarray, delx: np.ndarray, dely: np.ndarray) -> np.ndarray:
        """
        Modify the trajectory, send it to client, and replot the world as modified_world.png.
        """
        if not (len(indices) == len(delx) == len(dely)):
            raise ValueError("indices, delx, and dely must all have the same length")

        for idx, dx, dy in zip(indices, delx, dely):
            if 0 <= idx < len(self.trajectory):
                self.trajectory[idx, 0] += dx
                self.trajectory[idx, 1] += dy
            else:
                raise IndexError(f"Index {idx} out of bounds for trajectory of length {len(self.trajectory)}")

    def generateTrajectory(self, x: float, z: float) -> None:
        """
        Generate a trajectory through the given target point (x, y)
        and send the result back to the client over the socket.
        """
        # For now, just wrap the target point in a response
        traj_data = {
            "Target": {
                "x": x,
                "z": z
            }
        }

        # Serialize to JSON
        message = json.dumps(traj_data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        # Add a 2-byte little-endian length prefix
        length_prefix = len(message).to_bytes(4, "big", signed=False)

        # Send back over the active socket connection
        return length_prefix, message

    def generate_modified_trajectory_msg(self) -> None:
        """
        Serialize the current trajectory and send it over the socket
        with a 4-byte little-endian length prefix.
        """
        traj_data = {
            "Trajectory": {
                "Control Points": self.trajectory.tolist()
            }
        }
        message = json.dumps(traj_data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        length_prefix = len(message).to_bytes(4, byteorder="big", signed=False)
        return length_prefix, message


