# Standard library
from typing import Tuple
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
from LLMReasoner import LLMReasoner
import plotly.io as pio
from utils import * 
from classifier import IntentClassifier

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s: %(message)s"
)
logger = logging.getLogger(__name__)

if logger.isEnabledFor(logging.DEBUG):
    print("Logger is in DEBUG level (or lower).")
else:
    print("Logger is not in DEBUG level.")

class TcpServer:
    def __init__(self, host: str, port: int, read_buffer_size: int = 8192, send_buffer_size: int=262144, rcv_buffer_size:int = 262144, num_connections:int = 1): 
        self.host = host
        self.port = port
        self.read_buffer_size = read_buffer_size  
        self.send_buffer_size = send_buffer_size
        self.rcv_buffer_size = rcv_buffer_size
        self.num_connections = num_connections
        self.plotter = ScenePlotter()
        self.llmClient = LLMReasoner()
        self.intentRecognizer = IntentClassifier()
        
        
    def recv_all(self, sock, n: int) -> bytes | None:
        """Receive exactly n bytes from socket."""
        logger.debug(f"Starting recv_all: expecting {n} bytes")

        start_time = time.time()
        data = b""

        while len(data) < n:
            logger.debug(f"Waiting to receive {n - len(data)} more bytes...")
            try:
                packet = sock.recv(min(self.read_buffer_size, n - len(data)))
            except Exception as e:
                logger.critical(f"Socket recv() failed while expecting {n} bytes: {e}")
                return None

            if not packet:
                logger.critical(
                    f"Connection closed or no data received "
                    f"(expected {n}, got {len(data)})"
                )
                return None

            data += packet
            logger.debug(f"Received {len(packet)} bytes (total {len(data)}/{n})")

        end_time = time.time()
        duration = end_time - start_time
        rate = n / duration if duration > 0 else 0

        logger.debug(
            f"Unpacked {n} bytes in {duration:.4f}s "
            f"({rate/1024/1024:.2f} MB/s)"
        )

        return data

    def socketconfig(self, sock:socket ):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcv_buffer_size)
        logger.debug(f"SO_RCVBUF set to {self.rcv_buffer_size} bytes")

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buffer_size)
        logger.debug(f"SO_SNDBUF set to {self.send_buffer_size} bytes")

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logger.debug("SO_REUSEADDR enabled (socket can be rebound quickly)")

        sock.bind((self.host, self.port))
        logger.info(f"Socket bound to {self.host}:{self.port}")

        sock.listen(self.num_connections)
        logger.info(
            f"Server listening on {self.host}:{self.port} "
            f"(max {self.num_connections} queued connections)"
        )
    def send_error(self, conn, message: str):
        payload = json.dumps({"error": message}).encode("utf-8")
        prefix = len(payload).to_bytes(4, "big")
        conn.sendall(bytes([255]) + prefix + payload)  # 255 = error mode

    def retrv_data (self, conn):
        bytes_json_len = self.recv_all(conn, 4)
        if bytes_json_len is None:
            raise ConnectionError("Failed to receive JSON length prefix")
        
        bytes_transcript_len = self.recv_all(conn, 4)
        if bytes_transcript_len is None:
            raise ConnectionError("Failed to receive transcript length prefix")

        json_len = int.from_bytes(bytes_json_len,"big", signed=False)
        transcript_len = int.from_bytes(bytes_transcript_len,"big", signed=False)

        logger.debug(f"Expecting JSON data of length: {json_len} bytes, transcript length: {transcript_len} bytes")

        bytes_json = self.recv_all(conn,json_len)
        if bytes_json is None:
            raise ConnectionError("Failed to receive JSON data")
        bytes_transcript = self.recv_all(conn, transcript_len)
        if bytes_transcript is None:
            raise ConnectionError("Failed to receive transcript data")

        json_data = json.loads(bytes_json.decode('utf-8'))
        transcript_data = bytes_transcript.decode('utf-8')

        return json_data, transcript_data



    def pack_trajectory_ndarray(self,traj: np.ndarray, prefix_bytes: int = 4) -> Tuple[bytes, bytes]:
        """
        Convert an (N,2) numpy array into a JSON message with a length prefix.
        Returns (prefix, msg) so you can send with: conn.sendall(prefix + msg)

        JSON schema:
        {
          "Trajectory": {
            "Control Points": [[x1, y1], [x2, y2], ...]
          }
        }

        Args:
            traj: np.ndarray with shape (N, 2)
            prefix_bytes: length of the little-endian prefix (2 or 4). Defaults to 4.

        Returns:
            prefix (bytes), msg (bytes)
        """
        if traj is None:
            raise ValueError("traj is None")
        if not isinstance(traj, np.ndarray):
            raise TypeError(f"traj must be a numpy ndarray, got {type(traj)}")
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError(f"traj must have shape (N, 2); got {traj.shape}")
        if prefix_bytes not in (2, 4):
            raise ValueError("prefix_bytes must be 2 or 4")

        payload = {
            "Trajectory": {
                "Control Points": np.asarray(traj, dtype=float).tolist()
            }
        }

        msg = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        prefix = len(msg).to_bytes(prefix_bytes, byteorder="big", signed=False)
        return prefix, msg

    def pack_target_pointer(self, point, prefix_bytes: int = 4):
        payload = {
            "Trajectory": {
                "Control Points": np.asarray(point, dtype=float).tolist()
            }
        }

        msg = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        prefix = len(msg).to_bytes(prefix_bytes, byteorder="big", signed=False)
        return prefix, msg
        
    def sockfunc(self):
        """Main server loop."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socketconfig(sock)

        while True:
            conn, addr = sock.accept()
            self.connection = conn
            print(f"Connected by {addr}")
            
            # Disable Nagle's algorithm for better responsiveness
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            try:
                json_data, transcript_data = self.retrv_data(conn)
                #IMPORTANT UNCOMMENT WHEN DEPLOYING
                #current_scene, fig = self.plotter.buildScene(json_data)
                #json_string = json.dumps(json_data)
             #   print(json_data)
            #    print(transcript_data)
                #Setup the scene for the llm Client
                #json_data, transcript_data = self.retrv_data(conn)

                try:
                    current_scene, fig = self.plotter.buildScene(json_data)
                except Exception as e:
                    logger.exception("buildScene failed; using blank image")

                    fig = None
                    blank = Image.new("RGB", (512, 512), "white")
                    buf = BytesIO()
                    blank.save(buf, format="PNG")
                    current_scene = buf.getvalue()

                json_string = json.dumps(json_data)

                self.llmClient.setTrajectory(json_data)

                #logger.critical("REQUESTING USER INTENTION")
                before = time.time()

                #Get the user intention
                user_intention = self.llmClient.getUserIntention(transcript_data)
                #user_intention = self.intentRecognizer.classify_intent(transcript_data)
                #print(f"User intent: {user_intention}")
                #user_intention = user_intention
                print("RAW user_intention:", user_intention)
                print("TYPE:", type(user_intention))
                print("EXPECTED:", GENTARGETPOINT, MODIFYTRAJ)

                after = time.time()
                logger.critical(f"RECEIEVED USER INTENTION (elapsed: {after - before:.4f}s)")
                #Plot the current scene for the LLM
                pil_img = Image.open(BytesIO(current_scene))
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Saving Image of current Scene")
                    pio.write_image(fig, "old_scene.png", format="png", scale=2) 

                #Build the contents for the llm prompting with regard to their request 
                self.llmClient._build_scene_contents(transcript_data, json_data, pil_img)

                if (user_intention == GENTARGETPOINT):
                    prefix, msg = self.llmClient.reasonTargetPoint()
                    self.connection.sendall(bytes([1]) + prefix + msg)

                # MODIFY branch
                elif (user_intention == MODIFYTRAJ):
                    trajectory = self.llmClient.reasonTrajectoryEdits()
                    
                    x_new,y_new = eval_clamped_bspline_like_cpp(trajectory,400,3)
                    
                    for trace in fig.data:
                        if trace.name == "Trajectory":
                            trace.x = x_new
                            trace.y = y_new
                    
                    # --- remove old control-point shapes/annotations ---
                    #fig.layout.shapes = tuple(
                    #    s for s in fig.layout.shapes if getattr(s, "name", None) != "ctrl_pt"
                    #)
                    #fig.layout.annotations = tuple(
                    #    a for a in fig.layout.annotations if not a.text.startswith("CTRL_PT")
                    #)

                    # --- re-add control points (new positions) ---
                    #r = 0.1

                    xz,yz = trajectory[:,0], trajectory[:, 1]
                    #for idx, (x, y) in enumerate(zip(xz,yz)):
                    #    fig.add_shape(
                    #        type="circle", xref="x", yref="y",
                    #        x0=x-r, y0=y-r, x1=x+r, y1=y+r,
                    #        line=dict(color="blue", width=2),
                    #        fillcolor="rgba(0,0,255,1)",
                    #        name="ctrl_pt"   # so we can filter them out next update
                    #    )
                    #    fig.add_annotation(
                    #        x=x, y=y+r+0.1,
                    #        text=f"CTRL_PT {idx}",
                    #        showarrow=False,
                    #        font=dict(color="black", size=6)
                    #    )


                    #pio.write_image(fig, "new_scene.png", format="png", scale=2)
 
                    prefix, msg = self.pack_trajectory_ndarray(trajectory)  # (N,2) ndarray
                    self.connection.sendall(bytes([0]) + prefix + msg)                    
                                         
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

            finally:
                conn.close()
                print("Connection closed\n")



if __name__ == "__main__":
    load_dotenv()
    # assert os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY not found in environment variables."
    #Start the server
    server = TcpServer("", 3000, 8192)
    server.sockfunc()
