import socket
import json

HOST = "127.0.0.1"
PORT = 3000

def recv_all(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError(f"Socket closed early (got {len(data)}/{n} bytes)")
        data += chunk
    return data

scene = {
    "Trajectory": {"Control Points": [[0, 0], [1, 0], [2, 0]]},
    "Obstacles": {},
    "WorldHeadset": {
        "position": {"x": 0, "y": 0, "z": 0},
        "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}
    },
    "ROSbot": {
        "position": {"x": -1, "y": 0, "z": 0},
        "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}
    }
}

transcript = "Go in front of me"

scene_bytes = json.dumps(scene, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
transcript_bytes = transcript.encode("utf-8")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.settimeout(10)  # don’t hang forever
    s.connect((HOST, PORT))

    # Send framed request
    s.sendall(len(scene_bytes).to_bytes(4, "big"))
    s.sendall(len(transcript_bytes).to_bytes(4, "big"))
    s.sendall(scene_bytes)
    s.sendall(transcript_bytes)

    # IMPORTANT: tell server we're done sending
    s.shutdown(socket.SHUT_WR)

    # Receive framed response
    mode = recv_all(s, 1)
    size = int.from_bytes(recv_all(s, 4), "big")
    data = recv_all(s, size)

    print("Mode:", mode)
    print("Response:", data.decode("utf-8"))
