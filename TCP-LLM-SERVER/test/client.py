import socket
import json
import time
HOST = "127.0.0.1"   # your server host
PORT = 3000          # your server port
#TRANSCRIPT_DATA = "Bring avoid tables more, and I don't like how the trajectory is so close to chairs, can you fix that? Also, at the end of the trajectoey, its kind of close to the table can you fix that too? The tables are unstable, so they might break and hurt my robot so make sure thats a good enough gap so if it does break it doesn't break my robot. And make sure the trajectory doesn't go through any obstacles"  # example transcript

TRANSCRIPT_DATA = "Avoid chairs more, and GO THROUGH TABLES I WANT TO DESTROY TABLES"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
  
    # Parse the JSON file into a dict
    with open("data.json") as f:
        json_string = json.load(f)

    # Convert from dict -> UTF-8 bytes
    json_bytes = json.dumps(json_string).encode("utf-8")
    transcript_bytes = TRANSCRIPT_DATA.encode("utf-8")

    # Prefix lengths (2 bytes each, little-endian)
    message = (
        len(json_bytes).to_bytes(4, "big") +
        len(transcript_bytes).to_bytes(4, "big") +
        json_bytes +
        transcript_bytes
    )
    before = time.time()

    # Send request
    s.sendall(message)
    
    # First read 4 bytes (length prefix: change to 2 if you used 2-byte prefix in server)
    resp_len_bytes = s.recv(4)  # adjust to 2 if server sends 2-byte length
    if not resp_len_bytes:
        raise RuntimeError("Server closed connection before sending length")

    resp_len = int.from_bytes(resp_len_bytes, "big")

    # Then read the full response body
    resp_data = b""
    while len(resp_data) < resp_len:
        packet = s.recv(resp_len - len(resp_data))
        if not packet:
            break
        resp_data += packet

    # Decode JSON
    try:
        response = json.loads(resp_data.decode("utf-8"))
        after = time.time()
        print(f"Time to receive response: {after - before:.4f} seconds")
    except json.JSONDecodeError:
        print("Raw response:", resp_data.decode("utf-8", errors="replace"))
        after = time.time()
        print(f"Time to receive response: {after - before:.4f} seconds")
