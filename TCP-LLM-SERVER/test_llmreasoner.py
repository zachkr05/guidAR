from LLMReasoner import LLMReasoner
from PIL import Image

llm = LLMReasoner()

scene = {
    "Trajectory": {
        "Control Points": [[0,0],[1,0],[2,0]]
    },
    "Obstacles": {},
    "WorldHeadset": {"position": {"x":0,"z":0}},
    "ROSbot": {"position": {"x":-1,"z":0}}
}

img = Image.new("RGB", (256,256), "white")

llm._build_scene_contents(
    user_request="Go 1 meter in front of me",
    json_data=scene,
    pil_img=img
)

prefix, msg = llm.reasonTargetPoint()
print("Response:", msg.decode())
