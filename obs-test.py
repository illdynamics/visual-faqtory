import os

from obsws_python import ReqClient

HOST = os.environ.get("OBS_HOST", "127.0.0.1")
PORT = int(os.environ.get("OBS_PORT", "4455"))  # OBS 28+ default
PASSWORD = os.environ.get("OBS_PASSWORD")
if not PASSWORD:
    raise SystemExit("OBS_PASSWORD environment variable is required")

cl = ReqClient(host=HOST, port=PORT, password=PASSWORD)

print("Connected to OBS.")
scenes = cl.get_scene_list().scenes
for s in scenes:
    print("-", s["sceneName"])
