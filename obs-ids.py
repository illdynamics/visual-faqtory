import os

from obsws_python import ReqClient

HOST = os.environ.get("OBS_HOST", "127.0.0.1")
PORT = int(os.environ.get("OBS_PORT", "4455"))
PASSWORD = os.environ.get("OBS_PASSWORD")
if not PASSWORD:
    raise SystemExit("OBS_PASSWORD environment variable is required")

SCENE = "Ill Dynamics - Live on SkankOut"

cl = ReqClient(host=HOST, port=PORT, password=PASSWORD)

items = cl.get_scene_item_list(SCENE).scene_items

for it in items:
    print(it["sceneItemId"], it["sourceName"])
