from obsws_python import ReqClient

HOST = "127.0.0.1"
PORT = 4455  # OBS 28+ default
PASSWORD = "Setyup34!"

cl = ReqClient(host=HOST, port=PORT, password=PASSWORD)

print("Connected to OBS.")
scenes = cl.get_scene_list().scenes
for s in scenes:
    print("-", s["sceneName"])
