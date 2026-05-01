from obsws_python import ReqClient

HOST = "127.0.0.1"
PORT = 4455
PASSWORD = "Setyup34!"

SCENE = "Ill Dynamics - Live on SkankOut"

cl = ReqClient(host=HOST, port=PORT, password=PASSWORD)

items = cl.get_scene_item_list(SCENE).scene_items

for it in items:
    print(it["sceneItemId"], it["sourceName"])
