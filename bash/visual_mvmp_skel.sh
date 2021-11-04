# Start the server:
python apps/vis/vis_server.py --cfg config/vis3d/o3d_scene.yml write True out E:\study\EasyMocap\datasets\dongao\output-track/skel-multi camera.cz 3. camera.cy 0.5
# Send the keypoints:
python apps/vis/vis_client.py --path E:\study\EasyMocap\datasets\dongao\output\keypoints3d --step 4
