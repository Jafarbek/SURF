# import pyrealsense2 as rs
# import numpy as np
# import cv2
# thres = 0.45 # Threshold to detect object
# nms_threshold = 0.2

# classNames= []
# classFile = "/home/arcl/catkin_ws/src/Object_detection(CPU/coco.names"
# with open(classFile,'rt') as f:
#     classNames = f.read().rstrip('\n').split('\n')

# #print(classNames)
# configPath = "/home/arcl/catkin_ws/src/Object_detection(CPU/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
# weightsPath = "/home/arcl/catkin_ws/src/Object_detection(CPU/frozen_inference_graph.pb"

# pipe = rs.pipeline()
# cfg = rs.config()

# cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# pipe.start(cfg)

# while True:
#     frames = pipe.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     depth_frame = frames.get_depth_frame()

#     color_image = np.asanyarray(color_frame.get_data())
#     depth_image = np.asanyarray(depth_frame.get_data())


#     depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

#     gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('Color Image', color_image)
#     cv2.imshow('Depth Image', depth_image)
#     cv2.imshow('Depth Image in CM', gray_image)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# pipe.stop()

import pyrealsense2 as rs
import numpy as np
import cv2

# Thresholds
thres = 0.45  # Threshold to detect object
nms_threshold = 0.2

# Load class names
classNames = []
classFile = "/home/arcl/catkin_ws/src/Object_detection(CPU/coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model
configPath = "/home/arcl/catkin_ws/src/Object_detection(CPU/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/arcl/catkin_ws/src/Object_detection(CPU/frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Initialize RealSense
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipe.start(cfg)

while True:
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Object detection
    classIds, confs, bbox = net.detect(color_image, confThreshold=thres, nmsThreshold=nms_threshold)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Calculate distance
            x_center = box[0] + box[2] // 2
            y_center = box[1] + box[3] // 2
            distance = depth_image[y_center, x_center] * 0.001  # Convert from mm to meters

            # Ignore objects further than 1 meter
            if distance > 1.0 or distance == 0:
                continue

            # Ignore objects with confidence lower than 70%
            if confidence < 0.6:
                continue

            cv2.rectangle(color_image, box, color=(0, 255, 0), thickness=2)
            cv2.putText(color_image, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image, f'{distance:.2f}m', (box[0] + 10, box[1] + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('RealSense', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
pipe.stop()
cv2.destroyAllWindows()