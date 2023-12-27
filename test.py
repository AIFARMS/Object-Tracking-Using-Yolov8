#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('python --version')


# In[2]:


import ultralytics
ultralytics.__version__


# In[4]:


import torch
torch.__version__


# In[5]:


torch.cuda.get_device_name(0)


# # Detect, track and count Persons

# In[1]:


get_ipython().run_line_magic('cd', 'yolov8_DeepSORT')


# In[2]:


from ultralytics import YOLO

import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np
import pandas as pd

# In[4]:

# Create an empty DataFrame to store frame_time and centroid data
df = pd.DataFrame(columns=['FrameTime(s)', 'Centroid', 'Behavior'])

# Load the behavior data from test.xlsx
behavior_df = pd.read_excel('behavior_labels/12151_111_TW.xlsx')

# Define the video path
video_path = 'test_videos/12151_111.mp4'

cap = cv2.VideoCapture(video_path)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[5]:
# In[ ]:
model = YOLO("/iccp/jenish/Object_Tracking/runs/segment/train/weights/best.pt")  # load a pretrained model (recommended for training)
frame_number = 0
frame_time = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    if ret:
        frame_number += 1
        frame_time = frame_number / fps
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = og_frame.copy()

       # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

        results = model.predict(frame, device=0, classes=0, conf=0.8)

        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            probs = result.probs  # Class probabilities for classification outputs
            cls = boxes.cls.tolist()  # Convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh  # box with xywh format, (N, 4)

        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)
        # print("Coordinates: ", bboxes_xywh)

        print("Coordinates: ", xyxy)
        print("Frame Number: ", frame_number)
        print("Frame Time: ", frame_time)
        
        for coord in xyxy:
            x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = coord[2] - coord[0]  # Calculate width
            h = coord[3] - coord[1]  # Calculate height

            # Set color values for red, blue, and green
            red_color = (0, 0, 255)  # (B, G, R)
            color = red_color

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(og_frame, f"Pig", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            # calculating centroid of rectangle (note: we assume here that there is only one pig in a frame)
            c1 = x1 + (w / 2)
            c2 = y1 + (h / 2)

        # Draw a point or visual indication on the frame for the centroid
        center_coordinates = (int(c1), int(c2))
        radius = 5
        color = (0, 255, 0)  # Green color in BGR
        thickness = -1  # Filled circle
        cv2.circle(og_frame, center_coordinates, radius, color, thickness)

        print("centroid coordinates: ", c1, c2)
        # Write the frame to the output video file
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

        # centroid_data = {'FrameTime(s)': frame_time, 'Centroid': f"{c1},{c2}"}
        # df = df._append(centroid_data, ignore_index=True)

        # Find the corresponding behavior based on frame_time
        matching_row = behavior_df[(behavior_df['Start (s)'] <= frame_time) & (behavior_df['Stop (s)'] >= frame_time)]

        if not matching_row.empty:
            print("Yes")
            behavior = matching_row['Behavior'].values[0]
            print(behavior)
            data = {'FrameTime(s)': frame_time, 'Centroid': f"{c1}, {c2}", 'Behavior': behavior}
            df = df._append(data, ignore_index=True)
            # df.at[df.index[-1], 'Behavior'] = behavior

        else:
            centroid_data = {'FrameTime(s)': frame_time, 'Centroid': f"{c1}, {c2}"}
            df = df._append(centroid_data, ignore_index=True)

# Save the DataFrame to an Excel file
excel_file_path = 'output_data.xlsx'
df.to_excel(excel_file_path, index=False)

print(f"Data saved to {excel_file_path}")

cap.release()
out.release()
cv2.destroyAllWindows()

