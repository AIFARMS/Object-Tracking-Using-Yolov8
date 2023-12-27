
# from moviepy.editor import *

# # loading video gfg
# clip = VideoFileClip("/Users/jenish/Desktop/000112.mp4")
# # getting only first 5 seconds
# clip = clip.subclip(0, 5*60)
# # showing clip
# clip.write_videofile("12151_112.mp4")

import cv2

# Define the input video path
input_video_path = '/Users/jenish/Desktop/000111.mp4'

# Define the output video path
output_video_path = '/Users/jenish/Desktop/12151_111.mp4'

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the number of frames for the first 5 minutes
frames_to_extract = int(fps * 60 * 5)

# Create VideoWriter object for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Read and write the first 5 minutes of frames
for i in range(frames_to_extract):
    ret, frame = cap.read()
    if ret:
        out.write(frame)
    else:
        break

# Release the video capture and writer objects
cap.release()
out.release()

print(f"First 5 minutes extracted and saved to {output_video_path}")
