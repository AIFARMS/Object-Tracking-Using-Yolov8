import glob
import os
import cv2


Datafolder = '/Users/jenish/Desktop/PigLife_CVModels/Tracking-and-counting-Using-YOLOv8-and-DeepSORT/raw_videos/'
Exportfolder = 'extracted_images'
if not os.path.exists(Exportfolder):
    os.makedirs(Exportfolder)

Vilst = glob.glob(Datafolder + "*.mp4", recursive=False)
record = list()
for vname in Vilst:
    name = vname.split(os.sep)[-1]
    print(name)
    vnamess = name.split('.')[0]
    vidcap = cv2.VideoCapture(vname)
    count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    print(video_fps, count)
    for i in range(0, count):
        success, image = vidcap.read()
        # if i % video_fps == 0:
        if i%25 == 0 and success:
            new_name = "{}({}).png".format(vnamess,i)
            cv2.imwrite(os.path.join(Exportfolder, Exportfolder + new_name), image)
    vidcap.release()