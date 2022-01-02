import os
import cv2
import pandas as pd

phases = ['Train', 'Test', 'Validation']

df = pd.DataFrame()


for phase in phases:
    path = os.path.join('/content/DAiSEE2/DAiSEE/DataSet/', phase)
    subjects = os.listdir(path)
    for subject in subjects:
        print(phase, subject, flush=True)
        videos = os.listdir(os.path.join(path, subject))
        for video in videos:
            videoPath = os.path.join(path, subject, video, os.listdir(os.path.join(path, subject, video))[0])
            videoPathdf= os.path.join(path, subject, video)
            videoPathFrames = '/'.join(videoPath.split('.')[0].split('/')[:-1]).replace(phase, phase + 'Frames')
            os.makedirs(videoPathFrames, exist_ok=True)
            print(videoPathdf)
            df = df.append({'path' : videoPathdf},ignore_index=True)
            df.to_csv("/content/ResNet-TCN/labels_daisee.csv")

            capture = cv2.VideoCapture(videoPath)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            count = 0
            i = 0
            retaining = True
            while count < frame_count and retaining:
                retaining, frame = capture.read()
                if frame is None:
                    continue
                cv2.imwrite(filename=os.path.join(videoPathFrames, '{}.jpg'.format(str(i))), img=frame)
                i += 1
                count += 1
            capture.release()
