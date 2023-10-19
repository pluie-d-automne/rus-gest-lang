"""
Get some easy video files properties
"""

# Import packages:
import cv2
import argparse
import os
import pandas as pd

# Create the ArgumentParser object to parse the command-line arguments.
parser = argparse.ArgumentParser()
df = pd.DataFrame(columns = ['FILENAME',
                             'FRAME_WIDTH' ,
                             'FRAME_HEIGHT',
                             'FPS' ,
                             'FRAME_COUNT']
                  )

# We add 'video_folder' and output_path' argument using add_argument() including a help.
parser.add_argument("video_folder", help="path to the video file")
parser.add_argument("output_path", help="path to the output file")
args = parser.parse_args()

# For each video in the 'video_folder' create a VideoCapture object and read properties
videos = [f for f in os.listdir(args.video_folder) if f != 'description.csv' and 'bad' not in f]
for video_name in videos:
    capture = cv2.VideoCapture(args.video_folder+'/'+ video_name)
    df.loc[len(df)] = [
        video_name,
        capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        capture.get(cv2.CAP_PROP_FPS),
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    ]
    capture.release()
df.to_excel(args.output_path, index=False)