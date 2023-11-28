"""
Split big video files into small ones
"""

# Import packages:
import cv2
import argparse
import os
import pandas as pd
import json

# Create the ArgumentParser object to parse the command-line arguments.
parser = argparse.ArgumentParser()
df = pd.DataFrame(columns = ['FILENAME',
                             'FRAME_WIDTH' ,
                             'FRAME_HEIGHT',
                             'FPS' ,
                             'FRAME_COUNT']
                  )

# Add arguments using add_argument() including a help.
parser.add_argument("annotations_path", help="path to the directory with annotations")
parser.add_argument("video_folder", help="path to the video file")
parser.add_argument("output_folder", help="path to the output folder for short videos")
args = parser.parse_args()

# Get list of all annotations
annotations = [f for f in os.listdir(args.annotations_path) if f != 'description.csv' and 'bad' not in f]

# Calculate distinct annotation episodes for each videofile
translations_list = list()
for filename in annotations:
    translations_dict = dict()
    with open(args.annotations_path+'/'+filename, 'r') as f:
        data = json.load(f)
        for r in data:
            translations_dict[(r[1], r[2])] = r[0]
    translations_list.append([filename, translations_dict])
translations_cnt = sum([len(x[1]) for x in translations_list])
print(f"Общее кол-во слов и файлов, выделенных на основе аннотации: {translations_cnt}")    
    

df = pd.DataFrame(columns = ['FILENAME', 'TRANSLATION'])

# For each videofile create short videos based on the annotations
for filename, translations in translations_list:
    i = 0
    filename = filename.split('.')[0]
    capture = cv2.VideoCapture(f'{args.video_folder}/{filename}.webm')
    
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    #fourcc = capture.get(cv2.CAP_PROP_FOURCC)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    
    # Check if camera opened successfully:
    if capture.isOpened()is False:
        print("Error opening video stream or file")
        
    
    for item in translations.keys():      
        # We get the msec of the first and last frames of the episode, its translation and new filename:
        first_msec = item[0]
        last_msec = item[1]
        translation = translations[item]
        new_filename = f'{filename}_{i}.avi'
        i+=1
        df.loc[len(df)] = [new_filename, translation]
        frame_msec = first_msec
        # Create VideoWriter object. We use the same properties as the input camera.
        # Last argument is True to write the video in color
        out_video = cv2.VideoWriter(
            args.output_folder+'/'+new_filename,
            fourcc,
            int(fps),
            (int(frame_width),
             int(frame_height)
             ),
            True)


        
        # We set the current frame position:
        capture.set(cv2.CAP_PROP_POS_MSEC, frame_msec)
        # Read until video is completed:
        while capture.isOpened() and frame_msec<= last_msec:
            # Capture frame-by-frame from the video file:
            ret, frame = capture.read()
            if ret is True:
                out_video.write(frame)
                # frame_index = capture.get(cv2.CAP_PROP_POS_FRAMES)
                # capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                frame_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
                # Press q on keyboard to exit the program:
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break
        out_video.release()
    capture.release()
    
    
df.to_excel(f'{args.output_folder}/translations.xlsx', index=False)