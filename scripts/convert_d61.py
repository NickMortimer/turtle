import pandas as pd
from pathlib import Path
import cv2


source_path = '/home/mor582/Downloads/video_select2_filtered.txt'

def extract_detail(file_path):
    # Initialize lists to store the extracted data
    video_name = r'^Video: (?P<VideoName>.+)'
    video_pattern = r'(?P<Id>.+?-.+?_.+?-.+?-.+)-(?P<Date>\d{6})_(?P<Time>\d{6})_.+MP4'
    time_pattern = r'^Time: (?P<Elapsed_Time>\d{1}:\d{2}:\d{2}).+Detected: (?P<Detected>.+)'
    data = pd.read_csv(file_path,sep='\t',skiprows=2,header=None,names=['Line'])
    data['VideoName'] = data.Line.str.extract(video_name).ffill()
    data[['Id','Date','Time']] =data.VideoName.str.extract(video_pattern)
    data[['Elapsed_Time','Detected']] =data.Line.str.extract(time_pattern)
    data['Elapsed_Time']=pd.to_timedelta(data['Elapsed_Time'])
    data['TimeStamp'] = pd.to_datetime(data.Date + ' ' + data.Time,format='%y%m%d %H%M%S').ffill()
    data['CalculatedTime'] = data['TimeStamp'] + data['Elapsed_Time']
    return data

def extract_frames(video_path, center_time,delta):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames =[]
    for frame_time in range(center_time-delta,center_time+delta):
        # Calculate the frame number corresponding to the time
        frame_number = int(fps * frame_time)
        # Set the video to the frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # Read the frame
        ret, frame = cap.read()
        frames.append({'time':frame_number,'frame':frame})
    cap.release()
    return frames

# Define the path to the text file
file_path = Path(source_path)
output =extract_detail(file_path)
output.to_csv(file_path.with_suffix('.csv'),index=False)
video_path = Path('/home/mor582/')
frame_output = Path('/home/mor582/frames')
output
for index,row in output.iterrows():
    source_video = (video_path  / row.VideoName).with_suffix('')
    frames = extract_frames(source_video,row.Elapsed_Time,2)

    frame_path = source_video / f'{row.CalculatedTime.strftime("%Y%m%dT%H%M")}'
    for frame in frames
