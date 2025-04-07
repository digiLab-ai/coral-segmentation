import os

import argparse
import cv2

def get_frames_from_video(video_path: str, img_path: str, sampling_rate: int = 10, downsampling_factor: float = 0.3) -> None:
    """
    Function to pre-process and to obtain a set of frames from the video

    Parameters
    __________
    video_path: str
        Path to the input video
    img_path: str
        Path to store the pre-processed images
    sampling_rate: int
        Temporal Sampling Frequency.
        Specified as the number of frames out of which one is picked.
    downsampling_factor: float
        Spatial Downsampling factor to reduce the resolution of the image.
        Specified as a float in [0, 1].

    Returns
    _______
    None
    """
    capture = cv2.VideoCapture(video_path)

    video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    n_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Number of frames: ", video_length)

    frameNr = 0

    while True:

        success, frame = capture.read()
        frameNr = frameNr + 1

        if frameNr == 1:
            if not os.path.exists(img_path):
                os.mkdir(img_path)

        if success:
            if frameNr < n_frames:
                if frameNr % sampling_rate == 0:
                    resized_frame = cv2.resize(frame, (0, 0), fx = downsampling_factor, fy = downsampling_factor)
                    cv2.imwrite(
                        os.path.join(img_path, f"frame_{frameNr}.jpg"), resized_frame
                    )
            else:
                break

    print("Sampled Images from Video and copied into the specified directory")

    capture.release()

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, help="Input Video Path")
parser.add_argument("--img_path", type=str, default="imgs", help="Output path to store images")
parser.add_argument("--sampling_rate", type=int, default=10, help="Temporal Sampling Rate")
parser.add_argument("--downsampling_factor", type=float, default=0.3, help="Spatial Downsampling Factor")

args = parser.parse_args()

video_path = args.video_path
img_path = args.img_path
sampling_rate = args.sampling_rate
downsampling_factor = args.downsampling_factor

get_frames_from_video(video_path=video_path, img_path=img_path, sampling_rate=sampling_rate, downsampling_factor=downsampling_factor)



