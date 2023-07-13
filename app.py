from pathlib import Path
import streamlit as st
import cv2
import os
from stqdm import stqdm
from PIL import Image

from argparser import ArgParser
from inference import Detector, count_parameters, load_model_for_inference
from glob import glob


def video2frames(video_path: str, dest: str):
    # print("video_path", video_path, "dest", dest)
    os.makedirs(dest, exist_ok=True)
    video_object = cv2.VideoCapture(video_path)
    frame_index = 1
    total = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in stqdm(range(total)):
        ret, frame = video_object.read()
        if ret == False:
            break
        frame_name = "{}.jpg".format(frame_index)
        cv2.imwrite(os.path.join(dest, frame_name), frame)
        frame_index += 1
    video_object.release()


def fname_without_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


args = ArgParser().parse_args()
args.mot_path = "data/frames/upload"
args.output_dir = "data/outputs/upload"
args.rec = True
args.show = True
if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

with st.spinner("Loading model ..."):
    model = load_model_for_inference(args)
    st.text(f"Number of parameters {count_parameters(model)}")

f = st.file_uploader("Upload video", type=["mp4"])
if f is not None:
    video_path = f"data/videos/upload/{f.name}"
    with open(video_path, "wb") as tfile:
        tfile.write(f.read())

    st.video(video_path)
    fname_wo_ext = fname_without_ext(video_path)
    with st.spinner("1. Extract frames from video ..."):
        video2frames(
            video_path,
            os.path.join("data/frames/upload", fname_wo_ext),
        )

    with st.spinner("2. Inference ..."):
        det = Detector(args, model=model, video_name=fname_wo_ext)
        time_cost = det.detect(vis=args.show)

    with st.spinner("3. Make video from frames"):
        dirpath = f"data/outputs/upload/results/{fname_wo_ext}"
        num_frames = len(glob(f"{dirpath}/*.jpg"))
        os.makedirs("data/outputs/upload/videos", exist_ok=True)
        if num_frames > 0:
            first_frame = Image.open(f"{dirpath}/0.jpg")
            print(first_frame.size)
            vf = cv2.VideoCapture(video_path)
            fps = vf.get(cv2.CAP_PROP_FPS)
            print("fps", fps)
            out_path = f"data/outputs/upload/videos/{fname_wo_ext}.avi"
            final_out_path = f"data/outputs/upload/videos/{fname_wo_ext}.mp4"
            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                fps,
                first_frame.size,
            )
            for i in stqdm(range(num_frames)):
                out.write(cv2.imread(f"{dirpath}/{i}.jpg"))
            out.release()
            vf.release()

            os.system(
                f'ffmpeg -y -i "{out_path}" -vcodec libx264 "{final_out_path}" -hide_banner -loglevel error'
            )
            os.system(f"rm {out_path}")

    st.video(final_out_path)
