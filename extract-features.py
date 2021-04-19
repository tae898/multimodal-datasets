import argparse
import logging
import os
from tqdm import tqdm
import shutil
import requests
import jsonpickle
from PIL import Image
import numpy as np
from glob import glob
import docker
import av
import json
import random
import pickle
from joblib import Parallel, delayed

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class Features():
    def __init__(self, dataset, url_video2frames, url_insightface, face_features,
                 face_videos, visual_features, width_max, height_max, fps_max,
                 audio_features, text_features):
        self.dataset = dataset
        self.url_video2frames = url_video2frames
        self.url_insightface = url_insightface
        self.face_features = face_features
        self.face_videos = face_videos
        self.visual_features = visual_features
        self.width_max = width_max
        self.height_max = height_max
        self.fps_max = fps_max
        self.audio_features = audio_features
        self.text_features = text_features

        if self.face_features:
            self.extract_face_features()

    def extract_face_features(self):
        self._get_video_paths()
        if len(self.video_paths) > 0:
            logging.debug(f"creating face-features directories ...")
            for SPLIT in ['train', 'val', 'test']:
                os.makedirs(f'./{self.dataset}/face-features/{SPLIT}',
                            exist_ok=True)
        for video_path in tqdm(self.video_paths):
            num_wrong = 0
            try:
                frames, metadata = self._video2frames(video_path)
            except Exception as e:
                logging.warning(f"Couldn't process {video_path}!!!")
                num_wrong += 1

            logging.warning(f"There are in total of {num_wrong} videos "
                            f"failed to process.")

    def _get_video_paths(self):
        ext = {'MELD': 'mp4',
               'IEMOCAP': 'mp4'}[self.dataset]
        self.video_paths = glob(f'./{self.dataset}/raw-videos/*/*.{ext}')
        logging.info(
            f"There are in total of {len(self.video_paths)} videos found "
            f"in {self.dataset}")

    def _video2frames(self, video_path):
        logging.debug(f"processing {video_path} ...")
        files = {'video': open(video_path, 'rb')}
        data = {'fps_max': self.fps_max, 'width_max': self.width_max,
                'height_max': self.height_max}
        response = requests.post(
            self.url_video2frames, files=files, data=data)
        response = jsonpickle.decode(response.text)
        frames = response['frames']
        metadata = response['metadata']
        logging.debug(f"metadata of the video is {metadata}")

        return frames, metadata


def main(**kwargs):

    feat = Features(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='extract features from a multimodal dataset')
    parser.add_argument('--dataset', type=str)

    # parser.add_argument('--url-video2frames', type=str,
    #                     default='http://127.0.0.1:10001/extract-frames')

    # parser.add_argument('--url-insightface', type=str,
    #                     default='http://127.0.0.1:10002/face-analysis')
    parser.add_argument('--face-features', action='store_true')
    parser.add_argument('--face-videos', action='store_true')
    parser.add_argument('--visual-features', action='store_true')
    parser.add_argument('--width-max', type=int, default=10000)
    parser.add_argument('--height-max', type=int, default=10000)
    parser.add_argument('--fps-max', type=int, default=10000)

    parser.add_argument('--audio-features', action='store_true')
    parser.add_argument('--text-features', action='store_true')

    args = parser.parse_args()
    args = vars(args)

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
