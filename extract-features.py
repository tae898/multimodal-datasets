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
import av
import json
import random
import pickle
from joblib import Parallel, delayed
import time
from python_on_whales import docker

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def begin_face_features_extraction(dataset, video_paths, video2frames_port,
                                   face_analysis_port, fps_max, width_max,
                                   height_max, video_ext):

    BYTES_AT_LEAST = 256
    for video_path in tqdm(video_paths):
        try:
            SPLIT = video_path.split('/')[-2]
            assert SPLIT in ['train', 'val', 'test']

            basename = os.path.basename(video_path)
            basename_wo_ext = basename.split(f'{video_ext}')[0]

            save_path_metadata = f"./{dataset}/face-features-metadata/{SPLIT}/{basename_wo_ext}.json"
            save_path_face_features = f"./{dataset}/face-features/{SPLIT}/{basename_wo_ext}.pkl"

            if os.path.isfile(save_path_metadata) and \
                os.path.getsize(save_path_metadata) > BYTES_AT_LEAST and \
                    os.path.isfile(save_path_face_features) and \
                    os.path.getsize(save_path_face_features) > BYTES_AT_LEAST:
                logging.info(
                    f"{video_path}, {save_path_metadata}, {save_path_face_features} "
                    f"seems to be already done. skipping ...")
                continue

            with open(video_path, 'rb') as stream:
                binary_video = stream.read()
            data = {'fps_max': fps_max, 'width_max': width_max,
                    'height_max': height_max, 'video': binary_video}
            data = jsonpickle.encode(data)
            response = requests.post(
                f"{'http://127.0.0.1'}:{video2frames_port}/", json=data)
            response = jsonpickle.decode(response.text)
            frames = response['frames']
            metadata = response['metadata']

            with open(save_path_metadata, 'w') as stream:
                json.dump(metadata, stream, indent=4)

            assert len(frames) == len(metadata['frame_idx_original'])

            fa_results_all = []
            for frame, idx in zip(frames, metadata['frame_idx_original']):
                data = {'image': frame}
                data = jsonpickle.encode(data)
                response = requests.post(
                    f"{'http://127.0.0.1'}:{face_analysis_port}/", json=data)

                response = jsonpickle.decode(response.text)
                fa_results = response['fa_results']
                fa_results_all.append(fa_results)

            with open(save_path_face_features, 'wb') as stream:
                pickle.dump(fa_results_all, stream)
        except Exception as e:
            print(f"{e}: something went wrong while processing {video_path}!!! "
                  f"We will skip this file for now.")


class Features():
    def __init__(self, dataset, run_on_gpu):
        self.dataset = dataset
        self.run_on_gpu = run_on_gpu

    def extract_face_features(
            self, port_docker_video2frames, port_docker_face_analysis,
            width_max, height_max, fps_max, num_jobs):

        self.port_docker_video2frames = port_docker_video2frames
        self.port_docker_face_analysis = port_docker_face_analysis
        self.width_max = width_max
        self.height_max = height_max
        self.fps_max = fps_max
        self.num_jobs = num_jobs

        self._get_video_paths()
        if len(self.video_paths) == 0:
            error_msg = f"No videos found! {self.video_paths}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"creating face-features directories ...")
        for SPLIT in ['train', 'val', 'test']:
            os.makedirs(f'./{self.dataset}/face-features/{SPLIT}',
                        exist_ok=True)

        logging.debug(f"creating face-features metadata directories ...")
        for SPLIT in ['train', 'val', 'test']:
            os.makedirs(f'./{self.dataset}/face-features-metadata/{SPLIT}',
                        exist_ok=True)

        self._start_face_containers()
        self._batch_videos()

        logging.debug(f"face features extraction will begin ...")
        Parallel(n_jobs=self.num_jobs)(
            delayed(begin_face_features_extraction)(
                self.dataset, video_paths, video2frames_port, face_analysis_port,
                self.fps_max, self.width_max, self.height_max, self.video_ext)
            for video_paths, video2frames_port, face_analysis_port
            in zip(self.video_paths_batch, self.video2frames_ports, self.face_analysis_ports))

        logging.info(f"face feature extraction complete!")

        self._stop_docker_containers()

    def _get_video_paths(self):
        self.video_ext = {'MELD': '.mp4',
                          'IEMOCAP': '.mp4'}[self.dataset]
        self.video_paths = glob(
            f'./{self.dataset}/raw-videos/*/*{self.video_ext}')

        random.shuffle(self.video_paths)

        logging.info(
            f"There are in total of {len(self.video_paths)} videos found "
            f"in {self.dataset}")

    def _start_face_containers(self):
        self.video2frames_ports = [20000+i for i in range(self.num_jobs)]
        self.face_analysis_ports = [30000+i for i in range(self.num_jobs)]

        self.containers = {}
        logging.info(
            f"Creating {self.num_jobs} containers of video2frames ...")
        image_name = 'video2frames'
        self.containers['video2frames'] = []
        for i in range(self.num_jobs):
            container = \
                docker.run(
                    image=image_name,
                    detach=True,
                    publish=[(self.video2frames_ports[i], self.port_docker_video2frames)])
            self.containers['video2frames'].append(container)
            time_to_sleep = 5
            logging.debug(
                f"sleeping for {time_to_sleep} seconds to warm up {i} th container ...")
            time.sleep(time_to_sleep)
            logging.debug(f"sleeping done")

        logging.info(
            f"Creating {self.num_jobs} containers of face-analysis ...")
        if self.run_on_gpu:
            image_name = 'face-analysis-cuda'
        else:
            image_name = 'face-analysis'
        self.containers['face_analysis'] = []
        for i in range(self.num_jobs):
            container = \
                docker.run(
                    image=image_name,
                    gpus='all',
                    detach=True,
                    publish=[(self.face_analysis_ports[i], self.port_docker_face_analysis)])
            self.containers['face_analysis'].append(container)
            time_to_sleep = 30
            logging.debug(
                f"sleeping for {time_to_sleep} seconds to warm up {i} th container ...")
            time.sleep(time_to_sleep)
            logging.debug(f"sleeping done")

    def _stop_docker_containers(self):
        for image_name, containers in self.containers.items():
            for container in containers:
                logging.info(
                    f"stopping the container {container} of {image_name} ...")
                container.stop()

    def _batch_videos(self):
        logging.debug(f"batching videos into {self.num_jobs} batches ...")
        BATCH_SIZE = len(self.video_paths) // self.num_jobs
        self.video_paths_batch = [self.video_paths[BATCH_SIZE*i:BATCH_SIZE*(i+1)]
                                  for i in range(self.num_jobs)]

        self.video_paths_batch[-1] = self.video_paths_batch[-1] + \
            self.video_paths[self.num_jobs*BATCH_SIZE:]
        assert set(self.video_paths) == set(
            [bar for foo in self.video_paths_batch for bar in foo])
        logging.info(f"batching done. each batch has "
                     f"{len(self.video_paths_batch[0])} videos.")

    def extract_face_videos(self):
        raise NotImplementedError(f"next time, baby")

    def extract_visual_features(self):
        raise NotImplementedError(f"next time, baby")

    def extract_audio_features(self):
        raise NotImplementedError(f"next time, baby")

    def extract_text_features(self):
        raise NotImplementedError(f"next time, baby")


def main(dataset, port_docker_video2frames, port_docker_face_analysis, width_max,
         height_max, fps_max, num_jobs, face_features, face_videos,
         visual_features, audio_features, text_features, run_on_gpu):

    ft = Features(dataset, run_on_gpu)

    if face_features:
        kwargs = {'port_docker_video2frames': port_docker_video2frames,
                  'port_docker_face_analysis': port_docker_face_analysis,
                  'width_max': width_max,
                  'height_max': height_max,
                  'fps_max': fps_max,
                  'num_jobs': num_jobs}
        ft.extract_face_features(**kwargs)

    if face_videos:
        ft.extract_face_videos()

    if visual_features:
        ft.extract_visual_features()

    if audio_features:
        ft.extract_audio_features()

    if text_features:
        ft.extract_text_features()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='extract features from a multimodal dataset')
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--port-docker-video2frames', type=int,
                        default=10001)
    parser.add_argument('--port-docker-face-analysis', type=int,
                        default=10002)

    parser.add_argument('--face-features', action='store_true')
    parser.add_argument('--face-videos', action='store_true')
    parser.add_argument('--visual-features', action='store_true')
    parser.add_argument('--width-max', type=int, default=10000)
    parser.add_argument('--height-max', type=int, default=10000)
    parser.add_argument('--fps-max', type=int, default=10000)

    parser.add_argument('--audio-features', action='store_true')
    parser.add_argument('--text-features', action='store_true')

    parser.add_argument('--num-jobs', type=int, default=1)
    parser.add_argument('--run-on-gpu', action='store_true')

    args = parser.parse_args()
    args = vars(args)

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
