from glob import glob
import argparse
from joblib import Parallel, delayed
import shutil
import json
from sklearn.cluster import AgglomerativeClustering
import random
import pickle
import numpy as np
from tqdm import tqdm
import os
import coolname
import uuid
import shutil
from PIL import Image
import logging
import io
import jsonpickle
from python_on_whales import docker
import requests
import time

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def get_existing_path(DATASET, modality, SPLIT, uttid,
                      extensions):
    candidates = [os.path.join(DATASET, modality, SPLIT, uttid + extension)
                  for extension in extensions]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def get_unique_faces(embeddings, face_cos_distance_threshold):
    logging.debug(f"finding unique faces ...")

    if len(embeddings) == 1:
        labels_clustered = np.array([0])

    elif len(embeddings) == 0:
        return None

    else:
        ac = AgglomerativeClustering(n_clusters=None,
                                     affinity='cosine',
                                     linkage='average',
                                     distance_threshold=face_cos_distance_threshold)

        clustering = ac.fit(embeddings)
        labels_clustered = clustering.labels_

    labels_unique = np.unique(labels_clustered)
    while True:
        names_unique = [coolname.generate_slug(2) for _ in labels_unique]
        if len(labels_unique) == len(names_unique):
            break

    label2name = {lbl: nm for lbl, nm in zip(labels_unique, names_unique)}
    face_names = [label2name[lbl] for lbl in labels_clustered]

    return face_names


def run_parallel(emissor_object):
    emissor_object.process_dias()


class DataSet():
    def __init__(self, dataset):
        logging.debug(f"initializing {dataset} ...")
        emotion_path = f"./{dataset}/emotions.json"
        if os.path.isfile(emotion_path):
            logging.info(f"labeled emotions found at {emotion_path}")
            with open(emotion_path, 'r') as stream:
                self.emotions = json.load(stream)
        else:
            logging.info(f"labeled emotions not found")
            self.emotions = None

        self.dataset = dataset
        self.SPLITS = list(set([dir.split('/')[-1] for dir
                                in glob(f"./{self.dataset}/raw-*/*")]))

        logging.info(f"{self.SPLITS} splits found.")

        utterance_ordered_path = f"./{self.dataset}/utterance-ordered.json"

        if os.path.isfile(utterance_ordered_path):
            logging.info(f"ordered utterances in dialgoues found "
                         f"at {utterance_ordered_path}")
            with open(utterance_ordered_path, 'r') as stream:
                self.utterance_ordered = json.load(stream)
        else:
            logging.info(f"ordered utterances not found. Every utterance "
                         f"is considered to be one dialogue.")
            uttids = {SPLIT: glob(f"./{self.dataset}/raw-texts/{SPLIT}/*.json")
                      for SPLIT in self.SPLITS}
            self.utterance_ordered = \
                {SPLIT: {uttid: uttid for uttid in uttids[SPLIT]}
                 for SPLIT in self.SPLITS}

    def _batch_diaids(self):
        diaids = [f"{SPLIT}/{diaid}" for SPLIT, diaids
                  in self.utterance_ordered.items() for diaid in diaids]
        random.shuffle(diaids)
        logging.info(f"in total of {len(diaids)} dialogues found")
        logging.info(f"batching dialgoues into {self.num_jobs} batches ...")
        self.num_jobs = min(self.num_jobs, len(diaids))

        BATCH_SIZE = len(diaids) // self.num_jobs
        logging.debug(
            f"every batch will have {BATCH_SIZE} samples (dialogues)")

        self.diaids_batched = [diaids[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                               for i in range(self.num_jobs)]

        self.diaids_batched[-1] += diaids[self.num_jobs*BATCH_SIZE:]

        assert set(diaids) == set(
            [diaid for batch in self.diaids_batched for diaid in batch])

    def run_emissor(self, num_jobs, face_prob_threshold, face_cos_distance_threshold,
                    width_max, height_max, fps_max, port_docker_video2frames):

        self.num_jobs = num_jobs
        self._batch_diaids()

        ems = [Emissor(self.dataset, batch, 20000+idx, port_docker_video2frames,
                       face_prob_threshold, face_cos_distance_threshold,
                       width_max, height_max, fps_max, self.utterance_ordered,
                       self.emotions)
               for idx, batch in enumerate(self.diaids_batched)]

        Parallel(n_jobs=self.num_jobs)(delayed(run_parallel)(em) for em in ems)

        for em in ems:
            em.container.stop()


class Emissor():

    def __init__(self, dataset, diaids, video2frames_port,
                 port_docker_video2frames, face_prob_threshold,
                 face_cos_distance_threshold, width_max, height_max, fps_max,
                 utterance_ordered, emotions):
        self.dataset = dataset
        self.diaids = diaids

        self.video2frames_port = video2frames_port
        logging.debug(f"creating video2frames container, {self.video2frames_port} "
                      f"to  {port_docker_video2frames}...")
        self.container = \
            docker.run(image='tae898/video2frames', detach=True,
                       publish=[(self.video2frames_port, port_docker_video2frames)])
        logging.info(f"video2frames container created!")
        time_to_sleep = 5
        logging.debug(
            f"sleeping for {time_to_sleep} seconds to warm up the container ...")
        time.sleep(time_to_sleep)
        logging.debug(f"sleeping done")

        self.face_prob_threshold = face_prob_threshold
        self.face_cos_distance_threshold = face_cos_distance_threshold
        self.width_max = width_max
        self.height_max = height_max
        self.fps_max = fps_max
        self.utterance_ordered = utterance_ordered
        self.emotions = emotions

        self._get_paths()

    def _get_paths(self):
        logging.debug(f"finding the signal (data) paths ...")
        self.paths = {}
        exts = {'raw-videos': ['.mp4', '.avi'],
                'raw-audios': ['.mp3', '.wav'],
                'raw-texts': ['.json'],
                'face-features': ['.pkl'],
                'face-features-metadata': ['.json']}

        for modality in list(exts.keys()):
            modality_path = f"./{self.dataset}/{modality}/"
            if os.path.isdir(modality_path):
                logging.info(f"{modality} found at {modality_path}")
                self.paths[modality] = \
                    {SPLIT: {uttid: get_existing_path(self.dataset, modality,
                                                      SPLIT, uttid, exts[modality])
                             for diaid, uttids in diauttid.items()
                             for uttid in uttids}
                     for SPLIT, diauttid in self.utterance_ordered.items()}
            else:
                logging.info(f"no {modality} for self.dataset")
                self.paths[modality] = None

    def process_dias(self):
        logging.info(f"Processing {len(self.diaids)} dialogues ...")
        for diaid in tqdm(self.diaids):
            self.process_dia(diaid)
        logging.info(f"Processing {len(self.diaids)} dialogues done!")

        logging.debug(f"stopping the video2frames docker container ...")
        self.container.stop()

    def process_dia(self, diaid):
        """Dialogue (session) level.

        Note that a dialogue (session) contains at least one utterance.

        """
        SPLIT, diaid = diaid.split('/')[0], diaid.split('/')[1]
        chat_emissor_dia = []
        image_emissor_dia = []

        if self.paths['face-features'] is not None:
            self.face_recognition_dia(SPLIT, diaid)

        starttime_msec = 0
        for uttid in self.utterance_ordered[SPLIT][diaid]:
            frames, duration_msec, fps_original, face_features, frame_idx_original = \
                self.load_images_utt(SPLIT, uttid)
            if self.emotions is not None:
                emotion = self.emotions[SPLIT][uttid]
            if face_features is not None:
                image_emissor_utt = self.annotate_write_frames(
                    starttime_msec, SPLIT, diaid, uttid, emotion, frames,
                    fps_original, face_features, frame_idx_original)
                for image_emissor_utt_frame in image_emissor_utt:
                    image_emissor_dia.append(image_emissor_utt_frame)

            try:
                with open(self.paths['raw-texts'][SPLIT][uttid], 'r') as stream:
                    text = json.load(stream)
                chat_emissor_dia.append(
                    [text['Speaker'], text['Utterance'], starttime_msec])
            except Exception as e:
                logging.warning(f"{e}: no text information for {uttid}")

            # in case loading the video was not successful, we just add one
            # second (1000ms).
            if duration_msec is None:
                duration_msec = 1000

            starttime_msec += duration_msec

        if len(chat_emissor_dia) != 0:
            self.write_chat_emissor(SPLIT, diaid, chat_emissor_dia)

        if len(image_emissor_dia) != 0:
            self.write_image_emissor(SPLIT, diaid, image_emissor_dia)

    def face_recognition_dia(self, SPLIT, diaid):
        logging.info(f"running face recognition on {diaid}")
        face_features_dia = {}
        face_metadata_dia = {}
        embs_dia = {}

        logging.debug(f"{diaid} has {len(self.utterance_ordered[SPLIT][diaid])} "
                      f"utterance(s)")
        for uttid in self.utterance_ordered[SPLIT][diaid]:
            try:
                face_features_path = self.paths['face-features'][SPLIT][uttid]
                face_metadata_path = self.paths['face-features-metadata'][SPLIT][uttid]

                with open(face_features_path, 'rb') as stream:
                    face_features = pickle.load(stream)

                with open(face_metadata_path, 'r') as stream:
                    face_metadata = json.load(stream)

                assert len(face_features) == face_metadata['num_frames_original'], \
                    f"something ain't right."

            except Exception as e:
                logging.warning(f"{e}: no face info for {uttid}!")
                continue

            face_features_dia[uttid] = face_features
            face_metadata_dia[uttid] = face_metadata

            embs_utt = [[f['normed_embedding'] for f in ff]
                        for ff in face_features]
            embs_dia[uttid] = embs_utt

        # Changed in version 3.7: Dictionary order is guaranteed to be insertion order.
        # This behavior was an implementation detail of CPython from 3.6.
        embs_unpacked = []
        for i, (uttid, embs_utt) in enumerate(embs_dia.items()):
            for j, embs in enumerate(embs_utt):
                for k, emb in enumerate(embs):
                    embs_unpacked.append(emb)

        unique_faces_unpacked = get_unique_faces(embs_unpacked,
                                                 self.face_cos_distance_threshold)

        if unique_faces_unpacked is None:
            self.face_names_dia = None
        else:
            self.face_names_dia = {}
            count = 0
            for i, (uttid, embs_utt) in enumerate(embs_dia.items()):
                self.face_names_dia[uttid] = []
                for j, embs in enumerate(embs_utt):
                    faces_per_frame = []
                    for k, emb in enumerate(embs):
                        unique_name = unique_faces_unpacked[count]
                        faces_per_frame.append(unique_name)
                        count += 1
                    self.face_names_dia[uttid].append(faces_per_frame)

            assert len(embs_unpacked) == len(unique_faces_unpacked) == count

    def load_images_utt(self, SPLIT, uttid):
        """Utterance level. This is the most atomic level."""
        if self.paths['raw-videos'] is not None:
            video_path = self.paths['raw-videos'][SPLIT][uttid]
        else:
            video_path = None
        frames, duration_msec, fps_original, face_features, frame_idx_original = \
            None, None, None, None, None

        if video_path is not None:

            try:
                with open(video_path, 'rb') as stream:
                    binary_video = stream.read()

                data = {'fps_max': self.fps_max, 'width_max': self.width_max,
                        'height_max': self.height_max, 'video': binary_video}
                data = jsonpickle.encode(data)

                response = requests.post(
                    f"{'http://127.0.0.1'}:{self.video2frames_port}/", json=data)
                response = jsonpickle.decode(response.text)
                frames = response['frames']
                metadata = response['metadata']

                frame_idx_original = metadata['frame_idx_original']

                assert len(frames) == len(frame_idx_original)

                logging.debug(f"decompressing frames ...")
                frames = [Image.open(io.BytesIO(frame)) for frame in frames]
                duration_msec = metadata['duration_seconds'] * 1000
                fps_original = metadata['fps_original']
            except Exception as e:
                frames, duration_msec, fps_original, face_features, frame_idx_original = \
                    None, None, None, None, None
                logging.error(f"{e}: no video information on {video_path}!")

        if frames is not None:
            face_features_path = self.paths['face-features'][SPLIT][uttid]
            if video_path is not None and face_features_path is not None:
                with open(face_features_path, 'rb') as stream:
                    face_features = pickle.load(stream)
                face_features = [face_features[idx]
                                 for idx in metadata['frame_idx_original']]
                assert len(frames) == len(
                    face_features), f"{len(frames)}, {len(face_features)}"

                face_features = [[ff for ff in face_feature
                                  if ff['det_score'] > self.face_prob_threshold]
                                 for face_feature in face_features]

        return frames, duration_msec, fps_original, face_features, frame_idx_original

    def annotate_write_frames(self, starttime_msec, SPLIT, diaid, uttid,
                              emotion, frames, fps_original, face_features,
                              frame_idx_original):
        image_emissor_utt = []

        assert len(face_features) == len(frame_idx_original) == len(frames)

        for idx, frame, ff in zip(frame_idx_original, frames, face_features):
            frame_time = int(round(starttime_msec + idx / fps_original*1000))
            os.makedirs(os.path.join(
                self.dataset, 'emissor', SPLIT, diaid, 'image'),
                exist_ok=True)
            save_path = os.path.join(
                self.dataset, 'emissor', SPLIT, diaid, 'image',
                uttid + f'_frame{str(idx)}_{str(frame_time)}.jpg')
            frame.save(save_path)

            frame_info = {}
            frame_info['files'] = [os.path.join(
                'image', os.path.basename(save_path))]
            container_id = str(uuid.uuid4())
            frame_info['id'] = container_id
            frame_info['mentions'] = []

            frame_info['modality'] = 'image'
            frame_info['ruler'] = {'bounds': [0, 0, frame.size[0],
                                              frame.size[1]],
                                   'container_id': container_id,
                                   'type': 'MultiIndex'}
            frame_info['time'] = {'container_id': container_id,
                                  'start': frame_time,
                                  'end': frame_time + int(round(1000/fps_original)),
                                  'type': 'TemporalRuler'}
            frame_info['type'] = 'ImageSignal'

            for i, feat in enumerate(ff):
                # age / gender estimation is too poor.
                age = feat['age']['mean']
                gender = feat['gender']['m']
                gender = 'male' if gender > 0.5 else 'female'
                bbox = [int(num) for num in feat['bbox'].tolist()]
                faceprob = round(feat['det_score'], 3)
                name = self.face_names_dia[uttid][idx][i]

                annotations = []
                # emotions are not displayed nicely at the moment.
                # if k == 0:
                #     annotations.append({'source': 'human',
                #                         'timestamp': frame_time,
                #                         'type': 'emotion',
                #                         'value': emotion})

                annotations.append({'source': 'machine',
                                    'timestamp': frame_time,
                                    'type': 'person',
                                    'value': {'name': name,
                                              'age': age,
                                              'gender': gender,
                                              'faceprob': faceprob}})

                mention_id = str(uuid.uuid4())
                segment = [{'bounds': bbox,
                            'container_id': container_id,
                            'type': 'MultiIndex'}]
                frame_info['mentions'].append({'annotations': annotations,
                                               'id': mention_id,
                                               'segment': segment})
            image_emissor_utt.append(frame_info)

        return image_emissor_utt

    def write_image_emissor(self, SPLIT, diaid, image_emissor_dia):
        os.makedirs(os.path.join(
            self.dataset, 'emissor', SPLIT, diaid, 'image'), exist_ok=True)

        with open(os.path.join(self.dataset, 'emissor', SPLIT, diaid,
                               'image.json'), 'w') as stream:
            json.dump(image_emissor_dia, stream, indent=4)

    def write_chat_emissor(self, SPLIT, diaid, chat_emissor_dia):
        os.makedirs(os.path.join(self.dataset, 'emissor', SPLIT, diaid, 'text'),
                    exist_ok=True)
        with open(os.path.join(self.dataset, 'emissor', SPLIT,
                               diaid, 'text', f"{diaid}.csv"), 'w') as stream:
            stream.write('speaker,utterance,time\n')

            for line in chat_emissor_dia:
                speaker, utterance, starttime_msec = line
                stream.write(speaker)
                stream.write(',')
                stream.write(f"\"{utterance}\"")
                stream.write(',')
                stream.write(str(starttime_msec))
                stream.write('\n')


def main(dataset, num_jobs, face_prob_threshold, face_cos_distance_threshold,
         width_max, height_max, fps_max, port_docker_video2frames):

    ds = DataSet(dataset)
    ds.run_emissor(num_jobs, face_prob_threshold, face_cos_distance_threshold,
                   width_max, height_max, fps_max, port_docker_video2frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='annotate a dataset in the '
                                     'emissor format.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num-jobs', default=1,
                        type=int, help='number of jobs')
    parser.add_argument('--face-prob-threshold', default=0.9,
                        type=float, help='face detection probability threshold')
    parser.add_argument('--face-cos-distance-threshold', default=0.8,
                        type=float, help='cosine distance threshold. The '
                        'linkage distance threshold above which, clusters '
                        'will not be merged.')

    parser.add_argument('--width-max', type=int, default=10000)
    parser.add_argument('--height-max', type=int, default=10000)
    parser.add_argument('--fps-max', type=int, default=1, help='fps for the '
                        'image modality')

    parser.add_argument('--port-docker-video2frames', type=int,
                        default=10001)

    args = parser.parse_args()
    args = vars(args)
    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
