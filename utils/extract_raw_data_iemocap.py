from glob import glob
from tqdm import tqdm
import os
import json
import librosa
import av
import shutil
import numpy as np
import math
import json
from .audio import add_audio_on_video


def get_filepaths():
    videopaths = glob('./IEMOCAP/IEMOCAP_full_release/*/dialog/avi/DivX/*.avi')
    audiopaths = glob('./IEMOCAP/IEMOCAP_full_release/*/sentences/wav/*/*.wav')
    textpaths = glob(
        './IEMOCAP/IEMOCAP_full_release/*/dialog/transcriptions/*.txt')

    return videopaths, audiopaths, textpaths


def move_raw_files():
    videopaths, audiopaths, textpaths = get_filepaths()
    for paths, modality in tqdm(zip([videopaths, audiopaths, textpaths],
                                    ['raw-videos', 'raw-audios', 'raw-texts'])):
        for path in tqdm(paths):
            basename = os.path.basename(path)
            os.rename(path, f"./IEMOCAP/{modality}/{basename}")


def sort_audios():
    os.chdir('IEMOCAP/raw-audios')

    for filename in glob('./*.wav'):
        DIR = '_'.join(filename.split('_')[:-1])
        os.makedirs(DIR, exist_ok=True)
        os.rename(filename, os.path.join(DIR, filename))

    os.chdir('../../')
    print(f"current directory: {os.getcwd()}")


def make_splits():
    dialogxl_videoids = {}
    for SPLIT in ['train', 'val', 'test']:
        jsonpath = f'./utils/IEMOCAP-DialogXL/{SPLIT}_data.json'
        if SPLIT == 'val':
            jsonpath = f'./utils/IEMOCAP-DialogXL/dev_data.json'
        with open(jsonpath, 'r') as stream:
            dialogxl_videoids[SPLIT] = json.load(stream)

    for SPLIT in ['train', 'val', 'test']:
        dialogxl_videoids[SPLIT] = \
            [f['speaker'] for fo in dialogxl_videoids[SPLIT] for f in fo]

    for SPLIT in ['train', 'val', 'test']:
        dialogxl_videoids[SPLIT] = \
            sorted(list(set(['_'.join(foo.split('_')[:-1])
                             for foo in dialogxl_videoids[SPLIT]])))

    os.chdir('IEMOCAP/raw-texts')

    for foo in glob('*.txt'):

        belongsto = None
        for bar in ['train', 'val', 'test']:
            if os.path.basename(foo).split('.txt')[0] in dialogxl_videoids[bar]:
                belongsto = bar

        assert belongsto is not None

        os.rename(foo, os.path.join(belongsto, os.path.basename(foo)))

    os.chdir('../raw-videos')

    for foo in glob('*.avi'):

        belongsto = None
        for bar in ['train', 'val', 'test']:
            if os.path.basename(foo).split('.avi')[0] in dialogxl_videoids[bar]:
                belongsto = bar

        assert belongsto is not None

        os.rename(foo, os.path.join(belongsto, os.path.basename(foo)))

    os.chdir('../raw-audios')
    for foo in glob('*'):
        if foo in ['train', 'val', 'test']:
            continue

        belongsto = None
        for bar in ['train', 'val', 'test']:
            if os.path.basename(foo) in dialogxl_videoids[bar]:
                belongsto = bar

        assert belongsto is not None, f"{foo}, {bar}"

        os.rename(foo, os.path.join(belongsto, os.path.basename(foo)))

    os.chdir('../../')
    print(f"current directory: {os.getcwd()}")


def clean_texts():
    os.chdir('IEMOCAP/raw-texts')
    weird = 0
    durations = {'train': {}, 'val': {}, 'test': {}}
    diautt_ordered = {'train': {}, 'val': {}, 'test': {}}
    for foo in glob('*/*.txt'):
        SPLIT = foo.split('/')[0]
        dia = foo.split('/')[1].split('.txt')[0]
        os.makedirs(os.path.join(SPLIT, dia), exist_ok=True)
        durations[SPLIT][dia] = {}
        diautt_ordered[SPLIT][dia] = []

        with open(foo, 'r') as stream:
            lines = [line.strip() for line in stream.readlines()]
        for line in lines:
            tokens = line.split(' ')

            try:
                utterance_id = tokens[0]
                assert '_' in utterance_id
                assert 'XX' not in utterance_id
                speaker = utterance_id.split('_')[-1]
                speaker = ''.join([i for i in speaker if not i.isdigit()])
                dur = tokens[1]
                dur = dur.split(':')[0]
                dur = dur.split('-')
                start = float(dur[0].split('[')[1])
                end = float(dur[1].split(']')[0])

                durations[SPLIT][dia][utterance_id] = (start, end)
                utterance = ' '.join(tokens[2:])
                utterance = utterance.strip()
                print(utterance)
                diautt_ordered[SPLIT][dia].append(utterance_id)

                to_dump = {'Utterance': utterance,
                           'StartTime': start,
                           'EndTime': end,
                           'Speaker': {'M': 'Male', 'F': 'Female'}[speaker]}
                with open(os.path.join(SPLIT, dia, utterance_id + '.json'), 'w') as stream:
                    json.dump(to_dump, stream, indent=4)

            except:
                weird += 1
                pass

        os.remove(foo)
    os.chdir('../')

    diautt_ordered_ = {}

    for SPLIT in ['train', 'val', 'test']:
        diautt_ordered_[SPLIT] = {}
        for dia, utts in diautt_ordered[SPLIT].items():
            diautt_ordered_[SPLIT][dia] = []

            for utt in utts:
                path_to_check = os.path.join(
                    f'raw-audios/{SPLIT}/{dia}/{utt}.wav')
                print(path_to_check)
                if os.path.isfile(path_to_check):
                    diautt_ordered_[SPLIT][dia].append(utt)

    for SPLIT in ['train', 'val', 'test']:
        for dia, utts in diautt_ordered_[SPLIT].items():
            assert len(utts) == len(glob(f"raw-audios/{SPLIT}/{dia}/*.wav"))

    with open('utterance-ordered.json', 'w') as stream:
        json.dump(diautt_ordered_, stream, indent=4, sort_keys=True)

    print(f"There are in total of {weird} weird utterances. It's fine I went "
          f"through them. Nothing serious")
    os.chdir('../')
    print(f"current directory: {os.getcwd()}")


def slice_videos():
    with open('IEMOCAP/utterance-ordered.json', 'r') as stream:
        diautt_ordered = json.load(stream)

    dias = {SPLIT: sorted(os.listdir(f'IEMOCAP/raw-videos/{SPLIT}/'))
            for SPLIT in ['train', 'val', 'test']}

    dias = {SPLIT: [foo.split('.avi')[0] for foo in dias[SPLIT]]
            for SPLIT in ['train', 'val', 'test']}

    SAMPLING_RATE = 2**12

    # This magic value was emperically chosen.
    OFFSET_FRAMES = -5

    def video2numpy(path):
        container = av.open(path)
        frames = []
        for idx, frame in enumerate(container.decode(video=0)):
            numpy_RGB = np.array(frame.to_image())
            frames.append(numpy_RGB)
        container.close()
        return frames

    def get_start_end_sec(audio_utt_path):
        text_utt_path = audio_utt_path.replace(
            'raw-audios', 'raw-texts').replace('.wav', '.json')
        text_utt_path = text_utt_path

        with open(text_utt_path, 'r') as stream:
            text_utt = json.load(stream)

        start = text_utt['StartTime']
        end = text_utt['EndTime']

        return start, end

    def write_video(video_utt, savepath, fps):
        container = av.open(savepath, mode='w')
        stream = container.add_stream('mpeg4', rate=round(fps))
        stream.width = video_utt[0].shape[1]
        stream.height = video_utt[0].shape[0]

        for frame in video_utt:
            frame_ = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame_):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

        # Close the file
        container.close()

    def process_dia(SPLIT, dia):
        video_dia = video2numpy(f"IEMOCAP/raw-videos/{SPLIT}/{dia}.avi")
        audio_dia = librosa.core.load(
            f"IEMOCAP/raw-videos/{SPLIT}/{dia}.avi", sr=SAMPLING_RATE)[0]

        # start_from = 0
        for utt in diautt_ordered[SPLIT][dia]:
            uttwav = os.path.join(
                f"IEMOCAP/raw-audios/{SPLIT}/{dia}/{utt}.wav")
            audio_utt = librosa.core.load(uttwav, sr=SAMPLING_RATE)[0]

            start, end = get_start_end_sec(uttwav)
            start_frame = math.floor(
                len(video_dia) * start / (len(audio_dia) / SAMPLING_RATE))
            end_frame = math.ceil(len(video_dia) * end /
                                  (len(audio_dia) / SAMPLING_RATE))

            start_frame += OFFSET_FRAMES
            end_frame += OFFSET_FRAMES

            # number of frames divided by the duration in seconds
            fps = (end_frame - start_frame) / (len(audio_utt) / SAMPLING_RATE)
            video_utt = video_dia[start_frame:end_frame]

            os.makedirs(
                f"IEMOCAP/raw-videos/{SPLIT}/{dia}", exist_ok=True)

            uttmp4 = os.path.basename(uttwav).replace('.wav', '.mp4')
            save_path = f"IEMOCAP/raw-videos/{SPLIT}/{dia}/{uttmp4}"

            write_video(video_utt, save_path, fps)

    for SPLIT in tqdm(['train', 'val', 'test']):
        for dia in tqdm(dias[SPLIT]):
            process_dia(SPLIT, dia)

    for avipath in tqdm(glob('./IEMOCAP/raw-videos/*/*.avi')):
        os.remove(avipath)


def add_audios_to_videos():
    for videopath in tqdm(glob('./IEMOCAP/raw-videos/*/*/*.mp4')):
        audiopath = videopath.replace('raw-videos', 'raw-audios')
        audiopath = audiopath.replace('.mp4', '.wav')
        assert os.path.isfile(audiopath)

        add_audio_on_video(videopath, audiopath)


def clean_files():
    for modality in ['raw-videos', 'raw-audios', 'raw-texts']:
        extension = {'raw-videos': '.mp4',
                     'raw-audios': '.wav',
                     'raw-texts': '.json'}[modality]
        for filepath in glob(f"./IEMOCAP/{modality}/*/*/*{extension}"):
            newpath = '/'.join(filepath.split('/')[:-2])
            newpath = os.path.join(newpath, os.path.basename(filepath))

            os.rename(filepath, newpath)


def remove_empty_directories():
    os.chdir('./IEMOCAP')
    for modality in ['raw-videos', 'raw-audios', 'raw-texts']:
        os.chdir(modality)
        for SPLIT in ['train', 'val', 'test']:
            os.chdir(SPLIT)
            for folder_path in os.listdir('./'):
                if os.path.isdir(folder_path) and \
                        len(os.listdir(folder_path)) == 0:
                    shutil.rmtree(folder_path)
            os.chdir('../')
        os.chdir('../')
    os.chdir('../')


def create_labels():
    def parse_emotion_dialogue(textpath):
        diaid = os.path.basename(textpath).split('.txt')[0]
        parsed = {}
        with open(textpath, 'r') as stream:
            to_parse = [line.strip() for line in stream.readlines()]

        indexes = []
        for idx, line in enumerate(to_parse):
            if diaid not in line:
                continue
            indexes.append(idx)

        # include the last line for computataion convenience
        indexes.append(len(to_parse))

        for idx_prev, idx_next in zip(indexes[:-1], indexes[1:]):
            for i in range(idx_prev, idx_next):
                line = to_parse[i]
                if diaid in line:
                    uttid, voted = line.split('\t')[1], line.split('\t')[2]
                    parsed[uttid] = {'voted': voted}
                if 'C-' in line:
                    cx, label = line.split('\t')[0], line.split('\t')[1]
                    cx = cx.split(':')[0]
                    label = [lbl for lbl in label.split(';')]
                    label = [lbl for lbl in label if len(lbl) != 0]
                    label = [lbl.replace(" ", "") for lbl in label]
                    parsed[uttid][cx] = label

        return diaid, parsed

    text_paths = sorted(glob(
        './IEMOCAP/IEMOCAP_full_release/Session*/dialog/EmoEvaluation/Ses*.txt'))

    print(f"{len(text_paths)} labeled dialogs found.")

    labels = [parse_emotion_dialogue(textpath) for textpath in text_paths]
    labels = {lbl[0]: lbl[1] for lbl in labels}

    with open('IEMOCAP/utterance-ordered.json', 'r') as stream:
        utterance_ordered = json.load(stream)

    labels = {SPLIT: {diaid: foo for diaid, foo in labels.items()
                      if diaid in list(utterance_ordered[SPLIT].keys())}
              for SPLIT in ['train', 'val', 'test']}

    for SPLIT in ['train', 'val', 'test']:
        num_utts = len([val_ for key, val in labels[SPLIT].items()
                        for val_ in val])

        assert num_utts == len(
            [uttid for diaid, list_of_utts in utterance_ordered[SPLIT].items()
             for uttid in list_of_utts])

        print(
            f"{SPLIT} has {len(labels[SPLIT])} dialogues and "
            f"{len([val_ for key, val in labels[SPLIT].items() for val_ in val] )} utterances")

    voted_labels = [bar['voted'] for SPLIT in ['train', 'val', 'test']
                    for diaid, foo in labels[SPLIT].items()
                    for _, bar in foo.items()]

    print(f"There are {len(set(voted_labels))} unique labels.")

    label_3 = {'ang', 'dis', 'exc', 'fea', 'fru',
               'hap', 'neu', 'oth', 'sad', 'sur', 'xxx'}

    label_fullname = {'Anger', 'Disgust', 'Excited', 'Fear', 'Frustration',
                      'Happiness', 'Neutral', 'Other', 'Sadness', 'Surprise'}
    label_fullname.add('Undecided')

    label_map = {'ang': 'Anger',
                 'dis': 'Disgust',
                 'exc': 'Excited',
                 'fea': 'Fear',
                 'fru': 'Frustration',
                 'hap': 'Happiness',
                 'neu': 'Neutral',
                 'oth': 'Other',
                 'sad': 'Sadness',
                 'sur': 'Surprise',
                 'xxx': 'Undecided'}

    assert set(list(label_map.keys())) == label_3
    assert set(list(label_map.values())) == label_fullname

    undecided = {SPLIT:
                 {diaid:
                  {uttid: {baz: qux for baz, qux in bar.items() if baz != 'voted'}
                   for uttid, bar in foo.items() if bar['voted'] == 'xxx'}
                  for diaid, foo in labels[SPLIT].items()}
                 for SPLIT in ['train', 'val', 'test']}

    # remove diaid.
    undecided = {SPLIT: {uttid: bar for diaid, foo in undecided[SPLIT].items()
                         for uttid, bar in foo.items()}
                 for SPLIT in ['train', 'val', 'test']}

    with open('IEMOCAP/undecided-emotions.json', 'w') as stream:
        json.dump(undecided, stream, indent=4)

    labels = {SPLIT: {diaid: {uttid: label_map[bar['voted']].lower()
                              for uttid, bar in foo.items()}
                      for diaid, foo in labels[SPLIT].items()}
              for SPLIT in ['train', 'val', 'test']}

    # remove diaid.
    labels = {SPLIT: {uttid: bar for diaid, foo in labels[SPLIT].items()
                      for uttid, bar in foo.items()}
              for SPLIT in ['train', 'val', 'test']}

    with open('IEMOCAP/emotions.json', 'w') as stream:
        json.dump(labels, stream, indent=4)

    for jsonpath in glob(f"IEMOCAP/raw-texts/*/*.json"):
        with open(jsonpath, 'r') as stream:
            text = json.load(stream)
        SPLIT = jsonpath.split('/')[2]
        uttid = os.path.basename(jsonpath).split('.json')[0]
        emotion = labels[SPLIT][uttid]
        text['Emotion'] = emotion

        sessid = os.path.basename(jsonpath).split(
            '_')[0].split('M')[0].split('F')[0]
        text['SessionID'] = sessid

        with open(jsonpath, 'w') as stream:
            json.dump(text, stream, indent=4, ensure_ascii=False)

    README = f"This dataset has all three modalities!\n"\
        f"Every utterance is part of a dialogue. If you also want to take the dialogue\n"\
        f"into consideration, see utterance-ordered.json\n"\
        f"One thing annoying about this dataset is that there are a lot of 'xxx' labels.\n"\
        f", which means they are 'Undecided' due to the labelers not agreeing on one thing.\n"\
        f"If you want to see the votes, see 'undecided-emotions.json'\n\n"\
        f"This README is written by Taewoon Kim (https://taewoonkim.com)"

    with open(f"./IEMOCAP/README.txt", 'w') as stream:
        stream.write(README)


def run():

    move_raw_files()
    sort_audios()
    make_splits()
    clean_texts()
    slice_videos()
    add_audios_to_videos()
    clean_files()
    remove_empty_directories()
    create_labels()
