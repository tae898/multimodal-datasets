import os
from glob import glob
from tqdm import tqdm
from .audio import extract_wav_from_video
from . import convert_cp1252_to_utf8
import json
import csv


def get_vidpaths(SPLIT):
    if SPLIT == 'train':
        vidpaths = glob('./MELD/MELD.Raw/train_splits/*.mp4')
    elif SPLIT == 'val':
        vidpaths = glob('./MELD/MELD.Raw/dev_splits_complete/*.mp4')
    else:
        vidpaths = glob('./MELD/MELD.Raw/output_repeated_splits_test/*.mp4')

    return vidpaths


def extract_text():
    annotations = {}
    for DATASET in ['train', 'dev', 'test']:
        with open(f'./MELD/MELD.Raw/{DATASET}_sent_emo_converted.csv') as f:
            reader = csv.reader(f)
            if DATASET == 'dev':
                DATASET = 'val'
            annotations[DATASET] = list(reader)

    datasets = {}
    labels = {}
    utterance_ordered = {}
    for DATASET in ['train', 'val', 'test']:
        datasets[DATASET] = {}
        labels[DATASET] = {}
        utterance_ordered[DATASET] = {}

        for row in annotations[DATASET][1:]:
            SrNo, Utterance, Speaker, Emotion, Sentiment, Dialogue_ID,\
                Utterance_ID, Season, Episode, StartTime, EndTime = row

            Utterance = Utterance.strip()

            to_dump = {'SrNo': SrNo,
                       'Utterance': Utterance,
                       'Speaker': Speaker,
                       'Emotion': Emotion,
                       'Sentiment': Sentiment,
                       'Dialogue_ID': Dialogue_ID,
                       'Utterance_ID': Utterance_ID,
                       'Season': Season,
                       'Episode': Episode,
                       'StartTime': StartTime,
                       'EndTime': EndTime}

            if f"dia{Dialogue_ID}" not in list(utterance_ordered[DATASET].keys()):
                utterance_ordered[DATASET][f"dia{Dialogue_ID}"] = []
            utterance_ordered[DATASET][f"dia{Dialogue_ID}"].append(
                f"dia{Dialogue_ID}_utt{Utterance_ID}")

            labels[DATASET][f"dia{Dialogue_ID}_utt{Utterance_ID}"] = Emotion.lower(
            )

            with open(f"./MELD/raw-texts/{DATASET}/"
                      f"dia{Dialogue_ID}_utt{Utterance_ID}.json", 'w') as stream:
                json.dump(to_dump, stream, indent=4,
                          sort_keys=True, ensure_ascii=False)

        utts_all = [utt for dia, diautts in utterance_ordered[DATASET].items()
                    for utt in diautts]

        assert len(labels[DATASET]) == len(set(labels[DATASET])
                                           ) == len(utts_all) == len(set(utts_all))

    with open(f"./MELD/emotions.json", 'w') as stream:
        json.dump(labels, stream, indent=4, ensure_ascii=False)

    with open(f"./MELD/utterance-ordered.json", 'w') as stream:
        json.dump(utterance_ordered, stream, indent=4,
                  ensure_ascii=False)


def run():
    for SPLIT in tqdm(['train', 'val', 'test']):
        vidpaths = get_vidpaths(SPLIT)
        for vidpath in tqdm(vidpaths):
            basename = os.path.basename(vidpath)
            os.rename(vidpath,  f"./MELD/raw-videos/{SPLIT}/{basename}")

    for SPLIT in tqdm(['train', 'val', 'test']):
        vidpaths = glob(f'./MELD/raw-videos/{SPLIT}/*.mp4')
        for vidpath in tqdm(vidpaths):
            audiopath = vidpath.replace('raw-videos', 'raw-audios')
            audiopath = audiopath.replace('.mp4', '.wav')
            extract_wav_from_video(vidpath, audiopath)

    convert_cp1252_to_utf8.run()
    extract_text()

    README = \
        f"This dataset has all three modalities!\n"\
        f"Every utterance is part of a dialogue. If you also want to take the dialogue\n"\
        f"into consideration, see utterance-ordered.json\n\n"\
        f"This README is written by Taewoon Kim (https://taewoonkim.com)"

    with open(f"./MELD/README.txt", 'w') as stream:
        stream.write(README)
