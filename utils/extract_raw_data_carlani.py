from glob import glob

from tqdm import tqdm

from .audio import extract_wav_from_video


def run():
    for SPLIT in tqdm(["train", "val", "test"]):
        vidpaths = glob(f"./CarLani/raw-videos/{SPLIT}/*.mp4")
        for vidpath in tqdm(vidpaths):
            audiopath = vidpath.replace("raw-videos", "raw-audios")
            audiopath = audiopath.replace(".mp4", ".wav")
            extract_wav_from_video(vidpath, audiopath)
