import logging
import os
import subprocess

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def extract_wav_from_video(loadpath, savepath, sr=22050):
    subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            loadpath,
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            savepath,
        ]
    )


def add_audio_on_video(videopath, audiopath):

    temppath = videopath + ".TEMP.mp4"
    logging.debug(f"Adding {audiopath} to {videopath} ...")

    subprocess.call(
        [
            "ffmpeg",
            "-y",
            "-i",
            videopath,
            "-i",
            audiopath,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            temppath,
        ]
    )

    os.remove(videopath)
    os.rename(temppath, temppath.split(".TEMP.mp4")[0])
