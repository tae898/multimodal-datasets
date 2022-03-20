# Multimodal-Datasets

This repo collects multimodal datasets and process them in a nice manner. Please let me know if you have some interesting datasets to be processed.

## Supported Datasets

1. [MELD](https://affective-meld.github.io/)

   This dataset is from the tv series *Friends*.
   It's got visual, audio, and text modalities.

1. [IEMOCAP](https://sail.usc.edu/iemocap/)

   This dataset includes dyadic conversation between two people.
   There are 10 actors in total. It's got visual, audio, and text modalities.

1. [CarLani](https://surfdrive.surf.nl/files/index.php/s/I5Gg87eVN3l1KEP/download)

   This dataset is a conversation between the robot named Leolani and the human named Carl.

## Dataset Structure

Every dataset has their own directory and it'll look something like below.
If a dataset does not have all three modalities (i.e. visual, audio, and text),
then obviously those directories will be empty.

```console
DATASET
├── raw-videos
│   ├── train / val /test
├── raw-audios
│   ├── train / val /test
├── raw-texts
│   ├── train / val /test
├── face-videos
│   ├── train / val /test
├── face-features
│   ├── train / val /test
├── visual-features
│   ├── train / val /test
├── audio-features
│   ├── train / val /test
├── text-features
│   ├── train / val /test
├── foo.json
├── bar.json
└── README.txt
```

Every data sample belongs to one of the train, val, or test split.
Beware that the numbers are not always the same. For example, there might be a video but then if its transcription is not available or if audio extraction fails, then it won't have text or audio modalities.

1. `DATASET` is the name of the dataset.
1. `raw-videos` contains the raw non-processed videos.
1. `raw-audios` contains the raw non-processed audios.
1. `raw-texts` contains the raw non-processed texts.
1. `face-videos` contains the face videos made from the facial features.
1. `face-features` contains the detected faces and their features using this [repo](https://github.com/tae898/face-detection-recognition) and [repo](https://github.com/tae898/age-gender) The features are age, bounding box, face detection probability, gender, five landmarks, and 512-dimensional arcface embedding vector.
1. `visual-features` are other visual features (e.g. COCO 80 objects) that are not facial features.
1. `audio-features` contains audio features (e.g. spectrogram, encoded embeddings, etc.)
1. `text-features` contains text features (e.g. word embeddings, BERT-like model features, etc.)
1. `*.json` are some important data-specific text files (e.g. label, etc.)
1. `README.txt` briefly explains the dataset and gives you the metadata

## How to process each dataset

There several levels.

**Use python >= 3.8**

### Dataset extraction

This is about extracting the original datasets into the above mentioned structure (i.e. raw-videos, raw-audios, raw-texts)

1. Since I don't have license to all of the datasets, you should contact the dataset authors and download them yourself.

1. Install the python requirements by

   ```bash
   pip install -r requirements-extract-dataset.txt
   ```

   I highly recommend you to run it in a virtual environment.

1. Move the archive in the corresponding directory.

   - Put `MELD.Raw.tar.gz` in `MELD/`
   - Put `IEMOCAP_full_release.tar.gz` in `IEMOCAP/`
   - Put `CarLani.zip` in `CarLani/`

1. In this current directory, where `README.md` is located, run

   ```bash
   python extract-dataset.py --dataset DATASET
   ```

   Replace `DATASET` with your desired dataset (e.g. `python extract-dataset.py --dataset MELD`)

**In the next sections, we will go through extracting features and annotating with EMISSOR. Go ahead if you want to do it youself, otherwise you can just download them from the below links.**

1. [MELD](https://surfdrive.surf.nl/files/index.php/s/hGghcDwmKb5OM07/download)

   In the current repo root directory, unzip what you downloaded into the directory `./MELD/`

1. [IEMOCAP](https://surfdrive.surf.nl/files/index.php/s/HLECWlhvDSEhtgW/download)

   In the current repo root directory, unzip what you downloaded into the directory `./IEMOCAP/`

1. [CarLani](https://surfdrive.surf.nl/files/index.php/s/I5Gg87eVN3l1KEP/download)

   In the current repo root directory, unzip what you downloaded into the directory `./CarLani/`

### Feature Extraction

This is about extracting the featrues (e.g. face-features)

- For facial features, you will pull three docker images.
- For visual features, you should build something. I'll soon make them into docker containers.
- For audio features, you should build something. I'll soon make them into docker containers.
- For text features, you should build something. I'll soon make them into docker containers.

1. Install the python requirements by

   ```bash
   pip install -r requirements-extract-features.txt
   ```

   I highly recommend you to run it in a virtual environment.

1. In this current directory, where `README.md` is located, run

   ```bash
   python extract-features.py --dataset DATASET --face-features --face-videos --visual-features --audio-features --text-features --run-on-gpu --num-jobs NUM_JOBS 
   ```

   Replace `DATASET` with your desired dataset. Only add the boolean flags (i.e. --face-features, --face-videos, --visual-features, --audio-features, --text-features) that you want to extract. For example, if you only want to extract face features and audio features from the MELD dataset, the command should be `python extract-features.py --dataset MELD --face-features --audio-features`. If you want to run in parallel, you can add the gpu flag `--run-on-gpu` and even add more workers `--num-jobs NUM_JOBS`. Running on GPU requires you to have a NVIDIA GPU and you should build the GPU images for this. Read https://github.com/tae898/face for more information.

### Annotate the dataset in the EMISSOR format (optional)

This is optional. The processed datasets can also be annotated in the [EMISSOR annotation format](https://github.com/cltl/GMRCAnnotation). Using the EMISSOR annotation tool, users can visualize the data or even annotate the data by themselves.

1. Install the python requirements by

   ```bash
   pip install -r requirements-annotate-emissor.txt
   ```

   I highly recommend you to run it in a virtual environment.

1. In this current directory, where `README.md` is located, run

   ```bash
   python annotate-emissor.py --dataset DATASET --num-jobs NUM_JOBS
   ```

   The python script can only use the features extracted.

You can also download them from the below link.

1. [MELD](https://surfdrive.surf.nl/files/index.php/s/fwiMEPEDCnPjfGm/download)

   In the current repo root directory, unzip what you downloaded into the directory `./MELD/`

1. [IEMOCAP](https://surfdrive.surf.nl/files/index.php/s/fDfr1yRT8SqATuV/download)

   In the current repo root directory, unzip what you downloaded into the directory `./IEMOCAP/`

1. [CarLani](https://surfdrive.surf.nl/files/index.php/s/I5Gg87eVN3l1KEP/download)

   In the current repo root directory, unzip what you downloaded into the directory `./CarLani/`

## Troubleshooting

The best way to find and solve your problems is to see in the github issue tab. If you can't find what you want, feel free to raise an issue. We are pretty responsive.

## Cite our work

[![DOI](https://zenodo.org/badge/358332238.svg)](https://zenodo.org/badge/latestdoi/358332238)

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make style && quality` in the root repo directory, to ensure code
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## Contact

If you have any questions, or have interesting datasets, then please let me know.

- [Taewoon Kim](https://taewoonkim.com/)
