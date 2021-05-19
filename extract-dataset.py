import argparse
import logging
import os
from tqdm import tqdm
import shutil

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class Dataset():
    def __init__(self, dataset):
        self.archives = \
            {'MELD': 'MELD.Raw.tar.gz',
             'IEMOCAP': 'IEMOCAP_full_release.tar.gz'}
        self.SUPPORTED_DATASETS = list(self.archives.keys())
        self.dataset = dataset

    def sanity_check(self):
        if self.dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"{self.dataset} is not one of the supported datasets: "
                f"{self.SUPPORTED_DATASETS}!")

        assert os.path.isfile(os.path.join(
            self.dataset, self.archives[self.dataset])), \
            f"place the archive in the dataset directory!"

        logging.info(f"sanity check on {self.dataset} successful")

    def extract_archive(self):
        logging.info(f"extracting {self.archives[self.dataset]} ...")
        archive_path = os.path.join(self.dataset, self.archives[self.dataset])

        if self.dataset == 'MELD':
            from utils.extract_archive import extract_meld
            extract_meld(archive_path)
        elif self.dataset == 'IEMOCAP':
            from utils.extract_archive import extract_iemocap
            extract_iemocap(archive_path)
        logging.info(f"extraction complete.")

    def create_raw_directories(self):
        logging.debug(f"creating raw directories ...")
        self.modalities = ['videos', 'audios', 'texts']
        if self.dataset in ['EmoryNLP', 'DailyDialog']:
            self.modalities.remove('videos')
            self.modalities.remove('audios')

        for modality in self.modalities:
            for SPLIT in ['train', 'val', 'test']:
                os.makedirs(f"./{self.dataset}/raw-{modality}/{SPLIT}",
                            exist_ok=True)

    def extract_raw_data(self):
        logging.debug(f"extracting raw data to {self.modalities}...")
        if self.dataset == 'MELD':
            from utils import extract_raw_data_meld
            extract_raw_data_meld.run()
            shutil.rmtree('./MELD/MELD.Raw', ignore_errors=True)

        elif self.dataset == 'IEMOCAP':
            from utils import extract_raw_data_iemocap
            extract_raw_data_iemocap.run()
            shutil.rmtree('./IEMOCAP/IEMOCAP_full_release', ignore_errors=True)

        logging.info(f"extracting {self.modalities} raw data complete.")


def main(dataset):

    ds = Dataset(dataset)
    ds.sanity_check()
    ds.extract_archive()
    ds.create_raw_directories()
    ds.extract_raw_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='process a multimodal dataset')
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()
    args = vars(args)

    logging.info(f"arguments given to {__file__}: {args}")

    main(**args)
