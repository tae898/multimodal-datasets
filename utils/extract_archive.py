import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def extract_meld(archive_path):
    import tarfile
    my_tar = tarfile.open(archive_path)
    my_tar.extractall('./MELD/')
    my_tar.close()

    for subtar in ['train.tar.gz', 'dev.tar.gz', 'test.tar.gz']:
        my_tar = tarfile.open(f"./MELD/MELD.Raw/{subtar}")
        my_tar.extractall('./MELD/MELD.Raw/')
        my_tar.close()


def extract_iemocap(archive_path):
    import tarfile
    my_tar = tarfile.open(archive_path)
    my_tar.extractall('./IEMOCAP/')
    my_tar.close()

    os.remove("./IEMOCAP/._IEMOCAP_full_release")


def extract_emorynlp(archive_path):
    import tarfile
    my_tar = tarfile.open(archive_path)
    my_tar.extractall('./EmoryNLP/')
    my_tar.close()


def extract_dailydialog(archive_path):
    import zipfile
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall('./DailyDialog')

    for subzip in ['train.zip', 'validation.zip', 'test.zip']:
        with zipfile.ZipFile(f"DailyDialog/ijcnlp_dailydialog/{subzip}", 'r') \
                as zip_ref:
            zip_ref.extractall('./DailyDialog/ijcnlp_dailydialog/')
