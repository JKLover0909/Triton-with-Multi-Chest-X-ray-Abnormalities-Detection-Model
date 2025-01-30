import multiprocessing as mp
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
import pandas as pd
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

notebook = False




def get_progress(iterable=None, total=None, desc=None, leave=False):
    """
    get progress bar
    :param iterable: target to be iterated
    :param total: total length of the progress bar
    :param desc: description of the progress bar
    :return: progress bar
    """

    if notebook:
        return tqdm_nb(iterable=iterable, total=total, desc=desc, leave=leave)

    return tqdm(iterable=iterable, total=total, desc=desc, leave=leave)



def parallel_iterate(arr, iter_func, workers=8, use_index=False, **kwargs):
    """
    parallel iterate array
    :param arr: array to be iterated
    :param iter_func: function to be called for each data, signature (idx, arg) or arg
    :param workers: number of worker to run
    :param use_index: whether to add index to each call of iter func
    :return list of result if not all is None
    """
    with mp.Pool(workers) as p:
        if isinstance(arr, zip):
            jobs = [p.apply_async(iter_func, args=(i, *arg) if use_index else arg, kwds=kwargs) for i, arg in enumerate(arr)]
        elif isinstance(arr, pd.DataFrame):
            jobs = [p.apply_async(iter_func, args=(i, row) if use_index else (row,), kwds=kwargs) for i, row in arr.iterrows()]
        else:
            jobs = [p.apply_async(iter_func, args=(i, arg) if use_index else (arg,), kwds=kwargs) for i, arg in enumerate(arr)]
        results = [j.get() for j in get_progress(jobs)]
        return results
    


def normalize(arr):
        arr = (arr - arr.min()) / (arr.max()-arr.min())

        return arr
    


def train_transform():
    return A.Compose(
            [
                A.RandomBrightnessContrast(p=0.8),
                A.ChannelDropout(p=0.5),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.5),
                        A.MedianBlur(p=0.5),
                        A.GaussianBlur(p=0.5),
                        A.GaussNoise(p=0.5),
                    ],
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
