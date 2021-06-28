
def get_dataset(split, hps):
    if hps.dataset == 'bn':
        from .bn import Dataset
        dataset = Dataset(hps.dfile, hps.gfile, split, hps.batch_size, hps.rnd_rate)
    elif hps.dataset == 'vec':
        from .vec import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size)
    elif hps.dataset == 'ts':
        from .ts import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size, hps.time_steps)
    elif hps.dataset == 'cube':
        from .cube import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size)
    elif hps.dataset == 'env':
        from .env import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size)
    elif hps.dataset == 'dag':
        from .dag import Dataset
        dataset = Dataset(hps.file_path, hps.exp_id, split, hps.batch_size)
    elif hps.dataset == 'questions':
        from .questions import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size)
    else:
        raise Exception()

    assert dataset.d == hps.dimension

    return dataset
