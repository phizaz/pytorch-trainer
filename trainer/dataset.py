from torch.utils.data import Dataset


class SubsetDataset(Dataset):
    def __init__(self, original_dataset, size):
        self.original_dataset = original_dataset
        self.size = size
        assert len(
            self.original_dataset
        ) >= size, f'the subset size must be smaller than the original size.'

    def __len__(self):
        # resize the dataset
        return self.size

    def __getitem__(self, i):
        return self.original_dataset[i]