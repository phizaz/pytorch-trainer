from torch.utils.data import Dataset


class SubsetDataset(Dataset):
    def __init__(self, original_dataset, size):
        self.original_dataset = original_dataset
        self.size = min(size, len(original_dataset))

    def __len__(self):
        # resize the dataset
        return self.size

    def __getitem__(self, i):
        return self.original_dataset[i]

class SliceDataset(Dataset):
    def __init__(self, original_dataset, start_index: int, end_index: int):
        self.original_dataset = original_dataset
        assert start_index < len(self.original_dataset)
        assert end_index < len(self.original_dataset)
        self.start_index = start_index
        self.end_index = end_index

    def __len__(self):
        # resize the dataset
        return self.end_index - self.start_index

    def __getitem__(self, i):
        assert i + self.start_index < self.end_index
        return self.original_dataset[i + self.start_index]