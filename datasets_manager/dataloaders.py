from torch.utils.data import DataLoader

class InfiniteLoader(DataLoader):
    def __init__(
        self,
        *args,
        num_workers=0,
        **kwargs,
    ):
        super().__init__(
            *args,
            multiprocessing_context="spawn" if num_workers > 0 else None,
            num_workers=num_workers,
            **kwargs,
        )
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.dataset_iterator)
        except StopIteration:
            if hasattr(self.dataset_iterator, "_shutdown_workers"):
                self.dataset_iterator._shutdown_workers()   # avoid leaks
            self.dataset_iterator = super().__iter__()
            x = next(self.dataset_iterator)
        return x
