import random
from sklearn.utils import resample
from collections import Counter
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict

class HF(Dataset):
    def __init__(self, medmnist_dataset):
        self.dataset = medmnist_dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}
    
def convert_to_hf(medmnist_dataset):
    """
    Convert a MedMNIST dataset to a HuggingFace dataset
    Args:
        medmnist_dataset: The original MedMNIST dataset
    Returns:
        The HuggingFace dataset
    """

    wrapper_dataset = HF(medmnist_dataset)
    
    images = []
    labels = []
    
    for i in range(len(wrapper_dataset)):
        sample = wrapper_dataset[i]
        images.append(sample["image"])
        labels.append(sample["label"].item())
    
    hf_dataset = HFDataset.from_dict({"image": images,"label": labels})
    
    return hf_dataset

def dataset_balancing(dataset, alpha):
    """	
    Balance the dataset by undersampling the majority class
    Args:
        dataset: The original dataset
        alpha: The factor of how much more majority to keep (0 means perfect balance)
    Returns:
        The balanced dataset
    """

    mj_class = Counter(dataset['label']).most_common(1)[0][0] # majority class
    mn_class = abs(mj_class-1) # minority class
    data = dataset['image'] # all images

    mask = [lb == mj_class for lb in dataset['label']] # mask to separate majority and minority class
    X_majority = [img for img,flag in zip(data, mask) if flag] # majority class images
    X_minority = [img for img,flag in zip(data, mask) if not flag] # minority class images
    new_len_majority = len(X_minority) + int(alpha*len(X_minority))

    X_majority_resampled = resample(X_majority, 
                                    replace=False,
                                    n_samples=new_len_majority,
                                    random_state=42)
    
    X_resampled = X_majority_resampled + X_minority # resampled dataset
    y_resampled = [mj_class]*new_len_majority + [mn_class]*len(X_minority) # resampled labels
    combined = list(zip(X_resampled, y_resampled))
    random.seed(42)
    random.shuffle(combined)
    X_resampled, y_resampled = zip(*combined)

    dict_blanced_dataset = {"image": X_resampled, "label": y_resampled}
    balanced_dataset = HFDataset.from_dict(dict_blanced_dataset)

    return balanced_dataset

def load_dataset(dataset_name, size, balancing, alpha=0):
    """
    Load a MedMNIST dataset converting it to a HuggingFace dataset
    Args:
        dataset_name: The name of the MedMNIST dataset
        size: The size of the images (resolution)
        balancing: Whether to balance the dataset
    Returns:
        The HuggingFace dataset
    """
    
    splits = {
        "train": convert_to_hf(dataset_name(split='train', download=True, size=size)),
        "validation": convert_to_hf(dataset_name(split='val', download=True, size=size)),
        "test": convert_to_hf(dataset_name(split='test', download=True, size=size))
    }
    
    if balancing:
        splits["train"] = dataset_balancing(splits["train"], alpha)

    return DatasetDict(splits)