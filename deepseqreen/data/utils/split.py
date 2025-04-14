import torch
from typing import List, Union
from torch.utils import data

# FIXME Test and fix these split functions later


def random_split(dataset: data.Dataset, lengths, seed):
    return data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))


def cold_start(dataset: data.Dataset, frac: List[float], entities: Union[str, List[str]]):
    """Create cold-start splits for PyTorch datasets.

    Args:
        dataset (Dataset): PyTorch dataset object.
        frac (list): A list of train/valid/test fractions.
        entities (Union[str, List[str]]): Either a single "cold" entity or a list of "cold" entities
            on which the split is done.

    Returns:
        dict: A dictionary of splitted datasets, where keys are 'train', 'valid', and 'test',
            and values correspond to each dataset.
    """
    if isinstance(entities, str):
        entities = [entities]

    train_frac, val_frac, test_frac = frac

    # Collect unique instances for each entity
    entity_instances = {}
    for entity in entities:
        entity_instances[entity] = list(set([getattr(sample, entity) for sample in dataset]))

    # Sample instances belonging to the test datasets
    test_entity_instances = [
        torch.randperm(len(entity_instances[entity]))[:int(len(entity_instances[entity]) * test_frac)]
        for entity in entities
    ]

    # Select samples where all entities are in the test set
    test_indices = []
    for i, sample in enumerate(dataset):
        if all([getattr(sample, entity) in entity_instances[entity][test_entity_instances[j]] for j, entity in enumerate(entities)]):
            test_indices.append(i)

    if len(test_indices) == 0:
        raise ValueError('No test samples found. Try increasing the test frac or a less stringent splitting strategy.')

    # Proceed with validation data
    train_val_indices = list(set(range(len(dataset))) - set(test_indices))

    val_entity_instances = [
        torch.randperm(len(entity_instances[entity]))[:int(len(entity_instances[entity]) * val_frac / (1 - test_frac))]
        for entity in entities
    ]

    val_indices = []
    for i in train_val_indices:
        if all([getattr(dataset[i], entity) in entity_instances[entity][val_entity_instances[j]] for j, entity in enumerate(entities)]):
            val_indices.append(i)

    if len(val_indices) == 0:
        raise ValueError('No validation samples found. Try increasing the test frac or a less stringent splitting strategy.')

    train_indices = list(set(train_val_indices) - set(val_indices))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return {'train': train_dataset, 'valid': val_dataset, 'test': test_dataset}
