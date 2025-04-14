from numbers import Number
from typing import Optional, Union

import numpy as np
import pandas as pd

from deepseqreen.utils import get_logger

log = get_logger(__name__)

MOLARITY_TO_POTENCY = {
    'p': lambda x: x,
    'M': lambda x: -np.log10(x),
    'mM': lambda x: -np.log10(x) + 3,
    'μM': lambda x: -np.log10(x) + 6,
    'uM': lambda x: -np.log10(x) + 6,  # in case someone doesn't know how to type micromolar lol
    'nM': lambda x: -np.log10(x) + 9,
    'pM': lambda x: -np.log10(x) + 12,
    'fM': lambda x: -np.log10(x) + 15,
}


def molar_to_p(labels: pd.Series, units: pd.Series) -> pd.Series:
    # Check if all units are in the allowed list
    assert np.isin(units, list(MOLARITY_TO_POTENCY)).all(), f"Allowed units: {', '.join(MOLARITY_TO_POTENCY.keys())}."

    combined_df = pd.DataFrame({'label': labels, 'unit': units})
    converted_labels = combined_df.swifter.apply(lambda row: MOLARITY_TO_POTENCY[row['unit']](row['label']), axis=1)

    return converted_labels


def label_discretize(labels, thresholds):
    # if isinstance(threshold, Number):
    #     labels = np.where(labels < threshold, 1, 0)
    # else:
    #     labels = np.where(labels < threshold[0], 1, np.where(labels > threshold[1], 0, np.nan))
    if isinstance(thresholds, Number):
        labels = 1 - np.digitize(labels, [thresholds])
    else:
        labels = np.digitize(labels, np.sort(thresholds)[::-1])

    return labels


def label_transform(
        labels,
        thresholds: Optional[Union[float, list[Number]]],
        units: Optional[list[str]] = None,
        discard_intermediate: Optional[bool] = False,
):
    f"""Convert labels of all units to p scale (-log10[M]) and binarize them if specified.
        :param labels: a sequence of labels, continuous or binary values
        :type labels: array_like
        :param units: a sequence of label units in {', '.join(MOLARITY_TO_POTENCY)}
        :type units: array_like, optional
        :param thresholds: discretization threshold(s) for affinity labels, in p scale (-log10[M]). 
        A single number maps affinities below it to 1 and otherwise to 0.
        A tuple of two or more thresholds maps affinities to multiple discrete levels descendingly, assigning values 
        values below the lowest threshold to the highest level (e.g. 2) and values above the greatest threshold to 0 
        :type thresholds: list, float, optional
        :param discard_intermediate: whether to discard the intermediate (indeterminate) level if provided an odd 
        number of thresholds (>=3)
        :type discard_intermediate: bool
        :return: a numpy array of affinity labels in p scale (-log10[M]) or discrete labels
    """
    # # Check if labels are already discrete (ignoring NAs).
    # discrete = labels.dropna().isin([0, 1]).all()
    #
    # if discrete:
    #     assert discretize, "Cannot train a regression model with discrete labels."
    #     if thresholds:
    #         warn("Ignoring 'threshold' because 'Y' (labels) in the data table is already binary.")
    #     if units:
    #         warn("Ignoring 'units' because 'Y' (labels) in the data table is already binary.")
    #     labels = labels
    if units is not None:
        log.info(f'Converting labels (units: {units.unique()}) to p-scale...')
        labels = molar_to_p(labels, units)

    if thresholds:
        log.info(f'Descretizing labels based on p-scale thresholds ({thresholds})...')
        labels = label_discretize(labels, thresholds)
        if discard_intermediate:
            assert len(thresholds) % 2 == 0 and len(thresholds) >= 2, \
                "Must give an even number of (at least 2) thresholds to discard the intermediate level."
            intermediate_level = len(thresholds) / 2
            # Make the intermediate-level labels NaN (which will be filtered out later)
            log.info(f'Converting labels (units: {units.unique()}) to p-scale (between {thresholds[intermediate_level:intermediate_level + 1]})...')
            labels[labels == intermediate_level] = np.nan
            # Reduce all levels above the intermediate level by 1
            labels[labels > intermediate_level] -= 1

    return labels

