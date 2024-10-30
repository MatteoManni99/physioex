from typing import Callable, List

class ConceptLabeler:
    """
    A class to label concepts in physiological datasets using a pre-trained model.

    This class is designed for concept labeling tasks in physiological data analysis. It computes distances between model embeddings and prototypes, labels concepts based on these distances, and saves the labeled data. Additionally, it manages data preparation and normalization for efficient processing.

    Parameters:
        `dataset_name` (str): The name of the dataset for which concepts are labeled.
        `data_folder` (str): Path to the root folder containing the dataset.
        `model_class` (Callable): Class reference for loading the pre-trained model.
        `model_config` (dict): Configuration dictionary for initializing the model.
        `model_ckpt_path` (str): Path to the model checkpoint file.
        `channels_index` (List[int]): Indices for selecting specific channels from the dataset.
        `sequence_length` (int): Length of the input sequence window for processing.
        `lambda_fun` (Callable, optional): A custom function for calculating concept values from distances. Defaults to an inverse exponential function (lambda d: 1/(10000**d).

    Attributes:
        `model` (torch.nn.Module): The loaded pre-trained model.
        `data_folder` (str): Path to the root data directory.
        `dataset_folder` (str): Path to the dataset within the data folder.
        `data_path` (str): Path to the directory containing dataset samples.
        `channels_index` (List[int]): Selected channel indices for data processing.
        `L` (int): Length of the input sequence.
        `central_window` (int): Central window index in the input sequence.
        `mean` (np.ndarray): Array of mean values for data normalization.
        `std` (np.ndarray): Array of standard deviation values for normalization.
        `input_shape` (List[int]): Shape of a single input sample.
        `concepts_path` (str): Directory path for storing labeled concepts.
        `distances_path` (str): Directory path for storing computed distances.
        `max_dist_value` (float): Maximum observed distance value.
        `min_dist_value` (float): Minimum observed distance value.

    Example:
        ```python
        from physioex.train.models import MyCustomModel

        labeler = ConceptLabeler(
            dataset_name="my_dataset",
            data_folder="/path/to/data",
            model_class=MyCustomModel,
            model_config={"param1": value1, "param2": value2},
            model_ckpt_path="/path/to/checkpoint",
            channels_index=[0, 1, 2],
            sequence_length=21,
        )
        labeler.run()
        ```

    Notes:
        - Distance values are normalized using min-max scaling based on observed min and max values.
        - Processed data, including distances and labeled concepts, are saved in specified folders.
        - The distance directory is cleaned after labeling, conserving storage space.

    """

    def __init__(
        self,
        dataset_name: str,
        data_folder: str,
        model_class: Callable,
        model_config: dict,
        model_ckpt_path: str,
        channels_index: List[int],
        sequence_length: int,
        lambda_fun=None,
    ):
        """
        Initializes the ConceptLabeler with paths, configurations, and normalization settings.

        Args:
            dataset_name (str): Name of the dataset for concept labeling.
            data_folder (str): Path to the directory containing data folders.
            model_class (Callable): Model class reference for loading the model.
            model_config (dict): Dictionary of configuration parameters for initializing the model.
            model_ckpt_path (str): Checkpoint file path for loading model weights.
            channels_index (List[int]): List of indices for selected channels.
            sequence_length (int): Length of the input sequence for model processing.
            lambda_fun (Callable, optional): Function for calculating concept values from distances. Defaults to an inverse exponential scaling (lambda d: 1/(10000**d).
        """
        pass

    def run(self):
        """
        Executes the concept labeling process on the dataset.

        This method loads and processes each sample in the dataset, calculates distances between model embeddings and prototypes, and saves both distances and concept values. Cleans up the distances folder after processing.

        Returns:
            None
        """
        pass
