import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch

def compute_embedding_ssn_cem(ssn_cem_model, dataloader, emb_dim=128, con_emb_dim = 30, L=3, batch_size = 128, num_concepts=15):
    """
    Computes embeddings, concepts, and mean squared error (MSE) using a SSN-CEM model.

    This function processes input data through a provided SSN-CEM model to generate 
    embeddings, predicted concepts, MSE values, and concept embeddings for each input 
    batch. It returns these computed values in numpy arrays.

    Parameters:
    ----------
    ssn_cem_model : torch.nn.Module
        An instance of a SSN-CEM model that implements the encoding function to compute 
        embeddings and predictions.

    dataloader : torch.utils.data.DataLoader
        A PyTorch DataLoader containing the input data and labels. Each batch should consist 
        of inputs and a tuple of labels (labels_class, labels_concept).

    emb_dim : int, optional
        The dimensionality of the embeddings returned by the model. Default is 128.

    con_emb_dim : int, optional
        The dimensionality of the concept embeddings returned by the model. Default is 30.

    L : int, optional
        The length of the sequence to process. It is expected that the model processes data 
        in sequences of this length. Default is 3.

    batch_size : int, optional
        The number of samples to be processed in each batch. Default is 128.

    num_concepts : int, optional
        The number of concept dimensions. Default is 15.

    Returns:
    -------
    embeddings_array : numpy.ndarray
        A 2D numpy array of shape (num_samples, emb_dim) containing the computed embeddings 
        for each input.

    labels_array : numpy.ndarray
        A 1D numpy array of shape (num_samples,) containing the true class labels for each 
        input.

    concepts_array : numpy.ndarray
        A 1D numpy array of shape (num_samples,) containing the predicted concept labels for 
        each input based on the highest probability.

    mse_array : numpy.ndarray
        A 2D numpy array of shape (num_samples, num_concepts) containing the mean squared 
        error values between the predicted and true concept labels for each input.

    con_emb_array : numpy.ndarray
        A 2D numpy array of shape (num_samples, con_emb_dim) containing the computed concept 
        embeddings for each input.

    Example:
    --------
    import torch
    from torch.utils.data import DataLoader

    # Assuming `ssn_cem_model` is defined and `dataloader` is set up
    embeddings, labels, concepts, mse, con_emb = compute_embedding_ssn_cem(
        ssn_cem_model,
        dataloader,
        emb_dim=128,
        con_emb_dim=30,
        L=3,
        batch_size=128,
        num_concepts=15
    )

    Notes:
    ------
    - The function assumes that the SSN-CEM model has a method `encode` that returns a tuple 
      containing embeddings and predictions.
    - It requires PyTorch and numpy to be installed in your environment.
    - The function uses `torch.no_grad()` to disable gradient calculation, which is appropriate 
      during inference to save memory and improve performance.
    - The function utilizes a tqdm progress bar for monitoring the progress of batch processing.

    """
    num = len(dataloader)*batch_size
    emb_dim = emb_dim
    con_emb_dim = con_emb_dim
    central_epoch = int((L - 1) / 2)

    embeddings_array = np.empty((num, emb_dim))
    concepts_array = np.empty((num))
    mse_array = np.empty((num, num_concepts))
    labels_array = np.empty((num))
    con_emb_array = np.empty((num, con_emb_dim))
    
    device = ssn_cem_model.device

    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            labels_class, labels_concept = labels
            embedding, pred = ssn_cem_model.encode(inputs.to(device))
            emb, con_emb = embedding
            pred_concept, pred_class = pred
            
            labels_concept = labels_concept[:, central_epoch, :]
            pred_concept = pred_concept[:, central_epoch, :, 0].to("cpu")
            mse = (labels_concept - pred_concept) ** 2
        
            embeddings_array[i*batch_size : i*batch_size+inputs.size(0)] = emb[:, central_epoch].cpu().numpy()
            labels_array[i*batch_size : i*batch_size+inputs.size(0)] = labels_class[:, central_epoch].cpu().numpy()
            concepts_array[i*batch_size : i*batch_size+inputs.size(0)] = torch.argmax(labels_concept, dim=1).cpu().numpy()
            con_emb_array[i*batch_size : i*batch_size+inputs.size(0)] = con_emb[:, central_epoch].cpu().numpy()
            mse_array[i*batch_size : i*batch_size+inputs.size(0)] = mse.cpu().numpy()
            
            if i == len(dataloader)-1:
                n_last_batch = inputs.size(0)
                first_empty_element = num-batch_size+n_last_batch

    return embeddings_array[:first_empty_element], labels_array[:first_empty_element], concepts_array[:first_empty_element], mse_array[:first_empty_element], con_emb_array[:first_empty_element]

def compute_embedding_autoencoder(model, dataloader, emb_dim=32, L=3, batch_size = 128):
    """
    Computes embeddings using an autoencoder model.

    This function processes input data through a provided autoencoder model to generate 
    embeddings for each input batch. It returns the computed embeddings and corresponding 
    labels in numpy arrays.

    Parameters:
    ----------
    model : torch.nn.Module
        An instance of an autoencoder model that implements an `encode` method to compute 
        embeddings.

    dataloader : torch.utils.data.DataLoader
        A PyTorch DataLoader containing the input data and labels. Each batch should consist 
        of inputs and labels.

    emb_dim : int, optional
        The dimensionality of the embeddings returned by the model. Default is 32.

    L : int, optional
        The length of the sequence to process. It is expected that the model processes data 
        in sequences of this length. Default is 3.

    batch_size : int, optional
        The number of samples to be processed in each batch. Default is 128.

    Returns:
    -------
    embeddings_array : numpy.ndarray
        A 2D numpy array of shape (num_samples, emb_dim) containing the computed embeddings 
        for each input.

    labels_array : numpy.ndarray
        A 1D numpy array of shape (num_samples,) containing the true labels for each input.

    Example:
    --------
    import torch
    from torch.utils.data import DataLoader

    # Assuming `autoencoder_model` is defined and `dataloader` is set up
    embeddings, labels = compute_embedding_autoencoder(
        autoencoder_model,
        dataloader,
        emb_dim=32,
        L=3,
        batch_size=128
    )

    Notes:
    ------
    - The function assumes that the autoencoder model has a method `nn.encode` that returns 
      the embeddings for the input data.
    - It requires PyTorch and numpy to be installed in your environment.
    - The function uses `torch.no_grad()` to disable gradient calculation, which is appropriate 
      during inference to save memory and improve performance.

    """
    num = len(dataloader)*batch_size
    emb_dim = emb_dim
    central_epoch = int((L - 1) / 2)
    embeddings_array = np.empty((num, emb_dim))
    labels_array = np.empty((num))
    device = model.device

    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            embedding = model.nn.encode(inputs.to(device))
            embeddings_array[i*batch_size : i*batch_size+inputs.size(0)] = embedding[:, central_epoch].cpu().numpy()
            labels_array[i*batch_size : i*batch_size+inputs.size(0)] = labels[:, central_epoch].cpu().numpy()

            if i == len(dataloader)-1:
                n_last_batch = inputs.size(0)
                first_empty_element = num-batch_size+n_last_batch
    
    return embeddings_array[:first_empty_element], labels_array[:first_empty_element]
