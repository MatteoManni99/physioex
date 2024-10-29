import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch

def compute_embedding_ssn_cem(ssn_cem_model, dataloader, emb_dim=128, con_emb_dim = 30, L=3, batch_size = 128):
    num = len(dataloader)*batch_size
    emb_dim = emb_dim
    con_emb_dim = con_emb_dim
    central_epoch = int((L - 1) / 2)

    embeddings_array = np.empty((num, emb_dim))
    concepts_array = np.empty((num))
    mse_array = np.empty((num, 15))
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
