import sys
import os
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import numpy as np
from .AU_Detection.solver_inference_image import solver_in_domain_image
from argparse import Namespace # Potrebbe non essere strettamente necessario qui se non usi Namespace altrove
from .AU_Detection.inference import ConfigObject


weights_download_dir = "C:\\Users\\maria\\Desktop\\deepfake\\preprocessing\\libreface\\weights_libreface"
os.makedirs(f"{weights_download_dir}/AU_Detection/weights", exist_ok=True)

# Variabile globale per l'istanza del modello AU (inizializzazione pigra)
_au_model_instance = None
_au_transform = None # Anche la trasformazione può essere inizializzata una volta

def _initialize_au_model():
    """
    Inizializza il modello AU di LibreFace se non è già stato inizializzato.
    Questa funzione deve essere chiamata esplicitamente all'inizio del programma principale.
    """
    global _au_model_instance, _au_transform

    if _au_model_instance is None:
        print("LibreFace Adapter: Inizializzazione del modello AU...")

        # Configurazione predefinita (spostata all'interno della funzione per chiarezza,
        # o potrebbe essere una variabile globale se preferisci)
        au_config = ConfigObject({
            'seed': 0,
            'data_root': '',
            'ckpt_path': f'{weights_download_dir}/AU_Detection/weights/resnet.pt',
            'weights_download_id': '17v_vxQ09upLG3Yh0Zlx12rpblP7uoA8x',
            'data': 'BP4D',
            'fold': 'all',
            'image_size': 256,
            'crop_size': 224,
            'num_labels': 12,
            'sigma': 10.0,
            'model_name': 'resnet',
            'dropout': 0.1,
            'hidden_dim': 128,
            'half_precision': False,
            'num_epochs': 30,
            'interval': 500,
            'threshold': 0.0,
            'batch_size': 256,
            'learning_rate': '3e-5',
            'weight_decay': '1e-4',
            'loss': 'unweighted',
            'clip': 1.0,
            'when': 10,
            'patience': 5,
            'fm_distillation': False,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu' # Rendi dinamico il device
        })

        try:
            _au_model_instance = solver_in_domain_image(config=au_config)
            _au_model_instance.load_best_ckpt()
            #print("LibreFace Adapter: Modello AU caricato con successo.")
            #print(f"LibreFace Adapter: Modello caricato su dispositivo: {_au_model_instance.device}")
            
            # Inizializza la trasformazione qui, dopo aver impostato il modello
            _au_transform = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])

        except Exception as e:
            print(f"ERRORE CRITICO LibreFace Adapter: Impossibile inizializzare o caricare il modello AU. Dettagli: {e}")
            _au_model_instance = None # Imposta a None per gestire l'errore in get_au_from_face_ndarray
            _au_transform = None
            raise # Rilancia l'eccezione per fermare il programma principale
    else:
        print("LibreFace Adapter: Modello AU già inizializzato.")


def get_au_from_face_ndarray(face_batch: list[np.ndarray]) -> list[dict]:
    if _au_model_instance is None or _au_transform is None:
        print("AVVISO: LibreFace AU Model non inizializzato. Restituzione placeholder AU.")
        return [{
            "AU01": 0.0, "AU02": 0.0, "AU04": 0.0,
            "AU06": 0.0, "AU07": 0.0, "AU10": 0.0,
            "AU12": 0.0, "AU14": 0.0, "AU15": 0.0,
            "AU17": 0.0, "AU23": 0.0, "AU24": 0.0
        } for _ in face_batch]

    results = []
    # Raccogli tutti i tensori trasformati per un'inferenza batch se possibile
    transformed_tensors = []
    for face_array in face_batch:
        try:
            pil_image = Image.fromarray(face_array)
            transformed_tensors.append(_au_transform(pil_image))
        except Exception as e:
            print(f"ERRORE LibreFace Adapter: Errore nella trasformazione immagine. Dettagli: {e}")
            # Se una trasformazione fallisce, potremmo dover omettere questa immagine
            # o aggiungere un tensore "dummy" per mantenere le dimensioni del batch.
            # Per ora, in caso di errore, aggiungiamo AU zero.
            results.append({
                "AU01": 0.0, "AU02": 0.0, "AU04": 0.0,
                "AU06": 0.0, "AU07": 0.0, "AU10": 0.0,
                "AU12": 0.0, "AU14": 0.0, "AU15": 0.0,
                "AU17": 0.0, "AU23": 0.0, "AU24": 0.0
            })
            continue # Salta al prossimo volto

    if not transformed_tensors: # Nessun volto valido da processare
        return results

    # Unisci tutti i tensori trasformati in un unico batch tensor
    # Ogni transformed_tensor dovrebbe essere (C, H, W)
    # Stack li trasforma in (B, C, H, W)
    batch_tensor = torch.stack(transformed_tensors).to(_au_model_instance.device)

    try:
        # Esegui l'inferenza in batch se il modello lo supporta
        # Se image_inference supporta batch, questa sarà più efficiente
        #print(f"DEBUG: Dimensioni batch_tensor = {batch_tensor.shape}")

        batch_results_tensor = _au_model_instance.image_inference_batch(batch_tensor)
        
        # I risultati saranno un tensore (Batch_size, Num_AUs)
        # Converti i risultati in liste di dizionari AU
        for i, single_result_tensor in enumerate(batch_results_tensor):
            # Assicurati che single_result_tensor sia su CPU per tolist()
            aus = dict(zip(_au_model_instance.aus, single_result_tensor.cpu().squeeze().tolist()))
            results.append(aus)
        
    except Exception as e:
        print(f"ERRORE LibreFace Adapter: Errore durante l'inferenza AU in batch. Dettagli: {e}")
        # In caso di errore batch, aggiungi placeholder per tutti i volti che dovevano essere processati
        # Questo potrebbe sovrascrivere risultati parziali se l'errore non è all'inizio
        # Ma è una buona fallback per evitare crash.
        return [{
            "AU01": 0.0, "AU02": 0.0, "AU04": 0.0,
            "AU06": 0.0, "AU07": 0.0, "AU10": 0.0,
            "AU12": 0.0, "AU14": 0.0, "AU15": 0.0,
            "AU17": 0.0, "AU23": 0.0, "AU24": 0.0
        } for _ in face_batch] # Ritorna un numero di risultati pari all'input originale

    return results
