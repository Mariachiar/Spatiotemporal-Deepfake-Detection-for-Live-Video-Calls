# DeepFake Preprocessing Pipeline

Questo progetto implementa una pipeline per il preprocessing di video contenenti volti, con estrazione di landmark, Action Units (AU) e tracking multi-persona.  
È pensato per preparare i dati utili all’addestramento o alla valutazione di modelli di classificazione DeepFake.

## 🧠 Script principale

Lo script principale della pipeline è:

./preprocessing/preprocessing_parallel.py


## ⚙️ Come eseguire

Aprire un terminale nella cartella del progetto ed eseguire il seguente comando:

## Esempio

python ./preprocessing/preprocessing_parallel.py --input PATH --model preprocessing/yunet/face_detection_yunet_2023mar.onnx --mode save --vis --num_workers_per_frame 4 --output output.mp4



> Se vuoi usare la webcam in tempo reale, usa `--input 0`.

## 📥 Argomenti disponibili

| Parametro                      | Descrizione                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `--input`                     | Percorso al file video, cartella o `'0'` per la webcam                     |
| `--model`                     | Percorso al file YuNet `.onnx` per face detection                          |
| `--mode`                      | `save` per salvare su disco, `memory` per mantenere in RAM                 |
| `--vis`                       | Attiva visualizzazione con bounding box, landmark, ecc.                    |
| `--headless`                  | Disabilita qualsiasi finestra grafica (utile in ambiente server o Docker) |
| `--num_workers_per_frame`     | Numero di thread per frame (parallelismo MediaPipe)                        |
| `--show_faces`                | Mostra finestre separate con le facce ritagliate                           |
| `--yunet_res`                 | Risoluzione min. per YuNet (es. 320). 0 = usa risoluzione originale        |
| `--output`                    | Percorso per salvare il video con bounding box e landmark. Se omesso, non viene salvato. |

⚠️ Nota: in modalità `--headless`, il salvataggio video con `--output` è supportato, ma le finestre grafiche (`--vis`, `--show_faces`) vengono ignorate.


## 📁 Output

- I clip generati vengono salvati in:  
  `datasets/processed_dataset/`

- Ogni clip contiene:
  - `images.npy` e `images.pt`
  - `landmarks.npy`
  - `aus.npy`

- File di log:
  - `datasets/processed_dataset/master_clip_log.csv`  
  - `pipeline_performance_log.csv`
  - Grafici `total_pipeline_fps.png` e `time_per_component.png`

## 🛑 Interruzione

Premi `ESC` durante l’esecuzione per interrompere in sicurezza l'elaborazione.

## 📝 Requisiti

Assicurati di installare tutte le dipendenze richieste (es. `opencv`, `torch`, `mediapipe`, `matplotlib`, ecc.).

## 👩‍💻 Autrice

Mariachiara – Tesi di Laurea Magistrale  
Progetto: DeepFake Detection
