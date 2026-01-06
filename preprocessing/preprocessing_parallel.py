import os
# Set environment variable to allow duplicate OpenMP libraries, which can prevent crashes with some setups.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
from argparse import Namespace
import sys
import threading

# Add the parent directory to the system path to allow imports from the 'preprocessing' module.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Core Component Imports ---
from preprocessing.yunet.yunet import YuNet
from preprocessing.ByteTrack.byte_tracker import BYTETracker, STrack, TrackState
from preprocessing.ByteTrack.basetrack import BaseTrack

# --- Feature Extraction Imports ---
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import atexit

# --- Utility Imports ---
from tqdm import tqdm
import queue
from concurrent.futures import ThreadPoolExecutor

# --- Global Threading Events ---
# Event to signal that the final cleanup function has been called.
cleanup_called = threading.Event()
# Event to signal that the ESC key has been pressed.
esc_pressed = threading.Event()

# --- Queue for Asynchronous File Writing ---
clip_writer_queue = queue.Queue()
stop_writer_thread = threading.Event()

def esc_listener():
    """
    Listens for the ESC key press in a background thread.
    This function is cross-platform, supporting both Unix-like systems and Windows.
    """
    try:
        # --- Unix-like systems (Linux, macOS) ---
        import termios, tty, select
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not esc_pressed.is_set():
                # Check for input with a short timeout.
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    if sys.stdin.read(1) == '\x1b': # '\x1b' is the ASCII code for ESC.
                        print("\n[INFO] Tasto ESC premuto. Chiusura in corso...")
                        esc_pressed.set()
                        break
        finally:
            # Restore original terminal settings.
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (ImportError, AttributeError, OSError):
        try:
            # --- Windows ---
            import msvcrt
            while not esc_pressed.is_set():
                if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                    print("\n[INFO] Tasto ESC premuto (Windows). Chiusura in corso...")
                    esc_pressed.set()
                    break
                time.sleep(0.1) # Prevent high CPU usage.
        except ImportError:
            print("[WARN] Nessun metodo disponibile per ascoltare il tasto ESC. L'applicazione dovrà essere chiusa manualmente.")


# --- LibreFace (Action Unit Extraction) ---
# Attempt to import LibreFace; if it fails, use a placeholder function.
try:
    from preprocessing.libreface.libreface_adapter import get_au_from_face_ndarray, _initialize_au_model as libreface_init_au_model
    print("[INFO] LibreFace importato con successo per l'estrazione delle AU.")
    _libreface_available = True
except ImportError:
    print("[WARN] LibreFace non trovato. Verrà usata una funzione placeholder per l'estrazione delle AU.")
    def get_au_from_face_ndarray(face_rgbs_batch):
        """Placeholder function to simulate AU extraction when LibreFace is not available."""
        time.sleep(0.005 * len(face_rgbs_batch)) # Simulate processing time.
        # Return a dictionary of random AU values for each face in the batch.
        return [{"AU{:02d}".format(i): np.random.rand() for i in [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]} for _ in face_rgbs_batch]
    _libreface_available = False

# --- Constants ---
CLIP_LENGTH = 8         # Number of frames per saved clip.
CLIP_STEP = 4           # Sliding window step for creating clips.
CLIP_SIZE = (224, 224)  # Resolution for cropped face images.
AU_CLIP_LENGTH = 8      # Number of AU frames per clip.
AU_CLIP_STEP = 4
LAND_CLIP_LENGTH = 8    # Number of landmark frames per clip.
LAND_CLIP_STEP = 4
OUTPUT_BASE_DIR = "./datasets/processed_dataset"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)


# --- DNN Backend and Target Pairs for OpenCV ---
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA]
]

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Real-time Multi-person Deepfake Preprocessing Pipeline")
parser.add_argument('--model', '-m', type=str, default=os.path.join('preprocessing', 'yunet', 'face_detection_yunet_2023mar.onnx'), help='Percorso al modello ONNX di YuNet per il rilevamento dei volti.')
parser.add_argument('--backend_target', '-bt', type=int, default=0, help='Backend e target per YuNet. 0: OpenCV-CPU (default), 1: CUDA-GPU.')
parser.add_argument('--mode', type=str, choices=['save', 'memory'], default='save', help='Modalità di gestione delle clip: "save" su disco, "memory" in RAM.')
parser.add_argument('--vis', '-v', action='store_true', help='Abilita la visualizzazione in tempo reale con bounding box, ID, landmark e FPS.')
parser.add_argument('--num_workers_per_frame', type=int, default=os.cpu_count() or 1, help='Numero di thread per l\'elaborazione parallela dei volti (MediaPipe).')
parser.add_argument('--show_faces', action='store_true', help='Mostra finestre separate per ogni volto ritagliato con i landmark.')
parser.add_argument('--yunet_res', type=int, default=0, help='Risoluzione del lato più corto per il ridimensionamento dell\'input di YuNet (es. 320). 0 per la risoluzione originale.')
parser.add_argument('--input', '-i', type=str, default='0', help="Percorso a un file video o una cartella. '0' per la webcam.")
parser.add_argument('--headless', action='store_true', help='Disabilita tutte le visualizzazioni e i grafici (per ambienti headless/Docker).')
parser.add_argument('--output', type=str, default=None, help="Percorso per salvare il video di output. Se non impostato, nessun video verrà salvato.")
parser.add_argument('--frame_skip', type=int, default=1, help='Elabora solo un frame ogni N.')
args = parser.parse_args()


# --- Global Data Structures ---
global_clip_index = 0
all_clip_logs = []
thread_local_storage = threading.local()

def writer_worker(base_output_dir, input_base):
    """
    Thread di scrittura: salva le clip dalla coda su disco.
    """
    global global_clip_index, all_clip_logs
    #print("[INFO] Thread di scrittura avviato.")

    while not stop_writer_thread.is_set() or not clip_writer_queue.empty():
        try:
            #("[DEBUG] In attesa clip nella coda...")
            clip_task = clip_writer_queue.get(timeout=1)
            #print("[DEBUG] Clip ricevuta dalla coda.")

            if clip_task is None:
                #print("[DEBUG] Task nullo, continuo...")
                clip_writer_queue.task_done()
                continue

            (source_name, track_id, clip_idx, img_clip_data,
             landmarks_clip_data, aus_clip_data, frame_start_id,
             frame_end_id, full_video_path) = clip_task

            try:
                # --- Struttura directory ---
                if full_video_path and input_base and os.path.isdir(input_base):
                    relative_path = os.path.relpath(full_video_path, input_base)
                    relative_path_no_ext = os.path.splitext(relative_path)[0]
                    track_output_dir = os.path.join(base_output_dir, relative_path_no_ext, f"track_{track_id}")
                else:
                    source_name_no_ext = os.path.splitext(source_name)[0]
                    track_output_dir = os.path.join(base_output_dir, source_name_no_ext, f"track_{track_id}")

                clip_output_dir = os.path.join(track_output_dir, f"clip_{clip_idx:05d}")
                os.makedirs(clip_output_dir, exist_ok=True)

                #print(f"[SALVATAGGIO] Clip salvata in: {clip_output_dir}")

                # --- Salvataggio immagini ---
                np.save(os.path.join(clip_output_dir, "images.npy"), img_clip_data)
                torch.save(
                    torch.tensor(img_clip_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0,
                    os.path.join(clip_output_dir, "images.pt")
                )

                # --- Salvataggio landmarks ---
                serializable_landmarks = []
                for frame_landmarks in landmarks_clip_data:
                    if frame_landmarks:
                        serializable_landmarks.append(
                            [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in frame_landmarks.landmark]
                        )
                    else:
                        serializable_landmarks.append([])

                np.save(os.path.join(clip_output_dir, "landmarks.npy"), np.array(serializable_landmarks, dtype=object))

                # --- Salvataggio AUs ---
                np.save(os.path.join(clip_output_dir, "aus.npy"), np.array(aus_clip_data, dtype=object))

                # --- Log ---
                log_entry = {
                    "global_clip_id": global_clip_index,
                    "source_name": os.path.splitext(source_name)[0],
                    "track_id": track_id,
                    "clip_idx_in_track": clip_idx,
                    "clip_path": os.path.relpath(clip_output_dir, base_output_dir),
                    "frame_start_id": frame_start_id,
                    "frame_end_id": frame_end_id,
                    "clip_length_frames": CLIP_LENGTH,
                    "clip_size_pixels": f"{CLIP_SIZE[0]}x{CLIP_SIZE[1]}"
                }
                all_clip_logs.append(log_entry)
                global_clip_index += 1

            except Exception as e:
                print(f"[ERRORE] Errore durante il salvataggio della clip {clip_idx} della traccia {track_id}: {e}")

            finally:
                # ✅ Sempre segnalare fine task, anche in caso di errore
                clip_writer_queue.task_done()

        except queue.Empty:
            continue

    print("[INFO] Thread di scrittura terminato.")



def _get_face_mesh_detector():
    """
    Inizializza un'istanza di MediaPipe FaceMesh per il thread corrente se non esiste.
    Questo è necessario perché gli oggetti MediaPipe non sono thread-safe.
    """
    if not hasattr(thread_local_storage, 'face_mesh_detector'):
        print(f"[INFO] Inizializzazione di MediaPipe FaceMesh per il thread {threading.get_ident()}...")
        thread_local_storage.face_mesh_detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return thread_local_storage.face_mesh_detector

def _process_face_mesh_for_thread(face_rgb_input):
    """
    Elabora una singola immagine di un volto con MediaPipe FaceMesh per estrarre i landmark.
    Questa funzione viene eseguita dai thread worker nel ThreadPoolExecutor.
    """
    face_mesh_detector_instance = _get_face_mesh_detector()
    start_time = time.time()
    results_mesh = face_mesh_detector_instance.process(face_rgb_input)
    processing_time = time.time() - start_time
    landmarks = results_mesh.multi_face_landmarks[0] if results_mesh.multi_face_landmarks else None
    return landmarks, processing_time

def detect_and_track(frame, face_detector, tracker, yunet_input_size, frame_log):
    """Esegue il rilevamento dei volti (YuNet) e il tracciamento (ByteTrack) su un singolo frame."""
    h, w = frame.shape[:2]
    frame_for_yunet = frame
    if yunet_input_size != [w, h]:
        frame_for_yunet = cv2.resize(frame, (yunet_input_size[0], yunet_input_size[1]))

    start_time = time.time()
    detections = face_detector.infer(frame_for_yunet)
    frame_log["yunet_inference_time"] = time.time() - start_time

    if yunet_input_size != [w, h]:
        scale_x, scale_y = w / yunet_input_size[0], h / yunet_input_size[1]
        for det in detections:
            det[0], det[2] = det[0] * scale_x, det[2] * scale_x
            det[1], det[3] = det[1] * scale_y, det[3] * scale_y

    faces_detected_for_tracking = [STrack(det[:4], score=det[-1]) for det in detections]
    start_time = time.time()
    online_targets = tracker.update(faces_detected_for_tracking, (h, w), (w, h))
    frame_log["bytetrack_update_time"] = time.time() - start_time

    return online_targets, h, w

def preprocess_and_extract_features(frame, online_targets, frame_log, frame_id, last_feature_extraction_frame, last_known_aus):
    """
    Ritaglia i volti dal frame ed estrae le feature (AU, landmark).
    Utilizza un'ottimizzazione per saltare l'estrazione delle feature per alcuni frame per migliorare le prestazioni.
    """
    start_time = time.time()
    faces_data = []
    faces_to_process_fully = []

    for track in online_targets:
        if not track.is_activated or track.state == TrackState.Lost:
            continue

        x1, y1, w_box, h_box = map(int, track.tlwh)
        x2, y2 = x1 + w_box, y1 + h_box
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

        face_cropped = frame[y1:y2, x1:x2]
        if face_cropped.size == 0: continue

        face_resized = cv2.resize(face_cropped, CLIP_SIZE)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        track_id = track.track_id
        current_face_data = {
            "track_id": track_id, "face_rgb": face_rgb, "bbox": (x1, y1, x2, y2),
            "original_bbox_dims": (w_box, h_box)
        }
        faces_data.append(current_face_data)

        if frame_id >= last_feature_extraction_frame.get(track_id, -1):
            faces_to_process_fully.append(current_face_data)
            last_feature_extraction_frame[track_id] = frame_id

    frame_log["face_preprocessing_time"] = time.time() - start_time
    au_results_map = {}
    if faces_to_process_fully:
        start_au_time = time.time()
        faces_rgb_batch = [data["face_rgb"] for data in faces_to_process_fully]
        batched_aus = get_au_from_face_ndarray(faces_rgb_batch)
        for i, data in enumerate(faces_to_process_fully):
            au_results_map[data["track_id"]] = batched_aus[i]
            last_known_aus[data["track_id"]] = batched_aus[i]
        frame_log["au_extraction_time"] = time.time() - start_au_time
    else:
        frame_log["au_extraction_time"] = 0.0

    frame_log["num_faces"] = len(faces_data)
    return faces_data, faces_to_process_fully, au_results_map


def submit_tasks_to_executor(faces_data, au_results_map, executor):
    """Invia le attività di analisi FaceMesh al ThreadPoolExecutor per l'elaborazione parallela."""
    new_futures = []
    for face_data in faces_data:
        future = executor.submit(_process_face_mesh_for_thread, face_data["face_rgb"])
        new_futures.append({
            "future": future, "track_id": face_data["track_id"], "face_rgb": face_data["face_rgb"],
            "bbox": face_data["bbox"], "original_bbox_dims": face_data["original_bbox_dims"],
            "aus_pred": au_results_map.get(face_data["track_id"])
        })
    return new_futures

def collect_completed_futures(active_futures):
    """
    Controlla le attività FaceMesh completate, raccoglie i loro risultati
    e restituisce le attività ancora attive.
    """
    completed_results, remaining_futures = [], []
    for task in active_futures:
        if task["future"].done():
            try:
                landmarks, process_time = task["future"].result()
                if landmarks:
                    task["face_landmarks"] = landmarks
                    task["processing_time"] = process_time
                    completed_results.append(task)
            except Exception as e:
                print(f"[ERRORE] L'attività per track_id {task.get('track_id')} è fallita: {e}")
        else:
            remaining_futures.append(task)
    return completed_results, remaining_futures

def handle_clip_buffers(result, clip_buffer, au_buffer, land_buffer, frame_id, clips_in_ram,
                        video_clip_logs, track_clip_counters, current_source_name, video_path, last_known_data):
    """
    Gestisce i buffer dei dati per ogni traccia, riempiendo i buchi nei dati AU/landmark
    con gli ultimi valori noti. Salva clip quando i buffer raggiungono la lunghezza richiesta.
    """
    track_id = result["track_id"]

    # Inizializza buffer e stato noto se prima volta
    for buffer in [clip_buffer, au_buffer, land_buffer]:
        buffer.setdefault(track_id, [])
    last_known_data.setdefault(track_id, {"aus": None, "landmarks": None})

    # Aggiunge immagine attuale
    clip_buffer[track_id].append(result["face_rgb"])

    # Aggiorna dati se presenti
    if "aus_pred" in result and result["aus_pred"] is not None:
        last_known_data[track_id]["aus"] = result["aus_pred"]
    if "face_landmarks" in result and result["face_landmarks"] is not None:
        last_known_data[track_id]["landmarks"] = result["face_landmarks"]

    # Inserisce sempre gli ultimi dati validi nei buffer
    au_buffer[track_id].append(last_known_data[track_id]["aus"])
    land_buffer[track_id].append(last_known_data[track_id]["landmarks"])

    # Debug buffer status
    #print(f"[DEBUG] Track {track_id} | Frame {frame_id} | IMG={len(clip_buffer[track_id])}, AU={len(au_buffer[track_id])}, LAND={len(land_buffer[track_id])}")

    if len(clip_buffer[track_id]) >= CLIP_LENGTH:
        au_sequence = [item for item in au_buffer[track_id][:CLIP_LENGTH] if item is not None]
        land_sequence = [item for item in land_buffer[track_id][:CLIP_LENGTH] if item is not None]

        if len(au_sequence) >= AU_CLIP_LENGTH and len(land_sequence) >= LAND_CLIP_LENGTH:
            clip_data = np.stack(clip_buffer[track_id][:CLIP_LENGTH])

            if args.mode == "save":
                track_clip_counters.setdefault(track_id, 0)
                clip_idx = track_clip_counters[track_id]

                clip_task = (
                    current_source_name, track_id, clip_idx,
                    clip_data, land_sequence, au_sequence,
                    frame_id - CLIP_LENGTH + 1, frame_id, video_path
                )

                #print(f"[DEBUG] ✅ Clip messa in coda: track_id={track_id}, clip_idx={clip_idx}")
                clip_writer_queue.put(clip_task)
                track_clip_counters[track_id] += 1

            elif args.mode == "memory":
                images_tensor = torch.tensor(clip_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
                clips_in_ram.append({
                    "track_id": track_id,
                    "images": images_tensor,
                    "aus": au_sequence,
                    "landmarks": land_sequence
                })

        #else:
            #print(f"[DEBUG] ⛔ Clip non salvata: dati incompleti — AU={len(au_sequence)}, LAND={len(land_sequence)}")

        # Rimozione elementi vecchi dai buffer (sliding window)
        clip_buffer[track_id] = clip_buffer[track_id][CLIP_STEP:]
        au_buffer[track_id] = au_buffer[track_id][AU_CLIP_STEP:]
        land_buffer[track_id] = land_buffer[track_id][LAND_CLIP_STEP:]



def draw_visualizations(frame, tracked_faces, mesh_results, img_w, img_h, frame_log, frame_id):
    """Annota il frame con bounding box, ID e landmark per la visualizzazione."""

    for face in tracked_faces:
        x1, y1, x2, y2 = face["bbox"]
        track_id = face["track_id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    for mp_data in mesh_results:
        face_landmarks = mp_data.get("face_landmarks")
        if face_landmarks is None:
            continue

        bbox_x1, bbox_y1, _, _ = mp_data["bbox"]
        w_box, h_box = mp_data["original_bbox_dims"]
        x1f, y1f = float(bbox_x1), float(bbox_y1)

        adjusted_landmarks = []
        for lm in face_landmarks.landmark:
            px = lm.x * w_box + x1f
            py = lm.y * h_box + y1f
            adjusted_landmarks.append(landmark_pb2.NormalizedLandmark(
                x=px / img_w, y=py / img_h, z=lm.z))

        temp_landmarks_proto = landmark_pb2.NormalizedLandmarkList(landmark=adjusted_landmarks)

        mp_drawing.draw_landmarks(
            image=frame, landmark_list=temp_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=frame, landmark_list=temp_landmarks_proto,
            connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        if args.vis and args.show_faces and not args.headless:
            debug_face_img = cv2.cvtColor(mp_data["face_rgb"].copy(), cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image=debug_face_img, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=debug_face_img, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            cv2.imshow(f"Volto Ritagliato ID {mp_data['track_id']}", debug_face_img)
            cv2.waitKey(1)

    if not args.headless and 'frame' in locals():
        if args.vis:
            cv2.putText(frame, f"FPS: {frame_log['total_pipeline_fps']:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Tracking & Analisi Facciale", frame)
            cv2.waitKey(1)
        else:
            status_frame = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(status_frame, f"Elaborazione Frame: {frame_id}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(status_frame, f"FPS Totale: {frame_log['total_pipeline_fps']:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(status_frame, "Premi ESC per uscire", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Stato Pipeline (Premi ESC per uscire)", status_frame)
            cv2.waitKey(1)


def cleanup():
    """
    Funzione di pulizia finale da chiamare all'uscita.
    Chiude i thread, salva i log e genera i grafici delle prestazioni.
    """
    if cleanup_called.is_set(): return
    cleanup_called.set()
    print("\n[INFO] Esecuzione pulizia finale...")

    if "executor" in globals() and executor is not None:
        try:
            executor.shutdown(wait=True)
            print("[INFO] ThreadPoolExecutor (MediaPipe) terminato.")
        except Exception as e:
            print(f"[WARN] Errore durante la chiusura del ThreadPoolExecutor: {e}")

    if "writer_thread" in globals() and writer_thread.is_alive():
        print("[INFO] In attesa che il thread di scrittura finisca...")
        stop_writer_thread.set()
        writer_thread.join(timeout=10)
        if writer_thread.is_alive():
            print("[WARN] Il thread di scrittura non è terminato entro 10 secondi.")
        else:
            print("[INFO] Thread di scrittura completato.")

    if all_clip_logs:
        clips_df = pd.DataFrame(all_clip_logs)
        master_log_path = os.path.join(OUTPUT_BASE_DIR, "master_clip_log.csv")
        clips_df.to_csv(master_log_path, index=False)
        print(f"✅ Log master delle clip con {len(clips_df)} voci salvato in: {master_log_path}")

    if all_pipeline_logs and not args.headless:
        log_df = pd.DataFrame(all_pipeline_logs)
        log_df_path = "pipeline_performance_log.csv"
        log_df.to_csv(log_df_path, index=False)
        print(f"✅ Log aggregato delle prestazioni per {len(log_df)} frame salvato in: {log_df_path}")

        plt.figure(figsize=(14, 7))
        plt.plot(log_df.index, log_df["total_pipeline_fps"], label='Total Pipeline FPS', color='b', alpha=0.7)
        plt.title("Prestazioni Complessive della Pipeline (FPS) su Tutti i Video")
        plt.xlabel("Numero Frame (Complessivo)")
        plt.ylabel("Frame Per Secondo (FPS)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("total_pipeline_fps.png")
        plt.show()

        plt.figure(figsize=(14, 9))
        time_cols = ["read_frame_time", "yunet_inference_time", "bytetrack_update_time",
                     "face_preprocessing_time", "au_extraction_time",
                     "mediapipe_parallel_wall_time", "clip_handling_time", "drawing_time"]
        for col in time_cols:
            if col in log_df.columns:
                plt.plot(log_df.index, log_df[col], label=col.replace("_", " ").title())

        plt.title("Tempo di Esecuzione per Componente della Pipeline (Secondi)")
        plt.xlabel("Numero Frame (Complessivo)")
        plt.ylabel("Tempo (secondi)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig("time_per_component.png")
        plt.show()

    if all_pipeline_logs:
        df = pd.DataFrame(all_pipeline_logs)
        fps_medio = 1.0 / df["total_processing_time"].mean()
        print(f"\n✅ FPS medio globale: {fps_medio:.2f}")

    if not args.headless:
        cv2.destroyAllWindows()
    print("[INFO] Pulizia completata.")


atexit.register(cleanup)

if __name__ == "__main__" :
    executor = None
    all_pipeline_logs = []

    video_paths = []
    if args.input == '0':
        if args.headless:
            print("[ERRORE] L'input da webcam ('0') non può essere usato in modalità headless.")
            sys.exit(1)
        video_paths = ['0']
    elif os.path.isfile(args.input):
        video_paths = [args.input]
    elif os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    full_path = os.path.join(root, file)
                    video_paths.append(full_path)
        if not video_paths:
            print(f"[ERRORE] Nessun file video valido trovato ricorsivamente in '{args.input}'.")
            sys.exit(1)
    else:
        print(f"[ERRORE] Il percorso di input fornito non è un file o una directory valida: '{args.input}'")
        sys.exit(1)
    print("ciao")
    already_processed_files = set()
    print("[DEBUG] ➕ Inizio scansione dei video effettivamente processati...")

    for root, dirs, files in os.walk(OUTPUT_BASE_DIR):
        for d in dirs:
            if d.startswith("track_"):  # Trova solo tracce effettive
                parent_path = os.path.relpath(root, OUTPUT_BASE_DIR)  # es: FaceForensics++/original/000_003
                video_rel_path = os.path.splitext(parent_path)[0]  # es: Deepfakes/000_003
                already_processed_files.add(video_rel_path)
                print(f"[TROVATA] Traccia già presente per video: {video_rel_path}")



    print(f"[DEBUG] ✅ Totale video già processati (cartelle): {len(already_processed_files)}")
    print("prova1")

    # Filtro dei video già elaborati
    original_video_count = len(video_paths)
    video_paths = [ vp for vp in video_paths if os.path.splitext(os.path.relpath(vp, args.input))[0] not in already_processed_files]



    print(f"[DEBUG] 🎥 Video raccolti inizialmente: {original_video_count}")
    print(f"[DEBUG] 🔍 Video da processare (dopo filtro): {len(video_paths)}")
    for vp in video_paths:
        print(f"   - {vp}")

    print(f"[DEBUG] Video già processati (cartelle): {sorted(list(already_processed_files))}")


    threading.Thread(target=esc_listener, daemon=True).start()
    writer_thread = threading.Thread(target=writer_worker, args=(OUTPUT_BASE_DIR, args.input), daemon=True)
    writer_thread.start()
    print(f"[DEBUG] 🧵 Thread di scrittura avviato con base: {OUTPUT_BASE_DIR}")


    if not args.headless:
        cv2.namedWindow("Tracking & Analisi Facciale", cv2.WINDOW_NORMAL)
        loading_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(loading_img, "Caricamento modelli... Attendere.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Tracking & Analisi Facciale", loading_img)
        cv2.waitKey(1)

    if _libreface_available:
        try:
            libreface_init_au_model()
            print("[INFO] Modello AU di LibreFace caricato.")
        except Exception as e:
            print(f"[ERRORE] Impossibile inizializzare il modello AU di LibreFace: {e}"); sys.exit(1)

    backend_id, target_id = backend_target_pairs[args.backend_target]
    face_detector = YuNet(modelPath=args.model, inputSize=[640, 480], confThreshold=0.9, nmsThreshold=0.3, topK=500, backendId=backend_id, targetId=target_id)
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    executor = ThreadPoolExecutor(max_workers=args.num_workers_per_frame)

    progress_bar = None

    try:
        for video_path in video_paths:
            if esc_pressed.is_set(): break

            source_name = "webcam" if video_path == '0' else os.path.basename(video_path)

            if source_name != "webcam" and source_name in already_processed_files:
                print(f"[INFO] '{source_name}' è già stato processato. Salto.")
                continue



            if hasattr(STrack, "_count"):
                STrack._count = 0
            if hasattr(BaseTrack, "_count"):
                BaseTrack._count = 0
            print("[DEBUG] Contatore track_id azzerato.")


            cap = cv2.VideoCapture(int(video_path) if video_path == '0' else video_path)
            if not cap.isOpened():
                print(f"[ERRORE] Impossibile aprire il video '{video_path}'. Salto."); continue

            print(f"\n--- In elaborazione: {source_name} ---")
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            video_writer = None
            if args.output:
                if os.path.isdir(args.output) or not os.path.splitext(args.output)[1]:
                    os.makedirs(args.output, exist_ok=True)
                    output_filename = os.path.splitext(source_name)[0] + "_processed.mp4"
                    output_path = os.path.join(args.output, output_filename)
                else:
                    output_path = args.output
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                if not output_path.lower().endswith(('.mp4', '.avi')):
                    print(f"[ERRORE] Il file di output deve avere estensione .mp4 o .avi. Ricevuto: {output_path}")
                    sys.exit(1)

                video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w, h))
                if not video_writer.isOpened():
                    print(f"[ERRORE] Impossibile creare VideoWriter per il percorso: {output_path}")
                    sys.exit(1)
                print(f"[INFO] Il video di output verrà salvato in: {output_path}")
            else:
                print("[INFO] Il video di output NON verrà salvato.")

            if args.yunet_res > 0:
                aspect_ratio = w / h
                if w > h: new_w, new_h = int(args.yunet_res * aspect_ratio), args.yunet_res
                else: new_w, new_h = args.yunet_res, int(args.yunet_res / aspect_ratio)
                yunet_input_size = [new_w, new_h]
            else:
                yunet_input_size = [w, h]
            face_detector.setInputSize(yunet_input_size)

            tracker = BYTETracker(Namespace(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30, mot20=False), frame_rate=30)
            frame_id, active_futures = 0, []
            clip_buffer, au_buffer, land_buffer = {}, {}, {}
            track_clip_counters = {}
            pipeline_logs_for_this_video, video_clip_logs, clips_in_ram = [], [], []
            last_feature_extraction_frame = {}
            last_known_aus = {}
            last_known_data_for_clip = {}

            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    progress_bar = tqdm(total=total_frames, desc=f"Elaborazione {source_name}", unit="frame")
                else: progress_bar = None
            except:
                progress_bar = None

            try:
                while cap.isOpened():
                    if esc_pressed.is_set(): raise InterruptedError("ESC pressed.")

                    start_total_frame_time = time.time()
                    frame_log = {"frame_id": frame_id, "source_name": source_name, "clip_handling_time": 0.0}

                    ret, frame = cap.read()
                    if not ret: break
                    frame_log["read_frame_time"] = time.time() - start_total_frame_time

                    if frame_id % args.frame_skip != 0:
                        frame_id += 1
                        if progress_bar: progress_bar.update(1)
                        continue

                    online_targets, img_h, img_w = detect_and_track(frame, face_detector, tracker, yunet_input_size, frame_log)

                    all_faces, faces_for_full_process, au_map = preprocess_and_extract_features(
                        frame, online_targets, frame_log, frame_id, last_feature_extraction_frame, last_known_aus)

                    start_mediapipe_wall_time = time.time()
                    if faces_for_full_process:
                        new_tasks = submit_tasks_to_executor(faces_for_full_process, au_map, executor)
                        active_futures.extend(new_tasks)

                    completed_results, active_futures = collect_completed_futures(active_futures)
                    frame_log["mediapipe_parallel_wall_time"] = time.time() - start_mediapipe_wall_time
                    
                    total_mediapipe_thread_time = sum(res.get("processing_time", 0) for res in completed_results)
                    start_clip_time = time.time()

                    results_map = {res["track_id"]: res for res in completed_results}

                    for face_data in all_faces:
                        track_id = face_data["track_id"]
                        result_for_buffer = results_map.get(track_id, face_data)
                        handle_clip_buffers(result_for_buffer, clip_buffer, au_buffer, land_buffer, frame_id, clips_in_ram,
                                            video_clip_logs, track_clip_counters, source_name, video_path, last_known_data_for_clip)

                    frame_log["clip_handling_time"] = time.time() - start_clip_time
                    frame_log["mediapipe_thread_time_sum"] = total_mediapipe_thread_time

                    total_frame_time = time.time() - start_total_frame_time
                    frame_log["total_processing_time"] = total_frame_time
                    fps = 1.0 / total_frame_time if total_frame_time > 0 else 0
                    frame_log["total_pipeline_fps"] = fps
                    pipeline_logs_for_this_video.append(frame_log)
                    
                    start_draw_time = time.time()
                    draw_visualizations(frame, all_faces, completed_results, img_w, img_h, frame_log, frame_id)
                    frame_log["drawing_time"] = time.time() - start_draw_time
                    
                    if video_writer:
                        video_writer.write(frame)

                    frame_id += 1

                    if progress_bar:
                        progress_bar.update(1)

                    if esc_pressed.is_set():
                        raise InterruptedError("ESC premuto da tastiera.")

            except (KeyboardInterrupt, InterruptedError) as e:
                print(f"\n[INFO] Interruzione rilevata ({e}). Procedo alla pulizia per questo video...")
            finally:
                if cap.isOpened(): cap.release()
                if progress_bar: progress_bar.close()
                if 'video_writer' in locals() and video_writer and video_writer.isOpened(): video_writer.release()
                all_pipeline_logs.extend(pipeline_logs_for_this_video)
    finally:
        cleanup()
