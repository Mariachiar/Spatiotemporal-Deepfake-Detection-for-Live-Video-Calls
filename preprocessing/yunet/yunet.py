# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

"""
OpenCV (Open Source Computer Vision Library) è una libreria open source ampiamente utilizzata
per l'elaborazione di immagini, visione artificiale e machine learning. Fornisce strumenti per
lavorare con immagini e video, ed è usata per compiti come il rilevamento di oggetti, il tracciamento,
la manipolazione di immagini e molto altro.

Una delle funzionalità più potenti di OpenCV è il modulo `cv.dnn`, che permette di caricare ed eseguire
modelli di deep learning pre-addestrati in diversi formati, tra cui ONNX.

ONNX (Open Neural Network Exchange) è un formato aperto e standardizzato per rappresentare modelli
di reti neurali. È progettato per facilitare l'interoperabilità tra diversi framework di deep learning
come PyTorch, TensorFlow, Keras, e permette di eseguire modelli in ambienti diversi senza dipendere
dal framework originale con cui sono stati creati.

Nel nostro progetto, carichiamo il modello YuNet in formato ONNX tramite OpenCV e lo usiamo per
rilevare volti nei frame video. Questo approccio è leggero ed efficiente, e può essere facilmente
eseguito su CPU, GPU (es. con CUDA) o su dispositivi edge (es. OAK-D, MyriadX), semplicemente
cambiando il backend e il target di esecuzione.

In sintesi:
- OpenCV gestisce immagini/video e fornisce il motore di esecuzione per modelli AI.
- ONNX è il formato in cui è salvato il modello pre-addestrato (YuNet).
- La combinazione di entrambi ci consente di fare rilevamento facciale in tempo reale
  in modo portabile e indipendente dal framework originale del modello.
"""


from itertools import product

import numpy as np
import cv2 as cv

"""
modelPath: percorso al file ONNX (.onnx)
inputSize: dimensione fissa d'ingresso del modello (es. [320, 320])
confThreshold: soglia minima di confidenza per una detection valida
nmsThreshold: soglia di sovrapposizione per il Non-Maximum Suppression
topK: massimo numero di volti da rilevare per frame
backendId, targetId: selezionano CPU o GPU
"""
class YuNet:
    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.FaceDetectorYN.create( # API di OpenCV per caricare modelli ONNX preconfigurati per face detection.
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv.FaceDetectorYN.create( #Quando cambi backend/target, devi ricreare il modello da zero con i nuovi parametri — per questo si ricostruisce self._model.
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image): #Effettua l’inference (inferenza) sul frame di input e restituisce una lista di volti rilevati sotto forma di array NumPy, oppure un array vuoto se non ne rileva.
        # Forward
        """Chiama self._model.detect(image), che:
            Fa preprocessing automatico (resize, normalize)
            Passa l’immagine nel modello ONNX
            Applica NMS (Non-Maximum Suppression)
            Restituisce una tupla: (input_image, results) dove:
            results è un array N x 15 se sono presenti volti rilevati
            oppure None se nessun volto è stato trovato"""
        faces = self._model.detect(image)
        return np.empty(shape=(0, 5)) if faces[1] is None else faces[1]
    """Ritorna il risultato nel formato corretto:
        Se nessun volto è stato rilevato (faces[1] is None):
        Ritorna un array NumPy vuoto di shape (0, 5) (quindi nessuna detection).
        Altrimenti:
        Ritorna faces[1], ovvero l'array contenente tutte le facce rilevate."""
