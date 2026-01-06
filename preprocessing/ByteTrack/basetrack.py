import numpy as np
from collections import OrderedDict #mantiene l’ordine di inserimento degli elementi in un dizionario, utile per tenere traccia della storia del tracciamento nel tempo.

"""Questa classe enum simula uno stato associato a ciascun oggetto tracciato:
New: oggetto appena rilevato, non ancora confermato
Tracked: oggetto attivamente tracciato
Lost: oggetto non trovato in un frame, ma non ancora eliminato
Removed: oggetto eliminato definitivamente dal tracciamento
"""
class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0 #Contatore globale di oggetti per assegnare ID univoci.

    track_id = 0 # identificatore univoco dell’oggetto.
    is_activated = False #flag che indica se il track è attivo (associato a una detection).
    state = TrackState.New #stato corrente (tra New, Tracked, Lost, Removed).

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0 # frame in cui è apparso per la prima volta.
    frame_id = 0
    time_since_update = 0 #quanti frame sono passati dall’ultimo aggiornamento.

    # multi-camera
    location = (np.inf, np.inf) #Coordinate spaziali della posizione nel mondo (non usate di solito in single-camera).

    #metodo per sapere in che frame finisce il track
    @property
    def end_frame(self):
        return self.frame_id #Il frame finale di un track è semplicemente il suo frame_id più recente.

    # Metodo statico per generare ID
    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args): #inizializza il track (usato alla prima associazione con una detection).
        raise NotImplementedError

    def predict(self): #predice la posizione del track (es. con Kalman Filter).
        raise NotImplementedError

    def update(self, *args, **kwargs): #aggiorna con nuove coordinate/detection.
        raise NotImplementedError

    def mark_lost(self): #chiamato quando un oggetto non viene trovato per un po'
        self.state = TrackState.Lost
 
    def mark_removed(self): #chiamato quando l’oggetto è definitivamente fuori scena
        self.state = TrackState.Removed
