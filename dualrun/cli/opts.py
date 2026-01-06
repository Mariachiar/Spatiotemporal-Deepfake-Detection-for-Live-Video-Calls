import argparse
from pathlib import Path
from datetime import datetime


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--index", default=None)
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--attn_entropy", type=float, default=0.0, help="Peso regolarizzazione di entropia sui pesi di attenzione (0=off).")
    parser.add_argument("--attn_agree",   type=float, default=0.0, help="Peso KL-simmetrica tra mparserpe di attenzione AU e LMK (0=off).")


    parser.add_argument("--lam-align",   type=float, default=0.0,
                    help="Peso della loss di allineamento (0=off)")
    parser.add_argument("--lam-uniform", type=float, default=0.0,
                        help="Peso della loss di uniformità (0=off)")
    parser.add_argument("--uniform-t",   type=float, default=2.0,
                        help="Temperatura t per la uniformity loss")
    
    parser.add_argument('--aux_pred_w', type=float, default=0.0,
        help='Peso loss LMK->AU (MSE) sui soli reali')
    parser.add_argument('--aux_con_w', type=float, default=0.0,
        help='Peso loss contrastiva temporale AU↔LMK')
    

    parser.add_argument("--freeze-lmk", type=int, default=0)
    parser.add_argument("--freeze-au",  type=int, default=0)


    # dataset
    parser.add_argument("--T", type=int, default=8)
    #parser.add_argument("--zscore", default="none", choices=["none", "clip", "global"])
    parser.add_argument("--random-crop", action="store_true")
    parser.add_argument("--no-mmparser", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--target-fpr", type=float, default=None,
                help="Vincola la scelta soglia con FPR <= valore (es. 0.10)")
    


    # AUG: passati al dataset
    parser.add_argument("--aug-noise-au", type=float, default=0.0, help="Std del jitter gaussiano sulle AU (es. 0.01)")
    parser.add_argument("--aug-noise-lmk", type=float, default=0.0, help="Std del jitter gaussiano sui LMK (es. 0.005)")
    parser.add_argument("--aug-tdrop", type=float, default=0.0, help="Frazione di frame validi da azzerare (es. 0.05)")

    # modello
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)

    # train
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--batch-eval", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--scheduler", default="cosine", choices=["cosine","onecycle"],
                help="LR scheduler: cosine (default) o onecycle")
    parser.add_argument("--onecycle_div_factor", type=float, default=25.0,
                    help="Divisore per il LR iniziale: base_lr = max_lr / div_factor")
    parser.add_argument("--onecycle_final_div", type=float, default=1e4,
                    help="Divisore finale: min_lr ~ max_lr / final_div")
    parser.add_argument("--onecycle_pct_start", type=float, default=0.10,
                    help="Frazione di training in warm-up (salita LR)")
    parser.add_argument("--stitch-k", type=int, default=1)
    parser.add_argument("--lmk-add-deltas", action="store_true")

    parser.add_argument('--zscore', default='none', choices=['none','clip','global'],
                   help='Tipo di normalizzazione z-score.')
    parser.add_argument('--zscore-parserply', default='both', choices=['both','au','lmk'],
                  help='A quali modalità parserplicare lo z-score.')
    parser.add_argument('--qual_factorized', action='store_true')
    parser.add_argument('--dirty_p', type=float, default=0.9)
    parser.add_argument('--adv_quality_lambda', type=float, default=0.0)
    parser.add_argument('--use_contrastive', action='store_true')
    parser.add_argument('--contrastive_tau', type=float, default=0.1)
    parser.add_argument('--clean_fake_p', type=float, default=0.2)
    parser.add_argument('--clean_real_p', type=float, default=0.2)
    parser.add_argument('--protect_real_for_consistency', action='store_true')

    parser.add_argument("--slerp-feature-augmentation", action="store_true",
                    help="Attiva SLERP sugli embedding durante il training.")
    parser.add_argument("--slerp-feature-augmentation-range", type=float, nargs=2, default=(0.0, 1.0),
                    metavar=("T0","T1"),
                    help="Intervallo [t0,t1] per l'interpolazione sferica (default 0.0 1.0).")

    
    # LMK degradations (feature-space)
    parser.add_argument('--lmk_noise_std', type=float, default=0.01)     # rumore additivo in unità normalizzate
    parser.add_argument('--lmk_affine_deg', type=float, default=2.0)     # rotazione max in gradi
    parser.add_argument('--lmk_dropout_p', type=float, default=0.05)     # dropout di punti
    parser.add_argument('--lmk_temporal_alpha', type=float, default=0.0) # smoothing EMA [0..1], 0=off
    
    # AU degradations (feature-space)
    parser.add_argument('--au_noise_std', type=float, default=0.05)
    parser.add_argument('--au_dropout_p', type=float, default=0.05)
    parser.add_argument('--au_temporal_alpha', type=float, default=0.0)
    
    # Test-time
    parser.add_argument('--test_feature_smooth_alpha', type=float, default=0.0)
    parser.add_argument('--tta_quality_n', type=int, default=0)          # repliche con diversi seed
    parser.add_argument('--video_thresh_file', type=str, default='')
    parser.add_argument('--qual_lambda', type=float, default=0.0)
    parser.add_argument('--qual_ce_weight', type=float, default=1.0)




    # Bilanciamento *sempre attivo*
    parser.add_argument("--epoch-samples", type=int, default=20000,
                    help="Totale esempi per epoca (deve essere pari). Meta' real, meta' fake.")
    parser.add_argument("--shuffle-every-epoch", action="store_true")

    # LOO opzionale
    parser.add_argument("--heldout_tech", default=None)

    # Loss/extra
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=1.0)  # stabile per i tuoi dati
    parser.add_argument("--focal_alpha", type=float, default=0.45)
    parser.add_argument("--pos-weight", type=float, default=None,
                    help="Peso della classe positiva per BCEWithLogitsLoss (ignorato se --focal)")

    # Domain-Adversarial Training (DAT)
    parser.add_argument("--dat", action="store_true", help="Attiva Domain-Adversarial Training")
    parser.add_argument("--dat-lambda", type=float, default=0.0, help="Coefficiente iniziale λ per GRL/DAT")
    parser.add_argument("--dat-schedule", default="const", choices=["const", "linear"],
                    help="Schedulazione di λ: costante o lineare (da 0 a dat-lambda)")

    # Early stopping & metriche
    parser.add_argument("--es-metric", default="auc", choices=["youden", "balacc", "f1", "acc", "auc"],
                    help="Metrica su VAL usata per early stopping e scelta soglia")
    parser.add_argument("--es-warmup", type=int, default=0)

    parser.add_argument("--out", default=None)

    # ---- NUOVI FLAG ----
    parser.add_argument("--init", default=None, help="Checkpoint .pt da cui inizializzare i pesi del modello (usa chiave 'model').")
    parser.add_argument("--freeze-encoders", type=int, default=0, help="N epoche iniziali in cui congelare i due encoder (AU/LMK).")


    parser.add_argument("--train-agg", type=str, default="none",
        choices=["none","track_median","track_mean","video_or_median","video_or_mean"])

    parser.add_argument("--eval-agg", type=str, default="none",
        choices=["none","track_median","track_mean","video_or_median","video_or_mean"])
    parser.add_argument("--boost-tech", nargs="+", default=None,
        help='Lista tecnica:fattore, es. neuraltextures:3 faceswparser:1.5')
    parser.add_argument("--min-quota-fake", type=int, default=0,
        help="Minimo assoluto di fake per-tech per epoca")
    
    parser.add_argument("--pool-tau", type=float, default=1.0,
        help="Temperatura per l’AttentionPooling")
    
    # --- AltFreezing ---
    parser.add_argument('--altfreeze-enabled', type=int, default=0, choices=[0,1],
        help='1=attivo, 0=disattivo')
    parser.add_argument('--altfreeze-warmup', type=int, default=2,
        help='epoche iniziali joint')
    parser.add_argument('--altfreeze-period', type=int, default=2,
        help='durata di ciascuna fase A/B')
    parser.add_argument('--altfreeze-joint-tail', type=int, default=2,
        help='epoche finali joint')
    parser.add_argument('--altfreeze-start', type=int, default=1,
        help='epoca di inizio alternanza (1-indexed)')


    parser.add_argument("--regen-from-videos", action="store_true",
        help="Usa i VIDEO raw per rigenerare AU/LMK on-the-fly nel TRAIN")
    parser.add_argument("--train-videos-list", default=None,
        help="TXT/JSON con lista di path video per il TRAIN FT")
    parser.add_argument("--regen-jpeg",  type=int,   nargs=2, default=[3,25])
    parser.add_argument("--regen-scale", type=float, nargs=2, default=[0.3,0.8])
    parser.add_argument("--regen-offcenter", type=float, default=0.06)
    parser.add_argument("--regen-mblur", type=int, nargs=2, default=[0,9])  # kernel dispari max

    parser.add_argument("--val-from-videos", action="store_true")
    parser.add_argument("--val-videos-list", type=str, default=None)  # txt/json con path video
    parser.add_argument("--val-data", type=str, default=None)   

    parser.add_argument("--test-from-videos", action="store_true")
    parser.add_argument("--test-videos-list", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)   

    parser.add_argument("--consistency-w", type=float, default=0.0,
               help="peso MSE tra embedding clean e degraded")

    args = parser.parse_args()
    return args
