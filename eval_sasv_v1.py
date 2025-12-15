import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
# from collections import defaultdict
import logging
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

# ===== ASV imports =====
from ecapa.ECAPAModel import ECAPAModel

# ===== CM imports =====
from aasist.models.AASIST import Model as AASISTModel


# ============================================================
# EER helpers (ASV-style)
# ============================================================
def compute_eer_threshold(scores_tar: np.ndarray, scores_non: np.ndarray) -> Tuple[float, float]:
    """
    EER + threshold for ASV:
      - accept if score >= thr
      - miss = P(score_tar < thr)
      - fa   = P(score_non >= thr)
    """
    scores = np.concatenate([scores_tar, scores_non])
    labels = np.concatenate([np.ones_like(scores_tar), np.zeros_like(scores_non)])

    idx = np.argsort(scores)[::-1]
    s = scores[idx]
    y = labels[idx]

    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        raise ValueError("Need both target and nontarget to compute ASV EER.")

    tp = 0
    fp = 0
    fn = P
    tn = N

    best_gap = 1e9
    best_eer = 1.0
    best_thr = s[0]

    prev = None
    for score, lab in zip(s, y):
        if prev is None or score != prev:
            pmiss = fn / P
            pfa = fp / N
            gap = abs(pmiss - pfa)
            if gap < best_gap:
                best_gap = gap
                best_eer = (pmiss + pfa) / 2.0
                best_thr = score
        prev = score

        if lab == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

    return float(best_eer), float(best_thr)


def compute_eer_binary(scores: np.ndarray, labels01: np.ndarray) -> Tuple[float, float]:
    """
    EER for your SASV fused score on ASV trials:
      labels01: 1 = (target & bonafide), 0 = others (nontarget + spoof)
      accept if score >= thr
    """
    idx = np.argsort(scores)[::-1]
    s = scores[idx]
    y = labels01[idx]

    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        raise ValueError("Need both positive and negative samples to compute EER.")

    tp = 0
    fp = 0
    fn = P
    tn = N

    best_gap = 1e9
    best_eer = 1.0
    best_thr = s[0]

    prev = None
    for score, lab in zip(s, y):
        if prev is None or score != prev:
            far = fp / N
            frr = fn / P
            gap = abs(far - frr)
            if gap < best_gap:
                best_gap = gap
                best_eer = (far + frr) / 2.0
                best_thr = score
        prev = score

        if lab == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

    return float(best_eer), float(best_thr)


# ============================================================
# ASV (ECAPA) — same as your pipeline
# ============================================================
def load_ecapa(model_path: str, device: torch.device):
    model = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=5994, m=0.2, s=30, test_step=1)
    model.load_parameters(model_path)
    model.to(device)
    model.eval()
    return model


def process_ecapa_audio(file_paths: List[str], max_frames=300):
    audio_segments = []
    for path in file_paths:
        data, _ = sf.read(path)
        if len(data.shape) > 1:
            data = data[:, 0]
        audio_segments.append(data)

    full_audio = np.concatenate(audio_segments)
    data_1 = torch.FloatTensor(np.stack([full_audio], axis=0))

    max_audio = max_frames * 160 + 240
    if full_audio.shape[0] <= max_audio:
        shortage = max_audio - full_audio.shape[0]
        audio_padded = np.pad(full_audio, (0, shortage), "wrap")
    else:
        audio_padded = full_audio

    feats = []
    startframe = np.linspace(0, audio_padded.shape[0] - max_audio, num=5)
    for asf in startframe:
        feats.append(audio_padded[int(asf): int(asf) + max_audio])
    data_2 = torch.FloatTensor(np.stack(feats, axis=0))
    return data_1, data_2


@torch.no_grad()
def ecapa_embed_pair(model, file_paths: List[str], device: torch.device):
    data_1, data_2 = process_ecapa_audio(file_paths)
    data_1, data_2 = data_1.to(device), data_2.to(device)

    emb_1 = F.normalize(model.speaker_encoder.forward(data_1, aug=False), p=2, dim=1)
    emb_2 = F.normalize(model.speaker_encoder.forward(data_2, aug=False), p=2, dim=1)
    return emb_1, emb_2


@torch.no_grad()
def asv_score_from_emb(enr_emb: Tuple[torch.Tensor, torch.Tensor],
                       tst_emb: Tuple[torch.Tensor, torch.Tensor]) -> float:
    emb_1_A, emb_2_A = enr_emb
    emb_1_B, emb_2_B = tst_emb

    score_1 = torch.mean(torch.matmul(emb_1_A, emb_1_B.T))
    score_2 = torch.mean(torch.matmul(emb_2_A, emb_2_B.T))
    score = (score_1 + score_2) / 2.0
    return float(score.item())


# ============================================================
# CM (AASIST) — same as your pipeline
# ============================================================
def pad_aasist(x: np.ndarray, max_len=64600) -> np.ndarray:
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def load_aasist(config_path: str, weight_path: str, device: torch.device):
    with open(config_path, "r") as f:
        config = json.load(f)
    model = AASISTModel(config["model_config"]).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def cm_score_aasist(model, audio_path: str, device: torch.device) -> float:
    X, _ = sf.read(audio_path)
    if len(X.shape) > 1:
        X = X[:, 0]
    X_pad = pad_aasist(X, 64600)
    x_inp = torch.Tensor(X_pad).unsqueeze(0).to(device)

    _, output = model(x_inp)
    bonafide_logit = output[:, 1].item()

    # scale logit -> [-1, 1]
    aasist_min, aasist_max = -20.0, 20.0
    val_clipped = max(aasist_min, min(aasist_max, bonafide_logit))
    bonafide_score = 2 * (val_clipped - aasist_min) / (aasist_max - aasist_min) - 1
    return float(bonafide_score)


# ============================================================
# Protocol parsing (ASV / CM eval)
# ============================================================
def read_enroll_list(enroll_trn_path: str) -> Dict[str, List[str]]:
    enr = {}
    with open(enroll_trn_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spk, files = line.split()
            enr[spk] = files.split(",")
    return enr


def merge_enroll_maps(*maps: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for m in maps:
        for spk, files in m.items():
            if spk in out:
                raise ValueError(f"Duplicate speaker in enrollment maps: {spk}")
            out[spk] = files
    return out


@dataclass
class AsvTrial:
    spk: str
    test_id: str
    attack_or_bonafide: str
    key: str  # target | nontarget | spoof


def read_asv_trials(trl_path: str) -> List[AsvTrial]:
    trials = []
    with open(trl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spk, test_id, atk, key = line.split()
            trials.append(AsvTrial(spk=spk, test_id=test_id, attack_or_bonafide=atk, key=key))
    return trials


@dataclass
class CmTrial:
    speaker_id: str
    utt_id: str
    sys_id: str
    key: str  # bonafide | spoof


def read_cm_trials(cm_trl_path: str) -> List[CmTrial]:
    """
    CM eval protocol format (LA):
      SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
    """
    out = []
    with open(cm_trl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # expected 5 cols
            spk, utt, dash, sysid, key = parts
            out.append(CmTrial(speaker_id=spk, utt_id=utt, sys_id=sysid, key=key))
    return out


def resolve_audio_path(file_id: str, dirs: List[str]) -> str:
    fn = f"{file_id}.flac"
    for d in dirs:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {fn} in any of: {dirs}")


# ============================================================
# t-DCF (ASVspoof2019 LA cost model)
# ============================================================
def compute_min_tDCF_norm(cm_scores_bona: np.ndarray,
                          cm_scores_spoof: np.ndarray,
                          P_asv_miss: float,
                          P_asv_fa: float,
                          P_asv_miss_spoof: float,
                          # ASVspoof2019 LA priors/costs
                          pi_tar=0.9405, pi_non=0.0095, pi_spoof=0.05,
                          C_asv_miss=1, C_asv_fa=10,
                          C_cm_miss=1, C_cm_fa=10) -> Tuple[float, float]:
    """
    Return: (min_tDCF_norm, threshold_at_min)
    CM decision: accept bonafide if score > s, reject if score <= s
    """
    # Eq.(1) constants (Evaluation plan Table 2 + Eq.(1))
    C1 = pi_tar * (C_cm_miss - C_asv_miss * P_asv_miss) - pi_non * C_asv_fa * P_asv_fa
    C2 = C_cm_fa * pi_spoof * (1.0 - P_asv_miss_spoof)

    if C1 <= 0 or C2 <= 0:
        raise ValueError(f"Invalid tDCF constants: C1={C1}, C2={C2}. Check ASV error rates / score direction.")

    tDCF_default = min(C1, C2)

    # Sweep CM thresholds over unique scores
    all_scores = np.concatenate([cm_scores_bona, cm_scores_spoof])
    thresholds = np.unique(all_scores)

    n_bona = len(cm_scores_bona)
    n_spoof = len(cm_scores_spoof)
    if n_bona == 0 or n_spoof == 0:
        raise ValueError("Need both bonafide and spoof CM trials to compute tDCF.")

    best = (1e9, thresholds[0])

    for s in thresholds:
        Pcm_miss = np.mean(cm_scores_bona <= s)     # bonafide rejected
        Pcm_fa = np.mean(cm_scores_spoof > s)       # spoof accepted
        tDCF = C1 * Pcm_miss + C2 * Pcm_fa
        tDCF_norm = tDCF / tDCF_default
        if tDCF_norm < best[0]:
            best = (float(tDCF_norm), float(s))

    return best


def setup_logger(log_dir="logs", prefix="eval_sasv"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # tránh log bị lặp khi chạy nhiều lần

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # file
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Log file: {log_path}")
    return logger


# ============================================================
# Main
# ============================================================
def main():
    # Setup logger
    logger = setup_logger(log_dir="logs", prefix="eval_sasv")
    logger.info("===== START SASV EVAL =====")

    # ====== EDIT PATHS (eval) ======
    LA_ROOT = os.path.join(".", "data", "LA", "LA")

    # ASV eval protocols (gi)
    ASV_TRL_EVAL = os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols", "ASVspoof2019.LA.asv.eval.gi.trl.txt")
    ASV_TRN_M_EVAL = os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols", "ASVspoof2019.LA.asv.eval.male.trn.txt")
    ASV_TRN_F_EVAL = os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols", "ASVspoof2019.LA.asv.eval.female.trn.txt")

    # CM eval protocol
    CM_TRL_EVAL = os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")

    # audio dirs
    LA_EVAL_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_eval", "flac")
    PA_EVAL_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_PA_eval", "flac")  # enrollment sometimes here
    AUDIO_DIRS = [LA_EVAL_AUDIO_DIR, PA_EVAL_AUDIO_DIR]

    # model paths
    ECAPA_WEIGHT = "./ecapa/exps/pretrain.model"
    AASIST_CONF = "./aasist/config/AASIST.conf"
    AASIST_WEIGHT = "./aasist/models/weights/AASIST.pth"

    # w from your dev optimization
    best_w = 0.50  # from optimize_w_20251214_151843.log

    logger.info("===== CONFIG =====")
    logger.info(f"ECAPA_WEIGHT  : {ECAPA_WEIGHT}")
    logger.info(f"AASIST_WEIGHT : {AASIST_WEIGHT}")
    logger.info(f"best_w        : {best_w:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load models
    ecapa = load_ecapa(ECAPA_WEIGHT, device)
    aasist = load_aasist(AASIST_CONF, AASIST_WEIGHT, device)

    # ========================================================
    # Part A) EER of SASV fused score on ASV eval (gi) trials
    # ========================================================
    enroll_m = read_enroll_list(ASV_TRN_M_EVAL)
    enroll_f = read_enroll_list(ASV_TRN_F_EVAL)
    enroll_map = merge_enroll_maps(enroll_m, enroll_f)

    asv_trials = read_asv_trials(ASV_TRL_EVAL)
    logger.info(f"ASV eval trials (gi): {len(asv_trials)}")

    enroll_emb_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    test_emb_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    cm_cache: Dict[str, float] = {}

    fused_scores = []
    fused_labels01 = []

    # Also collect ASV scores split for tDCF later
    asv_scores_target = []
    asv_scores_nontarget = []
    asv_scores_spoof = []  # all spoof (any Axx)

    for i, tr in enumerate(asv_trials):
        # label for fused EER: positive iff (target & bonafide)
        is_pos = (tr.key == "target" and tr.attack_or_bonafide == "bonafide")
        fused_labels01.append(1 if is_pos else 0)

        enr_ids = enroll_map[tr.spk]
        enr_paths = [resolve_audio_path(fid, AUDIO_DIRS) for fid in enr_ids]
        tst_path = resolve_audio_path(tr.test_id, AUDIO_DIRS)

        if tr.spk not in enroll_emb_cache:
            enroll_emb_cache[tr.spk] = ecapa_embed_pair(ecapa, enr_paths, device)
        if tr.test_id not in test_emb_cache:
            test_emb_cache[tr.test_id] = ecapa_embed_pair(ecapa, [tst_path], device)

        s_asv = asv_score_from_emb(enroll_emb_cache[tr.spk], test_emb_cache[tr.test_id])

        if tr.test_id not in cm_cache:
            cm_cache[tr.test_id] = cm_score_aasist(aasist, tst_path, device)
        s_cm = cm_cache[tr.test_id]

        s_fused = best_w * s_cm + (1.0 - best_w) * s_asv
        fused_scores.append(s_fused)

        # Split ASV scores for tDCF ASV operating point (use only bonafide target/nontarget)
        if tr.attack_or_bonafide == "bonafide":
            if tr.key == "target":
                asv_scores_target.append(s_asv)
            elif tr.key == "nontarget":
                asv_scores_nontarget.append(s_asv)
        else:
            # spoof trials (any attack id)
            if tr.key == "spoof":
                asv_scores_spoof.append(s_asv)

        if (i + 1) % 2000 == 0:
            logger.info(f"[ASV] processed {i+1}/{len(asv_trials)}")

    fused_scores = np.asarray(fused_scores, dtype=np.float64)
    fused_labels01 = np.asarray(fused_labels01, dtype=np.int32)

    eer_fused, thr_fused = compute_eer_binary(fused_scores, fused_labels01)
    logger.info("\n===== SASV (fused) on ASV eval =====")
    logger.info(f"best_w = {best_w:.4f}")
    logger.info(f"EER(fused) = {eer_fused*100:.3f}%  (thr@eer={thr_fused:.6f})")

    # ========================================================
    # Part B) min t-DCF on eval (CM=AASIST) with your ASV fixed
    # ========================================================
    asv_scores_target = np.asarray(asv_scores_target, dtype=np.float64)
    asv_scores_nontarget = np.asarray(asv_scores_nontarget, dtype=np.float64)
    asv_scores_spoof = np.asarray(asv_scores_spoof, dtype=np.float64)

    asv_eer, asv_thr = compute_eer_threshold(asv_scores_target, asv_scores_nontarget)

    P_asv_miss = float(np.mean(asv_scores_target < asv_thr))
    P_asv_fa = float(np.mean(asv_scores_nontarget >= asv_thr))
    P_asv_miss_spoof = float(np.mean(asv_scores_spoof < asv_thr)) if len(asv_scores_spoof) > 0 else 0.0

    logger.info("\n===== ASV operating point (for t-DCF) =====")
    logger.info(f"ASV EER (target vs nontarget, bonafide) = {asv_eer*100:.3f}%")
    logger.info(f"ASV thr@eer = {asv_thr:.6f}")
    logger.info(f"P_asv_miss = {P_asv_miss:.6f}")
    logger.info(f"P_asv_fa   = {P_asv_fa:.6f}")
    logger.info(f"P_asv_miss_spoof = {P_asv_miss_spoof:.6f}")

    # Compute CM scores on CM eval protocol (unique utterances)
    cm_trials = read_cm_trials(CM_TRL_EVAL)
    logger.info(f"\nCM eval trials (unique files): {len(cm_trials)}")

    cm_scores_bona = []
    cm_scores_spoof_list = []

    # reuse cm_cache if already computed (by test_id in ASV loop)
    for i, tr in enumerate(cm_trials):
        utt_path = resolve_audio_path(tr.utt_id, [LA_EVAL_AUDIO_DIR])  # CM eval audio is in LA_eval
        if tr.utt_id not in cm_cache:
            cm_cache[tr.utt_id] = cm_score_aasist(aasist, utt_path, device)
        s_cm = cm_cache[tr.utt_id]

        if tr.key == "bonafide":
            cm_scores_bona.append(s_cm)
        else:
            cm_scores_spoof_list.append(s_cm)

        if (i + 1) % 5000 == 0:
            logger.info(f"[CM] processed {i+1}/{len(cm_trials)}")

    cm_scores_bona = np.asarray(cm_scores_bona, dtype=np.float64)
    cm_scores_spoof_list = np.asarray(cm_scores_spoof_list, dtype=np.float64)

    min_tdcf, thr_tdcf = compute_min_tDCF_norm(
        cm_scores_bona=cm_scores_bona,
        cm_scores_spoof=cm_scores_spoof_list,
        P_asv_miss=P_asv_miss,
        P_asv_fa=P_asv_fa,
        P_asv_miss_spoof=P_asv_miss_spoof,
    )

    logger.info("\n===== min normalized t-DCF (ASVspoof2019 LA) =====")
    logger.info(f"min t-DCF_norm = {min_tdcf:.6f}  (thr@min={thr_tdcf:.6f})")


if __name__ == "__main__":
    main()
