import os
import json
# import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
from collections import defaultdict
import sys
import traceback
from datetime import datetime
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

# ===== ASV imports =====
from ecapa.ECAPAModel import ECAPAModel

# ===== CM imports =====
from aasist.models.AASIST import Model as AASISTModel


# ======================
#  Utils: EER
# ======================
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    labels: 1 = genuine target bonafide, 0 = others (nontarget + spoof)
    returns: (eer, threshold_at_eer)  [threshold only for computing EER, NOT for final system usage]
    """
    # sort by score desc
    idx = np.argsort(scores)[::-1]
    scores_s = scores[idx]
    labels_s = labels[idx]

    P = np.sum(labels_s == 1)
    N = np.sum(labels_s == 0)
    if P == 0 or N == 0:
        raise ValueError("Need both positive and negative samples to compute EER")

    # Sweep threshold at each unique score
    # accept if score >= thr
    tp = 0
    fp = 0
    fn = P
    tn = N

    best_eer = 1.0
    best_thr = float("inf")

    # Initial: thr > max(score) => accept none => FAR=0, FRR=1
    # Iterate thresholds from high->low by adding one sample into "accepted" each step
    prev_score = None
    for s, y in zip(scores_s, labels_s):
        if prev_score is None or s != prev_score:
            # evaluate at threshold = prev_score (before moving to new score)
            far = fp / N
            frr = fn / P
            eer = (far + frr) / 2.0  # common approximation; also can use |far-frr| min
            # better: choose point where |FAR-FRR| minimal
            # We'll track via min |far-frr| and compute eer = (far+frr)/2 at that point.
        prev_score = s

        # move this sample to accepted set (thr <= s)
        if y == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

    # precise EER: find threshold where |FAR-FRR| is minimal
    tp = 0
    fp = 0
    fn = P
    tn = N
    prev_score = None
    best_gap = 1e9
    for s, y in zip(scores_s, labels_s):
        if prev_score is None or s != prev_score:
            far = fp / N
            frr = fn / P
            gap = abs(far - frr)
            if gap < best_gap:
                best_gap = gap
                best_eer = (far + frr) / 2.0
                best_thr = s
        prev_score = s

        if y == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

    return float(best_eer), float(best_thr)


# ======================
#  ASV (ECAPA) as in your pipeline
# ======================
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

    if len(audio_segments) == 0:
        raise ValueError("Empty file_paths")

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

    emb_1 = F.normalize(model.speaker_encoder.forward(data_1, aug=False), p=2, dim=1)  # (1,D)
    emb_2 = F.normalize(model.speaker_encoder.forward(data_2, aug=False), p=2, dim=1)  # (5,D)
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


# ======================
#  CM (AASIST) as in your pipeline
# ======================
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

    # scale logit -> [-1, 1] (same as your pipeline)
    aasist_min, aasist_max = -20.0, 20.0
    val_clipped = max(aasist_min, min(aasist_max, bonafide_logit))
    bonafide_score = 2 * (val_clipped - aasist_min) / (aasist_max - aasist_min) - 1
    return float(bonafide_score)


# ======================
#  Protocol parsing (ASV dev)
# ======================
def read_enroll_list(enroll_trn_path: str) -> Dict[str, List[str]]:
    """
    Format: speaker_id  file1,file2,file3,...
    """
    enr = {}
    with open(enroll_trn_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spk, files = line.split()
            enr[spk] = files.split(",")
    return enr


@dataclass
class Trial:
    spk: str
    test_id: str
    attack_or_bonafide: str
    key: str  # target | nontarget | spoof


def read_asv_trials(trl_path: str) -> List[Trial]:
    """
    Format: claimed_spk test_file_id attack_id_or_bonafide key
    """
    trials = []
    with open(trl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spk, test_id, atk, key = line.split()
            trials.append(Trial(spk=spk, test_id=test_id, attack_or_bonafide=atk, key=key))
    return trials


def subsample_trials_stratified(trials: List[Trial], ratio: float = 0.1, seed: int = 42) -> List[Trial]:
    """
    Giữ tỷ lệ theo key: target / nontarget / spoof để EER ổn định hơn.
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for tr in trials:
        buckets[tr.key].append(tr)

    out = []
    for k, lst in buckets.items():
        n = max(1, int(len(lst) * ratio))
        rng.shuffle(lst)
        out.extend(lst[:n])

    rng.shuffle(out)
    return out


def resolve_audio_path(file_id: str, dirs: List[str]) -> str:
    # file_id looks like LA_D_1234567
    fn = f"{file_id}.flac"
    for d in dirs:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {fn} in any of: {dirs}")


class TeeStdout:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

def setup_run_log(log_dir: str = "./log", prefix: str = "optimize_w"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{prefix}_{ts}.log")
    f = open(log_path, "w", encoding="utf-8")

    # Tee stdout + stderr
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = TeeStdout(sys.stdout, f)
    sys.stderr = TeeStdout(sys.stderr, f)

    def restore():
        sys.stdout = old_out
        sys.stderr = old_err
        f.close()

    return log_path, restore


def merge_enroll_maps(*maps: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for m in maps:
        for spk, files in m.items():
            if spk in out:
                raise ValueError(f"Duplicate speaker in enrollment maps: {spk}")
            out[spk] = files
    return out


# ======================
#  Main: compute fused scores once, then sweep w fast
# ======================
def main():
    # ------- EDIT THESE PATHS -------
    # Root folder of ASVspoof2019 LA (the folder that contains ASVspoof2019_LA_dev, ASVspoof2019_LA_asv_protocols, ...)
    LA_ROOT = os.path.join(".", "data", "LA", "LA")  # <-- chỉnh theo máy bạn

    ASV_TRL = os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols", "ASVspoof2019.LA.asv.dev.gi.trl.txt")

    ASV_TRN_M = os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols", "ASVspoof2019.LA.asv.dev.male.trn.txt")
    ASV_TRN_F = os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols", "ASVspoof2019.LA.asv.dev.female.trn.txt")

    # audio directories to search (dev audio always here)
    DEV_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_dev", "flac")
    # enrollment audio may be in PA_dev for some setups; include both to be safe (README mentions PA_dev for enroll) :contentReference[oaicite:2]{index=2}
    PA_DEV_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_PA_dev", "flac")

    AUDIO_DIRS = [DEV_AUDIO_DIR, PA_DEV_AUDIO_DIR]  # resolve by existence

    # Model paths
    ECAPA_WEIGHT = "./ecapa/exps/pretrain.model"
    AASIST_CONF = "./aasist/config/AASIST.conf"
    AASIST_WEIGHT = "./aasist/models/weights/AASIST.pth"

    # Sweep config
    w_step = 0.05  # 0.01 => 101 points

    # ====== setup log ======
    log_path, restore_io = setup_run_log("./log", prefix="optimize_w")
    print(f"[LOG] Writing to: {log_path}")

    # sub sample
    USE_SUBSAMPLE = False
    SUBSAMPLE_RATIO = 0.1
    SUBSAMPLE_SEED = 123

    # Ghi lại toàn bộ cấu hình run
    run_cfg = {
        "LA_ROOT": LA_ROOT,
        "ASV_TRL": ASV_TRL,
        "ASV_TRN_M": ASV_TRN_M,
        "ASV_TRN_F": ASV_TRN_F,
        "AUDIO_DIRS": AUDIO_DIRS,
        "ECAPA_WEIGHT": ECAPA_WEIGHT,
        "AASIST_CONF": AASIST_CONF,
        "AASIST_WEIGHT": AASIST_WEIGHT,
        "w_step": w_step,
        "USE_SUBSAMPLE": USE_SUBSAMPLE,
        "subsample_ratio": SUBSAMPLE_RATIO,
        "subsample_seed": SUBSAMPLE_SEED,
    }
    print("[RUN CONFIG]\n" + pformat(run_cfg) + "\n")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Load protocols (GI: trials gộp cả nam + nữ, enrollment phải gộp m + f)
        enroll_map_m = read_enroll_list(ASV_TRN_M)
        enroll_map_f = read_enroll_list(ASV_TRN_F)
        enroll_map = merge_enroll_maps(enroll_map_m, enroll_map_f)

        trials = read_asv_trials(ASV_TRL)

        missing_spk = [t.spk for t in trials if t.spk not in enroll_map]
        if missing_spk:
            uniq = sorted(set(missing_spk))
            raise KeyError(f"Missing enrollment for {len(uniq)} speakers. Examples: {uniq[:10]}")

        print(f"Enroll speakers (male): {len(enroll_map_m)}")
        print(f"Enroll speakers (female): {len(enroll_map_f)}")
        print(f"Enroll speakers (merged): {len(enroll_map)}")
        print(f"Trials (gi): {len(trials)}")

        if USE_SUBSAMPLE:
            trials = subsample_trials_stratified(trials, ratio=SUBSAMPLE_RATIO, seed=SUBSAMPLE_SEED)
            print(f"Subsampled trials: {len(trials)} (ratio={SUBSAMPLE_RATIO}, seed={SUBSAMPLE_SEED})")

        # Load models
        ecapa = load_ecapa(ECAPA_WEIGHT, device)
        aasist = load_aasist(AASIST_CONF, AASIST_WEIGHT, device)

        # Caches
        enroll_emb_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        test_emb_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        cm_cache: Dict[str, float] = {}

        s_asv_list = []
        s_cm_list = []
        y_list = []

        # Build per-trial ASV/CM scores (only once)
        for i, tr in enumerate(trials):
            # Label: positive only if target & bonafide
            # trial file provides: key=target/nontarget/spoof and attack_id/bonafide :contentReference[oaicite:3]{index=3}
            is_pos = (tr.key == "target" and tr.attack_or_bonafide == "bonafide")
            y_list.append(1 if is_pos else 0)

            # Resolve enrollment files for claimed speaker
            enr_ids = enroll_map.get(tr.spk)
            if enr_ids is None:
                raise KeyError(f"Missing enrollment list for speaker {tr.spk}")

            # Resolve paths
            enr_paths = [resolve_audio_path(fid, AUDIO_DIRS) for fid in enr_ids]
            tst_path = resolve_audio_path(tr.test_id, AUDIO_DIRS)

            # Enrollment embedding cache per speaker
            if tr.spk not in enroll_emb_cache:
                enroll_emb_cache[tr.spk] = ecapa_embed_pair(ecapa, enr_paths, device)

            # Test embedding cache per test file
            if tr.test_id not in test_emb_cache:
                test_emb_cache[tr.test_id] = ecapa_embed_pair(ecapa, [tst_path], device)

            s_asv = asv_score_from_emb(enroll_emb_cache[tr.spk], test_emb_cache[tr.test_id])

            # CM cache per test file
            if tr.test_id not in cm_cache:
                cm_cache[tr.test_id] = cm_score_aasist(aasist, tst_path, device)
            s_cm = cm_cache[tr.test_id]

            s_asv_list.append(s_asv)
            s_cm_list.append(s_cm)

            if (i + 1) % 2000 == 0:
                print(f"Processed {i+1}/{len(trials)} trials...")

        s_asv_arr = np.asarray(s_asv_list, dtype=np.float64)
        s_cm_arr = np.asarray(s_cm_list, dtype=np.float64)
        y_arr = np.asarray(y_list, dtype=np.int32)

        # Sweep w
        best = {"w": None, "eer": 1.0, "thr": None}
        ws = np.arange(0.0, 1.0 + 1e-9, w_step)

        for w in ws:
            s_final = w * s_cm_arr + (1.0 - w) * s_asv_arr
            eer, thr = compute_eer(s_final, y_arr)
            if eer < best["eer"]:
                best = {"w": float(w), "eer": float(eer), "thr": float(thr)}
            print(f"w={w:.2f}  EER={eer*100:.3f}%")

        print("\n========== BEST w (dev, by min EER) ==========")
        print(f"best_w = {best['w']:.4f}")
        print(f"best_EER = {best['eer']*100:.3f}%")
        print("=============================================\n")

    except Exception:
        print("\n[ERROR] Exception occurred:\n")
        traceback.print_exc()
        raise
    finally:
        # đảm bảo đóng file log và trả stdout về bình thường
        restore_io()


if __name__ == "__main__":
    main()
