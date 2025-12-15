import os
import json
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
# Giả định cấu trúc thư mục của bạn vẫn giữ nguyên
from ecapa.ECAPAModel import ECAPAModel

# ===== CM imports =====
from aasist.models.AASIST import Model as AASISTModel


# ======================
#  Utils: EER & Stats
# ======================
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    labels: 1 = genuine target bonafide, 0 = others (nontarget + spoof)
    returns: (eer, threshold_at_eer)
    """
    # Sắp xếp điểm số giảm dần
    idx = np.argsort(scores)[::-1]
    scores_s = scores[idx]
    labels_s = labels[idx]

    P = np.sum(labels_s == 1)
    N = np.sum(labels_s == 0)
    if P == 0 or N == 0:
        raise ValueError("Need both positive and negative samples to compute EER")

    tp = 0
    fp = 0
    fn = P
    tn = N

    best_eer = 1.0
    best_thr = float("inf")
    prev_score = None

    # Tìm điểm cắt tối ưu (nơi FAR ~ FRR)
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


def compute_z_norm_stats(scores: np.ndarray):
    """Tính mean và std để chuẩn hóa"""
    mu = np.mean(scores)
    std = np.std(scores)
    return mu, std


def apply_z_norm(scores: np.ndarray, mu: float, std: float):
    """Áp dụng chuẩn hóa (x - mu) / std"""
    return (scores - mu) / (std + 1e-9)


# ======================
#  ASV (ECAPA)
# ======================
def load_ecapa(model_path: str, device: torch.device):
    # Lưu ý: Đảm bảo tham số C=1024, n_class=5994 khớp với model pretrain của bạn
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
            data = data[:, 0]  # Lấy kênh đầu tiên nếu là stereo
        audio_segments.append(data)

    if len(audio_segments) == 0:
        raise ValueError("Empty file_paths")

    full_audio = np.concatenate(audio_segments)
    data_1 = torch.FloatTensor(np.stack([full_audio], axis=0))

    # Cấu hình cắt/padding audio (cần khớp với lúc train ECAPA)
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
    # Cosine Similarity
    score_1 = torch.mean(torch.matmul(emb_1_A, emb_1_B.T))
    score_2 = torch.mean(torch.matmul(emb_2_A, emb_2_B.T))
    score = (score_1 + score_2) / 2.0
    return float(score.item())


# ======================
#  CM (AASIST)
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
    """
    CẢI TIẾN: Trả về raw logit thay vì clip/scale thủ công.
    Để Z-score normalization ở hàm main xử lý phân phối.
    """
    X, _ = sf.read(audio_path)
    if len(X.shape) > 1:
        X = X[:, 0]
    X_pad = pad_aasist(X, 64600)
    x_inp = torch.Tensor(X_pad).unsqueeze(0).to(device)

    _, output = model(x_inp)
    bonafide_logit = output[:, 1].item()  # Index 1 là lớp bonafide
    return float(bonafide_logit)


# ======================
#  Protocol Parsing
# ======================
def read_enroll_list(enroll_trn_path: str) -> Dict[str, List[str]]:
    enr = {}
    if not os.path.exists(enroll_trn_path):
        print(f"[WARN] Enroll file not found: {enroll_trn_path}")
        return enr

    with open(enroll_trn_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            spk, files = line.split()
            enr[spk] = files.split(",")
    return enr


@dataclass
class Trial:
    spk: str
    test_id: str
    attack_or_bonafide: str
    key: str


def read_asv_trials(trl_path: str) -> List[Trial]:
    trials = []
    if not os.path.exists(trl_path):
        print(f"[WARN] Trial file not found: {trl_path}")
        return trials

    with open(trl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            spk, test_id, atk, key = line.split()
            trials.append(Trial(spk=spk, test_id=test_id, attack_or_bonafide=atk, key=key))
    return trials


def resolve_audio_path(file_id: str, dirs: List[str]) -> str:
    fn = f"{file_id}.flac"
    for d in dirs:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find {fn} in any of: {dirs}")


# ======================
#  Logging
# ======================
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
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = TeeStdout(sys.stdout, f)
    sys.stderr = TeeStdout(sys.stderr, f)

    def restore():
        sys.stdout = old_out
        sys.stderr = old_err
        f.close()

    return log_path, restore


# ======================
#  MAIN
# ======================
def main():
    # ------- CONFIG -------
    LA_ROOT = os.path.join(".", "data", "LA", "LA")

    # Audio dirs
    DEV_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_dev", "flac")
    PA_DEV_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_PA_dev", "flac")
    AUDIO_DIRS = [DEV_AUDIO_DIR, PA_DEV_AUDIO_DIR]

    # Model paths
    ECAPA_WEIGHT = "./ecapa/exps/pretrain.model"
    AASIST_CONF = "./aasist/config/AASIST.conf"
    AASIST_WEIGHT = "./aasist/models/weights/AASIST.pth"

    w_step = 0.05  # Quét mịn hơn: 0.01

    # Setup Log
    log_path, restore_io = setup_run_log("./log", prefix="optimize_w_znorm")
    print(f"[LOG] Writing to: {log_path}")

    # Subsample: NÊN TẮT để lấy tham số chính xác nhất
    # Nếu muốn debug nhanh thì bật lên, nhưng khi chốt tham số hãy để False
    USE_SUBSAMPLE = False

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # 1. LOAD PROTOCOLS (Male + Female)
        # CẢI TIẾN: Load cả nam và nữ để tối ưu tổng quát
        enroll_map = {}
        trials = []

        prot_dir = os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols")
        genders = ['female', 'male']

        print("Loading protocols...")
        for g in genders:
            trn_file = os.path.join(prot_dir, f"ASVspoof2019.LA.asv.dev.{g}.trn.txt")
            trl_file = os.path.join(prot_dir, f"ASVspoof2019.LA.asv.dev.{g}.trl.txt")

            # Merge enroll map
            cur_enr = read_enroll_list(trn_file)
            enroll_map.update(cur_enr)

            # Merge trials
            cur_trials = read_asv_trials(trl_file)
            trials.extend(cur_trials)
            print(f" -> Loaded {g}: {len(cur_enr)} enroll spks, {len(cur_trials)} trials.")

        print(f"TOTAL: {len(enroll_map)} enroll speakers, {len(trials)} trials.")

        if USE_SUBSAMPLE:
            # Code lấy mẫu cũ, chỉ dùng khi debug nhanh
            trials = trials[:2000]
            print(f"[DEBUG] Subsampling enabled. Using first {len(trials)} trials.")

        # 2. LOAD MODELS
        ecapa = load_ecapa(ECAPA_WEIGHT, device)
        aasist = load_aasist(AASIST_CONF, AASIST_WEIGHT, device)

        # 3. COMPUTE SCORES
        enroll_emb_cache = {}
        test_emb_cache = {}
        cm_cache = {}

        s_asv_list = []
        s_cm_list = []
        y_list = []

        print("Computing scores...")
        for i, tr in enumerate(trials):
            # Label: 1 if target bonafide, 0 otherwise (spoof or nontarget)
            is_pos = (tr.key == "target" and tr.attack_or_bonafide == "bonafide")
            y_list.append(1 if is_pos else 0)

            # ASV Processing
            enr_ids = enroll_map.get(tr.spk)
            if enr_ids is None:
                # Fallback: Đôi khi file trl có speaker không có trong trn (hiếm gặp ở LA dev chuẩn)
                # Nếu gặp lỗi này, kiểm tra lại file paths
                print(f"Skip trial {i}: Speaker {tr.spk} not found in enroll list")
                s_asv_list.append(-99.0)  # Dummy
                s_cm_list.append(-99.0)
                continue

            # Path Resolution
            try:
                enr_paths = [resolve_audio_path(fid, AUDIO_DIRS) for fid in enr_ids]
                tst_path = resolve_audio_path(tr.test_id, AUDIO_DIRS)
            except FileNotFoundError as e:
                print(f"Skip trial {i}: {e}")
                s_asv_list.append(-99.0)
                s_cm_list.append(-99.0)
                continue

            # Get ASV Score
            if tr.spk not in enroll_emb_cache:
                enroll_emb_cache[tr.spk] = ecapa_embed_pair(ecapa, enr_paths, device)

            if tr.test_id not in test_emb_cache:
                test_emb_cache[tr.test_id] = ecapa_embed_pair(ecapa, [tst_path], device)

            s_asv = asv_score_from_emb(enroll_emb_cache[tr.spk], test_emb_cache[tr.test_id])

            # Get CM Score
            if tr.test_id not in cm_cache:
                cm_cache[tr.test_id] = cm_score_aasist(aasist, tst_path, device)
            s_cm = cm_cache[tr.test_id]

            s_asv_list.append(s_asv)
            s_cm_list.append(s_cm)

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(trials)} trials...")

        # Convert to numpy
        s_asv_arr = np.asarray(s_asv_list, dtype=np.float64)
        s_cm_arr = np.asarray(s_cm_list, dtype=np.float64)
        y_arr = np.asarray(y_list, dtype=np.int32)

        # 4. Z-SCORE NORMALIZATION (CẢI TIẾN QUAN TRỌNG)
        # Tính thống kê trên toàn bộ tập Dev
        asv_mu, asv_std = compute_z_norm_stats(s_asv_arr)
        cm_mu, cm_std = compute_z_norm_stats(s_cm_arr)

        print("\n" + "=" * 40)
        print("NORMALIZATION STATISTICS (SAVE THESE!)")
        print("=" * 40)
        print(f"ASV Mean: {asv_mu:.6f} | ASV Std: {asv_std:.6f}")
        print(f"CM  Mean: {cm_mu:.6f} | CM  Std: {cm_std:.6f}")
        print("Use these values to normalize Evaluation scores.")
        print("=" * 40 + "\n")

        # Áp dụng chuẩn hóa
        s_asv_norm = apply_z_norm(s_asv_arr, asv_mu, asv_std)
        s_cm_norm = apply_z_norm(s_cm_arr, cm_mu, cm_std)

        # 5. SWEEP WEIGHT w
        print("Sweeping w for fused score: S = w * CM_norm + (1-w) * ASV_norm ...")

        best = {"w": None, "eer": 1.0, "thr": None}
        ws = np.arange(0.0, 1.0 + 1e-9, w_step)

        for w in ws:
            # Linear Fusion trên điểm đã chuẩn hóa
            s_final = w * s_cm_norm + (1.0 - w) * s_asv_norm
            eer, thr = compute_eer(s_final, y_arr)

            if eer < best["eer"]:
                best = {"w": float(w), "eer": float(eer), "thr": float(thr)}

            # In bớt log để đỡ rối, chỉ in mỗi 10 bước hoặc khi EER rất thấp
            if int(w * 100) % 10 == 0:
                print(f"w={w:.2f} -> EER={eer * 100:.3f}%")

        print("\n========== RESULT ==========")
        print(f"Best w (Z-normed) : {best['w']:.4f}")
        print(f"Best EER (Dev)    : {best['eer'] * 100:.4f}%")
        print(f"Best Threshold    : {best['thr']:.4f}")
        print("============================\n")

    except Exception:
        print("\n[ERROR] Exception occurred:\n")
        traceback.print_exc()
    finally:
        restore_io()


if __name__ == "__main__":
    main()