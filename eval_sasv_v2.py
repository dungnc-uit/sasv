import os
import json
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

# ===== IMPORTS (Giữ nguyên) =====
from ecapa.ECAPAModel import ECAPAModel
from aasist.models.AASIST import Model as AASISTModel

# ==========================================
#  1. CẤU HÌNH CÁC THAM SỐ TỐI ƯU (TỪ DEV)
# ==========================================
# Các giá trị này lấy từ kết quả chạy optimize_w_dev.py
ASV_MEAN = 0.363799
ASV_STD = 0.207763

CM_MEAN = -3.927683
CM_STD = 5.386927

BEST_W = 0.6000


# ==========================================
#  UTILS
# ==========================================
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


def setup_run_log(log_dir: str = "./log", prefix: str = "eval_sasv"):
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


def apply_norm(score, mu, std):
    return (score - mu) / (std + 1e-9)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tính EER (Labels: 1=Target Bonafide, 0=Spoof/Nontarget)"""
    idx = np.argsort(scores)[::-1]
    scores_s = scores[idx]
    labels_s = labels[idx]

    P = np.sum(labels_s == 1)
    N = np.sum(labels_s == 0)

    if P == 0:
        print("[ERROR] Không tìm thấy mẫu dương (Target) nào!")
        return 1.0
    if N == 0:
        print("[ERROR] Không tìm thấy mẫu âm (Spoof/Nontarget) nào!")
        return 1.0

    tp = 0
    fp = 0
    fn = P
    tn = N
    best_gap = 1e9
    best_eer = 1.0
    prev_score = None

    for s, y in zip(scores_s, labels_s):
        if prev_score is None or s != prev_score:
            far = fp / N
            frr = fn / P
            gap = abs(far - frr)
            if gap < best_gap:
                best_gap = gap
                best_eer = (far + frr) / 2.0
        prev_score = s

        if y == 1:
            tp += 1; fn -= 1
        else:
            fp += 1; tn -= 1

    return float(best_eer)


# ==========================================
#  MODEL LOADING (GIỮ NGUYÊN)
# ==========================================
def load_ecapa(model_path, device):
    model = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=5994, m=0.2, s=30, test_step=1)
    model.load_parameters(model_path)
    model.to(device)
    model.eval()
    return model


def process_ecapa_audio(file_paths, max_frames=300):
    audio_segments = []
    for path in file_paths:
        data, _ = sf.read(path)
        if len(data.shape) > 1: data = data[:, 0]
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
def ecapa_embed_pair(model, file_paths, device):
    data_1, data_2 = process_ecapa_audio(file_paths)
    data_1, data_2 = data_1.to(device), data_2.to(device)
    emb_1 = F.normalize(model.speaker_encoder.forward(data_1, aug=False), p=2, dim=1)
    emb_2 = F.normalize(model.speaker_encoder.forward(data_2, aug=False), p=2, dim=1)
    return emb_1, emb_2


@torch.no_grad()
def asv_score_from_emb(enr_emb, tst_emb) -> float:
    emb_1_A, emb_2_A = enr_emb
    emb_1_B, emb_2_B = tst_emb
    score_1 = torch.mean(torch.matmul(emb_1_A, emb_1_B.T))
    score_2 = torch.mean(torch.matmul(emb_2_A, emb_2_B.T))
    return float(((score_1 + score_2) / 2.0).item())


def pad_aasist(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len: return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


def load_aasist(config_path, weight_path, device):
    with open(config_path, "r") as f: config = json.load(f)
    model = AASISTModel(config["model_config"]).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def cm_score_aasist(model, audio_path, device) -> float:
    X, _ = sf.read(audio_path)
    if len(X.shape) > 1: X = X[:, 0]
    X_pad = pad_aasist(X, 64600)
    x_inp = torch.Tensor(X_pad).unsqueeze(0).to(device)
    _, output = model(x_inp)
    return float(output[:, 1].item())


# ==========================================
#  DATA LOADING (SỬA LẠI ĐỂ ĐỌC ĐÚNG FORMAT CỦA BẠN)
# ==========================================
def read_enroll_list(path: str) -> Dict[str, List[str]]:
    enr = {}
    if not os.path.exists(path): return enr
    with open(path, "r") as f:
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
    key: str  # target / nontarget / spoof


def read_asv_trials(path: str) -> List[Trial]:
    """
    Đọc file protocol ASV.
    Format mong đợi: SPK TEST_ID TYPE KEY
    Ví dụ: LA_0014 LA_E_8367413 bonafide target
           LA_0026 LA_E_1154302 A13 spoof
    """
    trials = []
    if not os.path.exists(path): return trials
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()

            # Đảm bảo dòng có đủ cột (ít nhất 4 cột)
            if len(parts) >= 4:
                spk = parts[0]
                test_id = parts[1]
                # parts[2] là type (bonafide/A13...), ta không cần dùng trực tiếp
                key = parts[3]  # Cột quan trọng nhất: target / nontarget / spoof

                trials.append(Trial(spk=spk, test_id=test_id, key=key))
    return trials


def resolve_path(file_id, dirs):
    fn = f"{file_id}.flac"
    for d in dirs:
        p = os.path.join(d, fn)
        if os.path.exists(p): return p
    return None


# ==========================================
#  MAIN PROGRAM
# ==========================================
def main():
    log_path, restore_io = setup_run_log("./log", prefix="eval_sasv_FIXED")
    print(f"[LOG] Logging started at: {log_path}")

    try:
        # ------- CONFIG PATHS (Hãy kiểm tra lại đường dẫn của bạn) -------
        LA_ROOT = os.path.join(".", "data", "LA", "LA")

        # Audio Eval
        EVAL_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_eval", "flac")
        # Audio Dev (Fallback cho enroll nếu cần)
        DEV_AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_dev", "flac")

        PROT_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols")
        OUTPUT_FILE = "sasv_eval_scores.txt"

        ECAPA_WEIGHT = "./ecapa/exps/pretrain.model"
        AASIST_CONF = "./aasist/config/AASIST.conf"
        AASIST_WEIGHT = "./aasist/models/weights/AASIST.pth"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # --- 1. LOAD DATA ---
        enroll_map = {}
        trials = []

        # Load Male + Female
        for g in ['female', 'male']:
            p_trn = os.path.join(PROT_DIR, f"ASVspoof2019.LA.asv.eval.{g}.trn.txt")
            enroll_map.update(read_enroll_list(p_trn))

            p_trl = os.path.join(PROT_DIR, f"ASVspoof2019.LA.asv.eval.{g}.trl.txt")
            trials.extend(read_asv_trials(p_trl))

        # Fallback to GI if needed
        if not trials:
            print("[WARN] Loading Gender Independent protocols...")
            p_trn = os.path.join(PROT_DIR, "ASVspoof2019.LA.asv.eval.gi.trn.txt")
            enroll_map.update(read_enroll_list(p_trn))
            p_trl = os.path.join(PROT_DIR, "ASVspoof2019.LA.asv.eval.gi.trl.txt")
            trials.extend(read_asv_trials(p_trl))

        print(f"[DATASET] Enroll Speakers: {len(enroll_map)}")
        print(f"[DATASET] Total Trials: {len(trials)}")

        if len(trials) == 0:
            print("[ERROR] No trials loaded. Check paths!")
            return

        # --- 2. LOAD MODELS ---
        ecapa = load_ecapa(ECAPA_WEIGHT, device)
        aasist = load_aasist(AASIST_CONF, AASIST_WEIGHT, device)

        # --- 3. EVALUATION LOOP ---
        enroll_emb_cache = {}
        test_emb_cache = {}
        cm_cache = {}

        final_scores = []
        final_labels = []

        audio_search_dirs = [EVAL_AUDIO_DIR, DEV_AUDIO_DIR]
        print(f"Searching audio in: {audio_search_dirs}")

        print("\nStarting Evaluation...")
        with open(OUTPUT_FILE, "w") as f_out:
            for i, tr in enumerate(trials):
                # ========================================================
                # [CRITICAL FIX] SỬA LOGIC NHÃN (LABEL LOGIC)
                # ========================================================
                # SASV Rule:
                # Chấp nhận (1) = Đúng người VÀ Đúng là người thật
                # Trong protocol của bạn: key == 'target' thỏa mãn điều này.
                # 'nontarget' = sai người -> Reject (0)
                # 'spoof'     = giả mạo   -> Reject (0)

                y = 1 if tr.key == 'target' else 0
                # ========================================================

                # Resolve paths
                enr_ids = enroll_map.get(tr.spk)
                if not enr_ids:
                    # Nếu speaker chưa enroll thì skip
                    continue

                enr_paths = []
                for fid in enr_ids:
                    p = resolve_path(fid, audio_search_dirs)
                    if p: enr_paths.append(p)

                tst_path = resolve_path(tr.test_id, audio_search_dirs)

                if not tst_path or not enr_paths:
                    # Missing audio file
                    continue

                # -- SCORE COMPUTATION --
                # ASV
                if tr.spk not in enroll_emb_cache:
                    enroll_emb_cache[tr.spk] = ecapa_embed_pair(ecapa, enr_paths, device)
                if tr.test_id not in test_emb_cache:
                    test_emb_cache[tr.test_id] = ecapa_embed_pair(ecapa, [tst_path], device)
                s_asv_raw = asv_score_from_emb(enroll_emb_cache[tr.spk], test_emb_cache[tr.test_id])

                # CM
                if tr.test_id not in cm_cache:
                    cm_cache[tr.test_id] = cm_score_aasist(aasist, tst_path, device)
                s_cm_raw = cm_cache[tr.test_id]

                # -- NORMALIZE & FUSE --
                s_asv_norm = apply_norm(s_asv_raw, ASV_MEAN, ASV_STD)
                s_cm_norm = apply_norm(s_cm_raw, CM_MEAN, CM_STD)

                s_sasv = BEST_W * s_cm_norm + (1.0 - BEST_W) * s_asv_norm

                final_scores.append(s_sasv)
                final_labels.append(y)

                f_out.write(f"{tr.spk} {tr.test_id} {s_sasv:.5f} {y}\n")

                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1}/{len(trials)} trials...")

        # --- 4. RESULTS ---
        scores_arr = np.array(final_scores)
        labels_arr = np.array(final_labels)

        eer = compute_eer(scores_arr, labels_arr)

        print("\n" + "=" * 40)
        print(f"EVALUATION COMPLETE")
        print("=" * 40)
        print(f"Total Trials processed: {len(scores_arr)}")
        print(f"Positive samples (target): {np.sum(labels_arr == 1)}")
        print(f"Negative samples (nontarget+spoof): {np.sum(labels_arr == 0)}")
        print(f"Used W       : {BEST_W}")
        print(f"SASV EER     : {eer * 100:.4f}%")
        print(f"Score File   : {os.path.abspath(OUTPUT_FILE)}")
        print(f"Log File     : {os.path.abspath(log_path)}")
        print("=" * 40 + "\n")

    except Exception:
        print("\n[ERROR] Exception occurred!")
        traceback.print_exc()
    finally:
        restore_io()


if __name__ == "__main__":
    main()