import numpy as np
import os

# 1. Cấu hình tham số (từ yêu cầu của bạn)
PARAMS = {
    "pi_tar": 0.9405,
    "pi_non": 0.0095,
    "pi_spoof": 0.05,
    "C_miss": 1,
    "C_fa_imp": 10,  # C_asv_fa
    "C_fa_spoof": 10  # C_cm_fa
}


def load_protocols(paths):
    """Load nhãn thực tế từ file protocol"""
    truth = {}
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        # Key là cặp (Speaker, TestID) để đảm bảo duy nhất
                        truth[(parts[0], parts[1])] = parts[3]  # target/nontarget/spoof
    return truth


def compute_min_tdcf(score_file, truth_map, p):
    # Tách điểm số theo 3 loại nhãn
    scores_tar, scores_imp, scores_spf = [], [], []

    with open(score_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3: continue
            spk, test_id = parts[0], parts[1]
            try:
                score = float(parts[2])
            except:
                continue

            label = truth_map.get((spk, test_id))
            if label == 'target':
                scores_tar.append(score)
            elif label == 'nontarget':
                scores_imp.append(score)
            elif label == 'spoof':
                scores_spf.append(score)

    # Chuyển sang numpy và sort
    s_tar = np.sort(scores_tar)
    s_imp = np.sort(scores_imp)
    s_spf = np.sort(scores_spf)

    # Tạo các ngưỡng (thresholds) từ tất cả các điểm số
    thresholds = np.unique(np.concatenate([s_tar, s_imp, s_spf]))

    # Tính P_miss và P_fa tại mọi ngưỡng (Vectorized)
    n_tar, n_imp, n_spf = len(s_tar), len(s_imp), len(s_spf)

    # Vị trí ngưỡng trong mảng đã sort
    idx_tar = np.searchsorted(s_tar, thresholds)
    idx_imp = np.searchsorted(s_imp, thresholds)
    idx_spf = np.searchsorted(s_spf, thresholds)

    # P_miss: Tỷ lệ Target < Threshold
    p_miss = idx_tar / n_tar
    # P_fa: Tỷ lệ Impostor/Spoof >= Threshold
    p_fa_imp = (n_imp - idx_imp) / n_imp
    p_fa_spf = (n_spf - idx_spf) / n_spf

    # Tính Cost tại mỗi ngưỡng
    costs = (p["C_miss"] * p["pi_tar"] * p_miss) + \
            (p["C_fa_imp"] * p["pi_non"] * p_fa_imp) + \
            (p["C_fa_spoof"] * p["pi_spoof"] * p_fa_spf)

    min_cost = np.min(costs)

    # Tính Cost mặc định (Normalize)
    # Hệ thống chấp nhận tất cả hoặc từ chối tất cả
    cost_reject_all = p["C_miss"] * p["pi_tar"]
    cost_accept_all = (p["C_fa_imp"] * p["pi_non"]) + (p["C_fa_spoof"] * p["pi_spoof"])
    norm_factor = min(cost_reject_all, cost_accept_all)

    return min_cost / norm_factor


# --- THỰC THI ---
LA_ROOT = os.path.join(".", "data", "LA", "LA")
PROT_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_asv_protocols")

protocol_files = [
    os.path.join(PROT_DIR, "ASVspoof2019.LA.asv.eval.female.trl.txt"),
    os.path.join(PROT_DIR, "ASVspoof2019.LA.asv.eval.male.trl.txt")
]
score_file = "sasv_eval_scores.txt"

# Load và tính toán
truth = load_protocols(protocol_files)
if truth:
    tdcf = compute_min_tdcf(score_file, truth, PARAMS)
    print(f"Normalized min t-DCF: {tdcf:.4f}")
else:
    print("Không tìm thấy file protocol.")