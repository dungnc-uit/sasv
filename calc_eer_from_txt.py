import os
import numpy as np
# import sys

# ================= CẤU HÌNH (SỬA LẠI ĐƯỜNG DẪN CỦA BẠN) =================

# 1. Đường dẫn file điểm số bạn đã tạo ra (sasv_eval_scores.txt)
SCORE_FILE = "sasv_eval_scores.txt"

# 2. Đường dẫn các file Protocol ASV (Chứa nhãn target/nontarget/spoof)
# Bạn nên điền cả file Female và Male vào list này
PROTOCOL_FILES = [
    "./data/LA/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.female.trl.txt",
    "./data/LA/LA/ASVspoof2019_LA_asv_protocols/ASVspoof2019.LA.asv.eval.male.trl.txt"
]


# ========================================================================

def compute_eer(scores, labels):
    """Tính EER chuẩn"""
    scores = np.array(scores)
    labels = np.array(labels)

    # Sắp xếp giảm dần theo điểm số
    idx = np.argsort(scores)[::-1]
    scores_s = scores[idx]
    labels_s = labels[idx]

    P = np.sum(labels_s == 1)
    N = np.sum(labels_s == 0)

    if P == 0 or N == 0:
        return 1.0, 0.0, 0.0

    tp = 0
    fp = 0
    fn = P
    tn = N
    best_gap = 1.0
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

    return best_eer, P, N


def load_ground_truth(protocol_paths):
    """Đọc file protocol để lấy nhãn đúng của từng file test"""
    gt = {}
    print("Loading protocols...")
    for path in protocol_paths:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue

        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4: continue

                # Format: SPK TEST_ID TYPE KEY
                # VD: LA_0014 LA_E_8367413 bonafide target
                test_id = parts[1]
                key = parts[3]  # target / nontarget / spoof

                # LOGIC QUAN TRỌNG NHẤT CỦA SASV:
                # Label 1 (Accept) = 'target'
                # Label 0 (Reject) = 'nontarget' HOẶC 'spoof'
                label = 1 if key == 'target' else 0

                gt[test_id] = label
    return gt


def main():
    # 1. Load nhãn đúng
    ground_truth = load_ground_truth(PROTOCOL_FILES)
    if not ground_truth:
        print("Error: Could not load any ground truth labels.")
        return
    print(f"-> Loaded labels for {len(ground_truth)} files.")

    # 2. Load điểm số từ file txt
    if not os.path.exists(SCORE_FILE):
        print(f"Error: Score file not found: {SCORE_FILE}")
        return

    print(f"Reading scores from {SCORE_FILE}...")
    final_scores = []
    final_labels = []

    missing = 0
    cnt_target = 0
    cnt_nontarget = 0
    cnt_spoof = 0  # Lưu ý: nontarget và spoof đều tính là Negative (0)

    with open(SCORE_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Format file score: spk test_id score label_cu(sai)
            if len(parts) < 3: continue

            test_id = parts[1]
            try:
                score = float(parts[2])
            except ValueError:
                continue

            # Ghép điểm với nhãn đúng
            if test_id in ground_truth:
                true_label = ground_truth[test_id]
                final_scores.append(score)
                final_labels.append(true_label)

                if true_label == 1:
                    cnt_target += 1
                else:
                    cnt_nontarget += 1  # Gộp chung nontarget + spoof
            else:
                missing += 1

    # 3. Tính toán
    if len(final_scores) == 0:
        print("No matching files found between Score file and Protocols.")
        return

    print(f"-> Matched {len(final_scores)} trials. (Missing in protocol: {missing})")

    eer, P, N = compute_eer(final_scores, final_labels)

    print("\n" + "=" * 40)
    print("RE-CALCULATED RESULTS (SASV)")
    print("=" * 40)
    print(f"Total Positive (Target)     : {int(P)}")
    print(f"Total Negative (Imp + Spoof): {int(N)}")
    print("-" * 20)
    print(f"Final SASV EER              : {eer * 100:.4f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()