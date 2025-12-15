import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf


# ===== ASV imports =====
from ecapa.ECAPAModel import ECAPAModel

# ===== CM imports =====
from aasist.models.AASIST import Model as AASISTModel


# --------------------------
# ASV (ECAPA) utilities
# --------------------------
def load_ecapa(model_path: str, device: torch.device):
    model = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=5994, m=0.2, s=30, test_step=1)
    model.load_parameters(model_path)
    model.to(device)
    model.eval()
    return model


def process_ecapa_audio(file_paths, max_frames=300):
    audio_segments = []
    for path in file_paths:
        data, sr = sf.read(path)
        if len(data.shape) > 1:
            data = data[:, 0]
        audio_segments.append(data)

    if len(audio_segments) == 0:
        raise ValueError("Enrollment/Test file list is empty")

    full_audio = np.concatenate(audio_segments)

    data_1 = torch.FloatTensor(np.stack([full_audio], axis=0))

    max_audio = max_frames * 160 + 240
    if full_audio.shape[0] <= max_audio:
        shortage = max_audio - full_audio.shape[0]
        audio_padded = np.pad(full_audio, (0, shortage), 'wrap')
    else:
        audio_padded = full_audio

    feats = []
    startframe = np.linspace(0, audio_padded.shape[0] - max_audio, num=5)
    for asf in startframe:
        feats.append(audio_padded[int(asf):int(asf) + max_audio])

    data_2 = torch.FloatTensor(np.stack(feats, axis=0))
    return data_1, data_2


@torch.no_grad()
def asv_score_ecapa(model, enroll_files, test_files, device: torch.device) -> float:
    data_1_A, data_2_A = process_ecapa_audio(enroll_files)
    data_1_B, data_2_B = process_ecapa_audio(test_files)

    data_1_A, data_2_A = data_1_A.to(device), data_2_A.to(device)
    data_1_B, data_2_B = data_1_B.to(device), data_2_B.to(device)

    emb_1_A = F.normalize(model.speaker_encoder.forward(data_1_A, aug=False), p=2, dim=1)
    emb_2_A = F.normalize(model.speaker_encoder.forward(data_2_A, aug=False), p=2, dim=1)

    emb_1_B = F.normalize(model.speaker_encoder.forward(data_1_B, aug=False), p=2, dim=1)
    emb_2_B = F.normalize(model.speaker_encoder.forward(data_2_B, aug=False), p=2, dim=1)

    score_1 = torch.mean(torch.matmul(emb_1_A, emb_1_B.T))
    score_2 = torch.mean(torch.matmul(emb_2_A, emb_2_B.T))
    score = (score_1 + score_2) / 2.0

    return float(score.item())  # ~[-1, 1]


# --------------------------
# CM (AASIST) utilities
# --------------------------
def pad_aasist(x, max_len=64600):
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
def cm_score_aasist(model, test_audio_path: str, device: torch.device) -> float:
    X, _ = sf.read(test_audio_path)
    if len(X.shape) > 1:
        X = X[:, 0]

    X_pad = pad_aasist(X, 64600)
    x_inp = torch.Tensor(X_pad).unsqueeze(0).to(device)

    _, output = model(x_inp)
    bonafide_logit = output[:, 1].item()

    # scale logit -> [-1, 1] giống file của bạn
    aasist_min = -20.0
    aasist_max = 20.0
    val_clipped = max(aasist_min, min(aasist_max, bonafide_logit))
    bonafide_score = 2 * (val_clipped - aasist_min) / (aasist_max - aasist_min) - 1
    return float(bonafide_score)  # [-1, 1]


# --------------------------
# Fusion
# --------------------------
def fuse_scores(s_cm: float, s_asv: float, w: float) -> float:
    """
    Weighted sum fusion.
    w = 0   -> chỉ ASV
    w = 1   -> chỉ CM
    """
    w = max(0.0, min(1.0, float(w)))
    return w * s_cm + (1.0 - w) * s_asv


def decide(s_final: float, threshold_final: float = 0.0) -> bool:
    return s_final >= threshold_final


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Paths =====
    ECAPA_WEIGHT = "./ecapa/exps/pretrain.model"
    AASIST_CONF = "./aasist/config/AASIST.conf"
    AASIST_WEIGHT = "./aasist/models/weights/AASIST.pth"

    # ===== Example inputs =====
    ENROLL_FILES = [
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_4356541.flac",
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_6092752.flac",
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_5090421.flac",
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_6718237.flac",
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_6073354.flac",
    ]

    # target
    # TEST_FILE_LIST = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_3513335.flac"]
    TEST_FILE_LIST = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_7933208.flac"]
    # nontarget
    # TEST_FILE_LIST = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_8980393.flac"]
    # TEST_FILE_LIST = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_1608630.flac"]
    # spoof
    # TEST_FILE_LIST = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_4067937.flac"]
    # TEST_FILE_LIST = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_6479435.flac"]
    TEST_FILE_FOR_CM = TEST_FILE_LIST[0]

    # ===== Checks =====
    for p in ENROLL_FILES + TEST_FILE_LIST:
        if not os.path.exists(p):
            print(f"Missing audio: {p}")
            sys.exit(1)
    for p in [ECAPA_WEIGHT, AASIST_CONF, AASIST_WEIGHT]:
        if not os.path.exists(p):
            print(f"Missing model/config: {p}")
            sys.exit(1)

    # ===== Load models =====
    ecapa = load_ecapa(ECAPA_WEIGHT, device)
    aasist = load_aasist(AASIST_CONF, AASIST_WEIGHT, device)

    # ===== Run =====
    s_asv = asv_score_ecapa(ecapa, ENROLL_FILES, TEST_FILE_LIST, device)
    s_cm = cm_score_aasist(aasist, TEST_FILE_FOR_CM, device)
    w = 0.6
    s_final = fuse_scores(s_cm=s_cm, s_asv=s_asv, w=w)

    print("========== RESULTS ==========")
    print(f"S_ASV   (ECAPA cosine): {s_asv:.4f}")
    print(f"S_CM    (AASIST bona):  {s_cm:.4f}")
    print(f"S_Final (w={w}):        {s_final:.4f}")
    print("=============================")
