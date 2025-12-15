import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import sys
import os

# from torch.xpu import device

# Import model structure từ các file có sẵn
from ecapa.ECAPAModel import ECAPAModel


def load_model(model_path, device):
    """
    Khởi tạo và load trọng số cho mô hình.
    Các tham số C=1024, m=0.2, s=30, n_class=5994 là mặc định của pretrain.model
    """
    # Khởi tạo mô hình với tham số mặc định khớp với 'trainECAPAModel.py'
    model = ECAPAModel(lr=0.001, lr_decay=0.97, C=1024, n_class=5994, m=0.2, s=30, test_step=1)

    # Load trọng số đã huấn luyện (pretrained)
    print(f"Đang tải mô hình từ: {model_path}...")
    model.load_parameters(model_path)
    model.to(device)
    model.eval()  # Chuyển sang chế độ đánh giá
    return model


def process_audio(file_paths, max_frames=300):
    """
    Đọc file FLAC/WAV và xử lý giống như trong ECAPAModel.eval_network
    """
    audio_segments = []

    # 1. Đọc từng file và đưa vào danh sách
    for path in file_paths:
        data, sr = sf.read(path)
        # Đảm bảo là mono
        if len(data.shape) > 1:
            data = data[:, 0]
        audio_segments.append(data)

    # 2. Nối các đoạn lại với nhau (Concatenate)
    if len(audio_segments) > 0:
        full_audio = np.concatenate(audio_segments)
    else:
        raise ValueError("Danh sách file trống")

    # 3. Tiếp tục xử lý như bình thường (tạo data_1 và data_2)
    # data_1: Full utterance (đoạn đã nối dài)
    data_1 = torch.FloatTensor(np.stack([full_audio], axis=0))

    # data_2: Cắt đoạn (Cropping) từ đoạn đã nối
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


def compute_score(model, file1, file2, device):
    """
    Tính điểm tương đồng giữa 2 file
    """
    # Xử lý âm thanh đầu vào
    data_1_A, data_2_A = process_audio(file1)
    data_1_B, data_2_B = process_audio(file2)

    # Đưa lên GPU/CPU
    data_1_A, data_2_A = data_1_A.to(device), data_2_A.to(device)
    data_1_B, data_2_B = data_1_B.to(device), data_2_B.to(device)

    with torch.no_grad():
        # --- Lấy Embedding cho File 1 ---
        # Embedding toàn bộ câu
        emb_1_A = model.speaker_encoder.forward(data_1_A, aug=False)
        emb_1_A = F.normalize(emb_1_A, p=2, dim=1)
        # Embedding các đoạn cắt
        emb_2_A = model.speaker_encoder.forward(data_2_A, aug=False)
        emb_2_A = F.normalize(emb_2_A, p=2, dim=1)

        # --- Lấy Embedding cho File 2 ---
        emb_1_B = model.speaker_encoder.forward(data_1_B, aug=False)
        emb_1_B = F.normalize(emb_1_B, p=2, dim=1)
        emb_2_B = model.speaker_encoder.forward(data_2_B, aug=False)
        emb_2_B = F.normalize(emb_2_B, p=2, dim=1)

        # --- Tính điểm Cosine Similarity ---
        # Score 1: So sánh toàn bộ câu với toàn bộ câu
        score_1 = torch.mean(torch.matmul(emb_1_A, emb_1_B.T))

        # Score 2: So sánh trung bình các đoạn cắt
        score_2 = torch.mean(torch.matmul(emb_2_A, emb_2_B.T))

        # Điểm cuối cùng là trung bình cộng
        score = (score_1 + score_2) / 2

    return score.item()


if __name__ == "__main__":
    # Cấu hình
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # Đường dẫn đến file model pretrain (bạn cần file này từ thư mục exps)
    MODEL_PATH = "exps/pretrain.model"

    # Đường dẫn 2 file âm thanh (FLAC) cần so sánh
    # Bạn thay thế bằng đường dẫn thực tế của bạn
    FILE_AUDIO_1 = [
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_4356541.flac",
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_6092752.flac",
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_5090421.flac",
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_6718237.flac",
        "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_6073354.flac",
                    ]
    # target
    # FILE_AUDIO_2 = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_3513335.flac"]
    FILE_AUDIO_2 = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_7933208.flac"]
    # nontarget
    # FILE_AUDIO_2 = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_8980393.flac"]
    # FILE_AUDIO_2 = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_1608630.flac"]
    # spoof
    # FILE_AUDIO_2 = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_4067937.flac"]
    # FILE_AUDIO_2 = ["./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_6479435.flac"]

    # Kiểm tra file tồn tại
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file model tại {MODEL_PATH}")
        sys.exit(1)

    # Load model
    # try:
    model = load_model(MODEL_PATH, device)
    print("Mô hình đã sẵn sàng.")

    # Tính điểm
    print(f"Đang so sánh:\n1. {FILE_AUDIO_1}\n2. {FILE_AUDIO_2}")
    score = compute_score(model, FILE_AUDIO_1, FILE_AUDIO_2, device)

    print("-" * 30)
    print(f"Điểm tương đồng (Cosine Score): {score:.4f}")
    print("-" * 30)

    # Ngưỡng tham khảo (thường là khoảng 0.2 - 0.4 tùy vào tập dữ liệu)
    # Nếu > Threshold -> Cùng một người
    print("Nhận xét sơ bộ:")
    if score > 0.25:  # Ngưỡng ví dụ
        print("-> Khả năng cao là CÙNG một người.")
    else:
        print("-> Khả năng cao là KHÁC người.")

    # except Exception as e:
    #     print(f"Đã xảy ra lỗi: {e}")
