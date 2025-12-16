import json
import torch
import numpy as np
import soundfile as sf
from aasist.models.AASIST import Model  # Import kiến trúc mô hình từ file models/AASIST.py

def pad(x, max_len=64600):
    """
    Hàm xử lý độ dài âm thanh: cắt nếu quá dài hoặc lặp lại nếu quá ngắn.
    Được trích xuất từ data_utils.py
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # Padding bằng cách lặp lại tín hiệu nếu ngắn hơn max_len
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def predict_bonafide_score(audio_path, config_path, model_weight_path):
    # 1. Thiết lập thiết bị (GPU hoặc CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Tải cấu hình từ file .conf (định dạng JSON)
    with open(config_path, "r") as f:
        config = json.load(f)

    # 3. Khởi tạo mô hình và tải trọng số
    # Khởi tạo kiến trúc AASIST dựa trên tham số trong config
    model = Model(config["model_config"]).to(device)
    # Tải trọng số đã huấn luyện (pretrained weights)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval() # Chuyển sang chế độ đánh giá (evaluation mode)

    # 4. Xử lý file âm thanh đầu vào
    try:
        # Đọc file âm thanh (X là dữ liệu sóng âm, _ là sample rate)
        X, _ = sf.read(audio_path)
    except Exception as e:
        print(f"Không thể đọc file âm thanh: {e}")
        return None

    # Cắt/Đệm về độ dài cố định 64600 mẫu (~4 giây) theo yêu cầu của mô hình
    X_pad = pad(X, 64600)
    # Chuyển thành Tensor và thêm chiều batch: (1, 64600)
    x_inp = torch.Tensor(X_pad).unsqueeze(0).to(device)

    # 5. Dự đoán (Inference)
    with torch.no_grad():
        # Mô hình trả về (last_hidden, output). Output chứa logits của 2 lớp.
        _, output = model(x_inp)

        # Lớp 0 là Spoof, Lớp 1 là Bonafide. Lấy điểm số của lớp Bonafide.
        # Điểm số này càng cao thì khả năng là giọng thật càng lớn.
        bonafide_score = output[:, 1].item()

    aasist_min = -20.0
    aasist_max = 20.0
    val_clipped = max(aasist_min, min(aasist_max, bonafide_score))
    bonafide_score = 2 * (val_clipped - aasist_min) / (aasist_max - aasist_min) - 1

    return bonafide_score

if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    # Đường dẫn đến file âm thanh bạn muốn kiểm tra
    # target
    input_audio = "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_3513335.flac"
    # nontarget
    # input_audio = "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_1608630.flac"
    # input_audio = "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_8980393.flac"
    # spoof
    # input_audio = "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_4067937.flac"
    # input_audio = "./data/LA/LA/ASVspoof2019_LA_dev/flac/LA_D_6479435.flac"

    # Đường dẫn đến file cấu hình và trọng số (theo file bạn đã upload)
    config_file = "./aasist/config/AASIST.conf"
    weight_file = "./aasist/models/weights/AASIST.pth"

    # --- CHẠY DEMO ---
    score = predict_bonafide_score(input_audio, config_file, weight_file)

    if score is not None:
        print("--------------------------------------------------")
        print(f"File: {input_audio}")
        print(f"Bonafide Score (Điểm giọng thật): {score:.4f}")
        print("--------------------------------------------------")
