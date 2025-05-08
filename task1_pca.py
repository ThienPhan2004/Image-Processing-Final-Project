import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # Thuật toán PCA để giảm chiều dữ liệu
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio  # Công cụ đo lường chất lượng ảnh
import os

# Hàm padding ảnh sao cho chia hết cho block_size
def pad_image(img, block_size):
    """
    Thêm padding vào ảnh để kích thước chiều cao và chiều rộng chia hết cho block_size.
    - img: Ảnh đầu vào (mảng numpy, thường có 3 kênh RGB hoặc 1 kênh xám).
    - block_size: Kích thước khối (int), thường là 8 hoặc 16 để chia ảnh thành các khối vuông.
    - pad_h, pad_w: Số pixel padding thêm vào chiều cao và chiều rộng.
    - Trả về: Ảnh đã padding, pad_h, pad_w để xử lý sau khi tái tạo.
    """
    h, w = img.shape[:2]  # Lấy chiều cao (h) và chiều rộng (w) của ảnh
    pad_h = (block_size - h % block_size) % block_size  # Tính số padding cần cho chiều cao
    pad_w = (block_size - w % block_size) % block_size  # Tính số padding cần cho chiều rộng
    padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')  # Thêm padding với giá trị 0
    return padded_img, pad_h, pad_w

# Hàm chia một kênh ảnh thành các khối vuông và làm phẳng thành vector
def divide_into_blocks(channel, block_size):
    """
    Chia một kênh ảnh thành các khối vuông kích thước block_size x block_size và làm phẳng thành vector.
    - channel: Kênh màu của ảnh (mảng numpy 2D, thường là R, G, hoặc B).
    - block_size: Kích thước khối (int), xác định kích thước mỗi khối vuông.
    - Trả về: Mảng numpy chứa các vector đã làm phẳng từ các khối.
    """
    h, w = channel.shape  # Lấy kích thước của kênh ảnh
    blocks = []  # Danh sách để lưu các khối
    for i in range(0, h, block_size):  # Duyệt theo bước block_size theo chiều cao
        for j in range(0, w, block_size):  # Duyệt theo bước block_size theo chiều rộng
            block = channel[i:i+block_size, j:j+block_size]  # Lấy khối con
            blocks.append(block.flatten())  # Làm phẳng khối thành vector và thêm vào danh sách
    return np.array(blocks)  # Chuyển danh sách thành mảng numpy

# Hàm tái tạo lại các khối ảnh từ PCA
def reconstruct_blocks(blocks_pca, pca, block_size):
    """
    Tái tạo các khối ảnh từ dữ liệu đã giảm chiều bằng PCA.
    - blocks_pca: Dữ liệu đã giảm chiều từ PCA (mảng numpy).
    - pca: Đối tượng PCA đã được huấn luyện để tái tạo dữ liệu.
    - block_size: Kích thước khối (int), dùng để định hình lại các khối.
    - Trả về: Mảng chứa các khối đã tái tạo.
    """
    reconstructed = pca.inverse_transform(blocks_pca)  # Tái tạo dữ liệu gốc từ thành phần chính
    blocks = reconstructed.reshape((-1, block_size, block_size))  # Định hình lại thành các khối 2D
    return blocks

# Hàm ghép các khối nhỏ lại thành ảnh
def merge_blocks(blocks, h, w, block_size):
    """
    Ghép các khối nhỏ lại thành ảnh hoàn chỉnh.
    - blocks: Mảng chứa các khối đã tái tạo (kích thước block_size x block_size).
    - h, w: Chiều cao và chiều rộng của ảnh gốc (trước khi padding).
    - block_size: Kích thước khối (int), dùng để định vị các khối.
    - Trả về: Ảnh hoàn chỉnh sau khi ghép (mảng numpy).
    """
    img = np.zeros((h, w), dtype=np.uint8)  # Tạo mảng ảnh mới với kích thước gốc, giá trị khởi tạo là 0
    idx = 0  # Chỉ số để duyệt qua các khối
    for i in range(0, h, block_size):  # Duyệt theo bước block_size theo chiều cao
        for j in range(0, w, block_size):  # Duyệt theo bước block_size theo chiều rộng
            img[i:i+block_size, j:j+block_size] = np.clip(blocks[idx], 0, 255)  # Gán khối vào vị trí tương ứng, clip để giữ giá trị trong [0, 255]
            idx += 1  # Tăng chỉ số khối
    return img

# Hàm nén một kênh ảnh bằng PCA
def compress_channel(channel, block_size, n_components):
    """
    Nén một kênh ảnh (R, G, hoặc B) bằng PCA trên các khối.
    - channel: Kênh màu của ảnh (mảng numpy 2D).
    - block_size: Kích thước khối (int), chia ảnh thành các khối vuông.
    - n_components: Số thành phần chính giữ lại (int), xác định mức độ nén.
    - Trả về: Kênh ảnh đã nén, đối tượng PCA, và số hệ số mỗi khối.
    """
    h, w = channel.shape  # Lấy kích thước của kênh ảnh
    blocks = divide_into_blocks(channel, block_size)  # Chia thành các khối và làm phẳng
    pca = PCA(n_components=n_components)  # Khởi tạo PCA với số thành phần chính
    transformed = pca.fit_transform(blocks)  # Giảm chiều dữ liệu
    reconstructed_blocks = reconstruct_blocks(transformed, pca, block_size)  # Tái tạo các khối
    return merge_blocks(reconstructed_blocks, h, w, block_size), pca, transformed.shape[1]  # Trả về ảnh nén, PCA, và số hệ số

# Hàm chính: nén ảnh bằng PCA
def compress_image_pca(image_path, block_size=8, n_components=20):
    """
    Nén ảnh bằng phương pháp PCA trên các khối.
    - image_path: Đường dẫn đến tệp ảnh đầu vào (string).
    - block_size: Kích thước khối (default=8), xác định kích thước mỗi khối vuông.
    - n_components: Số thành phần chính giữ lại (default=20), điều chỉnh mức độ nén.
    - Thực hiện: Đọc ảnh, padding, nén từng kênh (R, G, B), tái tạo ảnh, tính MSE/PSNR, lưu và hiển thị kết quả.
    """
    # Kiểm tra file có tồn tại không
    if not os.path.exists(image_path):  # Kiểm tra xem tệp có tồn tại trong hệ thống không
        print(f"❌ Lỗi: File '{image_path}' không tồn tại.")  # Báo lỗi nếu không tìm thấy
        return

    # Đọc ảnh
    img = cv2.imread(image_path)  # Đọc ảnh bằng OpenCV (trả về mảng BGR)
    if img is None:  # Kiểm tra xem ảnh có đọc được không
        print(f"❌ Lỗi: Không thể đọc được ảnh từ '{image_path}'. Có thể định dạng không hỗ trợ hoặc file bị hỏng.")  # Báo lỗi nếu không đọc được
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR (OpenCV) sang RGB (Matplotlib)
    filename = os.path.splitext(os.path.basename(image_path))[0]  # Lấy tên file không bao gồm phần mở rộng

    # Pad ảnh để kích thước chia hết cho block_size
    padded_img, pad_h, pad_w = pad_image(img, block_size)  # Thêm padding vào ảnh
    h, w = padded_img.shape[:2]  # Lấy kích thước của ảnh đã padding

    compressed_channels = []  # Danh sách để lưu các kênh đã nén
    total_coeffs = 0  # Tổng số hệ số sau khi nén
    for i in range(3):  # Lặp qua 3 kênh màu (R, G, B)
        channel = padded_img[:, :, i]  # Lấy từng kênh màu
        compressed_channel, pca, coeffs_per_block = compress_channel(channel, block_size, n_components)  # Nén kênh
        compressed_channels.append(compressed_channel)  # Thêm kênh nén vào danh sách
        total_coeffs += (h // block_size) * (w // block_size) * coeffs_per_block  # Cập nhật tổng số hệ số

    # Ghép lại ảnh
    reconstructed_img = np.stack(compressed_channels, axis=2)  # Kết hợp 3 kênh thành ảnh RGB
    reconstructed_img = reconstructed_img[:h - pad_h if pad_h else h, :w - pad_w if pad_w else w, :]  # Loại bỏ padding

    # Tính MSE (Mean Squared Error) và PSNR (Peak Signal-to-Noise Ratio)
    mse = mean_squared_error(img, reconstructed_img)  # Tính lỗi bình phương trung bình giữa ảnh gốc và ảnh nén
    psnr = peak_signal_noise_ratio(img, reconstructed_img, data_range=255)  # Tính tỷ số tín hiệu trên nhiễu, data_range=255 cho ảnh 8-bit

    # Tính tỷ lệ nén
    compression_ratio = (total_coeffs * 4.0) / (img.shape[0] * img.shape[1] * 3)  # Tỷ lệ nén dựa trên số bit trước/sau nén

    # Lưu ảnh
    output_filename = f"{filename}_pca_k{n_components}.jpg"  # Tạo tên file đầu ra với tham số n_components
    cv2.imwrite(output_filename, cv2.cvtColor(reconstructed_img.astype(np.uint8), cv2.COLOR_RGB2BGR))  # Lưu ảnh, chuyển lại sang BGR

    # Hiển thị
    plt.figure(figsize=(10, 5))  # Tạo figure với kích thước 10x5 inch
    plt.subplot(1, 2, 1)  # Tạo subplot 1 cho ảnh gốc
    plt.imshow(img)  # Hiển thị ảnh gốc
    plt.title("Ảnh gốc")  # Thêm tiêu đề
    plt.axis('off')  # Ẩn trục tọa độ

    plt.subplot(1, 2, 2)  # Tạo subplot 2 cho ảnh nén
    plt.imshow(reconstructed_img.astype(np.uint8))  # Hiển thị ảnh nén
    plt.title(f"Ảnh nén (k = {n_components})")  # Thêm tiêu đề với số thành phần chính
    plt.axis('off')  # Ẩn trục tọa độ
    plt.tight_layout()  # Tối ưu bố cục
    plt.show()  # Hiển thị ảnh

    # In kết quả
    print(f"MSE: {mse:.2f}")  # In lỗi MSE với 2 chữ số thập phân
    print(f"PSNR: {psnr:.2f} dB")  # In PSNR với đơn vị decibel
    print(f"Tỷ lệ nén (ước lượng): {compression_ratio:.4f}")  # In tỷ lệ nén với 4 chữ số thập phân
    print(f"✅ Ảnh kết quả đã lưu tại: {output_filename}")  # Xác nhận ảnh đã lưu

# Gọi hàm nén ảnh
compress_image_pca("task1.jpg", block_size=16, n_components=5)  # Thực thi hàm với ảnh cụ thể, block_size=16, n_components=5