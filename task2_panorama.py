import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Hàm để tải ảnh với kiểm tra lỗi
def load_images(img1_path, img2_path):
    # Kiểm tra sự tồn tại của file ảnh đầu vào
    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"Lỗi: File ảnh '{img1_path}' không tồn tại.")
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"Lỗi: File ảnh '{img2_path}' không tồn tại.")
    
    # Đọc các ảnh đầu vào dưới dạng ảnh màu
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    
    # Kiểm tra kết quả trả về từ cv2.imread
    if img1 is None:
        raise ValueError(f"Lỗi: Không thể đọc file ảnh '{img1_path}'. Đảm bảo file không bị hỏng và định dạng được hỗ trợ.")
    if img2 is None:
        raise ValueError(f"Lỗi: Không thể đọc file ảnh '{img2_path}'. Đảm bảo file không bị hỏng và định dạng được hỗ trợ.")
    
    return img1, img2

# Hàm để phát hiện các đặc trưng SIFT và tính toán descriptors
def detect_sift_features(img1, img2):
    # Chuyển đổi ảnh sang ảnh xám cho SIFT
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Tạo bộ phát hiện SIFT
    sift = cv2.SIFT_create()
    
    # Phát hiện các điểm khóa và tính toán descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    return keypoints1, descriptors1, keypoints2, descriptors2

# Hàm để khớp các điểm khóa sử dụng BFMatcher và Lowe's Ratio Test
def match_keypoints(descriptors1, descriptors2, ratio=0.75):
    # Tạo BFMatcher với các tham số mặc định
    bf = cv2.BFMatcher()
    
    # Thực hiện khớp k-nearest neighbor (k=2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Áp dụng Lowe's Ratio Test để lọc các cặp khớp tốt
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    
    return good_matches

# Hàm để ước lượng ma trận homography sử dụng RANSAC
def estimate_homography(keypoints1, keypoints2, good_matches):
    # Trích xuất các điểm khớp
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Ước lượng ma trận homography sử dụng RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Đếm số lượng inliers
    inliers = np.sum(mask)
    
    return H, inliers

# Hàm để biến đổi và ghép ảnh
def stitch_images(img1, img2, H):
    # Lấy kích thước của các ảnh
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Xác định các góc của ảnh thứ nhất
    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # Biến đổi các góc của ảnh thứ nhất
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    # Kết hợp với các góc của ảnh thứ hai
    all_corners = np.concatenate((transformed_corners, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)
    
    # Tìm khung bao
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Tính ma trận dịch chuyển
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    
    # Biến đổi ảnh thứ nhất
    warped_img1 = cv2.warpPerspective(img1, translation.dot(H), (x_max - x_min, y_max - y_min))
    
    # Biến đổi ảnh thứ hai (biến đổi đơn vị)
    warped_img2 = cv2.warpPerspective(img2, translation, (x_max - x_min, y_max - y_min))
    
    # Tạo mặt nạ để trộn ảnh
    mask1 = np.any(warped_img1 != 0, axis=2).astype(np.float32)
    mask2 = np.any(warped_img2 != 0, axis=2).astype(np.float32)
    
    # Trộn đơn giản: sử dụng warped_img1 nếu có, nếu không thì dùng warped_img2
    mask = mask1 / (mask1 + mask2 + 1e-10)  # Tránh chia cho 0
    mask = np.stack([mask] * 3, axis=2)
    
    # Trộn các ảnh
    panorama = warped_img1 * mask + warped_img2 * (1 - mask)
    panorama = panorama.astype(np.uint8)
    
    return panorama

# Hàm để trực quan hóa các cặp khớp (tùy chọn)
def draw_matches(img1, keypoints1, img2, keypoints2, good_matches):
    # Vẽ các cặp khớp
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Chuyển đổi BGR sang RGB cho Matplotlib
    match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    
    # Hiển thị các cặp khớp
    plt.figure(figsize=(15, 5))
    plt.imshow(match_img_rgb)
    plt.title('Các Cặp Khớp Điểm Khóa SIFT')
    plt.axis('off')
    plt.savefig('keypoint_matches.png')
    plt.close()

# Hàm chính để tạo ảnh panorama
def create_panorama(img1_path, img2_path, output_path='panorama_result.jpg', min_matches=10):
    # Bước 1: Tải ảnh
    img1, img2 = load_images(img1_path, img2_path)
    
    # Bước 2: Phát hiện các đặc trưng SIFT
    keypoints1, descriptors1, keypoints2, descriptors2 = detect_sift_features(img1, img2)
    
    # Bước 3: Khớp các điểm khóa với Lowe's Ratio Test
    good_matches = match_keypoints(descriptors1, descriptors2)
    
    # Kiểm tra xem có đủ cặp khớp tốt hay không
    if len(good_matches) < min_matches:
        raise ValueError(f"Số cặp khớp tốt không đủ: {len(good_matches)} (yêu cầu: {min_matches})")
    
    # Bước 4: Ước lượng homography với RANSAC
    H, inliers = estimate_homography(keypoints1, keypoints2, good_matches)
    
    # Kiểm tra xem có đủ inliers hay không
    if inliers < min_matches:
        raise ValueError(f"Số inliers không đủ: {inliers} (yêu cầu: {min_matches})")
    
    # Bước 5: Biến đổi và ghép ảnh
    panorama = stitch_images(img1, img2, H)
    
    # Bước 6: Lưu ảnh panorama
    cv2.imwrite(output_path, panorama)
    
    # Bước 7: Hiển thị ảnh panorama bằng Matplotlib
    panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 20))
    plt.imshow(panorama_rgb)
    plt.title('Kết Quả Panorama')
    plt.axis('off')
    plt.savefig('panorama_display.png')
    plt.tight_layout()
    plt.show()
    
    # Bước 8 (Tùy chọn): Trực quan hóa các cặp khớp
    draw_matches(img1, keypoints1, img2, keypoints2, good_matches)
    
    return panorama

# Ví dụ sử dụng
if __name__ == "__main__":
    try:
        # Các đường dẫn đến ảnh
        img1_path = "task2_image1.jpg"
        img2_path = "task2_image2.jpg"
        output_path = "panorama_result.jpg"
        
        # Tạo ảnh panorama
        panorama = create_panorama(img1_path, img2_path, output_path)
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")