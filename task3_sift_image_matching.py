import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Hàm để tải ảnh với kiểm tra lỗi
def load_images(scene_path, object_path):
    # Kiểm tra sự tồn tại của file ảnh đầu vào
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Lỗi: File ảnh cảnh '{scene_path}' không tồn tại.")
    if not os.path.exists(object_path):
        raise FileNotFoundError(f"Lỗi: File ảnh đối tượng '{object_path}' không tồn tại.")
    
    # Đọc các ảnh đầu vào dưới dạng ảnh màu
    scene_img = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    object_img = cv2.imread(object_path, cv2.IMREAD_COLOR)
    
    # Kiểm tra kết quả trả về từ cv2.imread
    if scene_img is None:
        raise ValueError(f"Lỗi: Không thể đọc file ảnh cảnh '{scene_path}'. Đảm bảo file không bị hỏng và định dạng được hỗ trợ.")
    if object_img is None:
        raise ValueError(f"Lỗi: Không thể đọc file ảnh đối tượng '{object_path}'. Đảm bảo file không bị hỏng và định dạng được hỗ trợ.")
    
    return scene_img, object_img

# Hàm để phát hiện SIFT keypoints và tính descriptors
def detect_sift_features(img1, img2):
    # Chuyển đổi ảnh sang ảnh xám cho SIFT
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Tạo bộ phát hiện SIFT
    sift = cv2.SIFT_create()
    
    # Phát hiện keypoints và tính descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    return keypoints1, descriptors1, keypoints2, descriptors2

# Hàm để khớp keypoints sử dụng BFMatcher và Lowe's Ratio Test
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

# Hàm để tìm và vẽ khung bao dựa trên Homography
def find_and_draw_bounding_box(scene_img, object_img, good_matches, keypoints1, keypoints2):
    # Trích xuất các điểm khớp
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Ước lượng ma trận Homography với RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Lấy tọa độ 4 góc của ảnh đối tượng
    h, w = object_img.shape[:2]
    object_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    
    # Biến đổi tọa độ 4 góc từ ảnh đối tượng sang ảnh cảnh sử dụng Homography
    scene_corners = cv2.perspectiveTransform(object_corners, H)
    
    # Vẽ khung bao trên ảnh cảnh
    scene_with_box = scene_img.copy()
    cv2.polylines(scene_with_box, [np.int32(scene_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Lưu ảnh cảnh với khung bao
    cv2.imwrite('scene_with_detected_object.jpg', scene_with_box)
    
    return scene_with_box, mask, scene_corners

# Hàm để hiển thị các cặp điểm khớp với phân biệt inliers/outliers
def draw_matches_with_status(scene_img, object_img, keypoints1, keypoints2, good_matches, mask):
    # Tạo ảnh để vẽ các cặp khớp
    match_img = cv2.drawMatches(object_img, keypoints1, scene_img, keypoints2, good_matches, None,
                                matchColor=(0, 255, 0),  # Màu xanh cho tất cả các cặp khớp ban đầu
                                singlePointColor=None,
                                flags=cv2.DrawMatchesFlags_DEFAULT)
    
    # Vẽ lại các inliers (điểm khớp tốt) bằng màu đỏ
    inliers = mask.ravel().tolist()
    for i, (m, inlier) in enumerate(zip(good_matches, inliers)):
        if inlier:
            # Vẽ lại điểm khớp bằng màu đỏ
            pt1 = (int(keypoints1[m.queryIdx].pt[0]), int(keypoints1[m.queryIdx].pt[1]))
            pt2 = (int(keypoints2[m.trainIdx].pt[0] + object_img.shape[1]), int(keypoints2[m.trainIdx].pt[1]))
            cv2.line(match_img, pt1, pt2, (0, 0, 255), 2)
    
    # Hiển thị ảnh các cặp điểm khớp
    match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 5), dpi=120)
    plt.imshow(match_img_rgb)
    plt.title('Các Cặp Khớp Điểm (Xanh: Tất cả, Đỏ: Inliers)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Hàm chính để thực hiện khớp ảnh
def match_image(scene_path, object_path, output_path='scene_with_detected_object.jpg', min_matches=10):
    # Bước 1: Tải ảnh
    scene_img, object_img = load_images(scene_path, object_path)
    
    # Bước 2: Phát hiện SIFT features
    keypoints1, descriptors1, keypoints2, descriptors2 = detect_sift_features(object_img, scene_img)
    
    # Bước 3: Khớp keypoints với Lowe's Ratio Test
    good_matches = match_keypoints(descriptors1, descriptors2)
    
    # Kiểm tra số lượng khớp tốt
    if len(good_matches) < min_matches:
        raise ValueError(f"Số cặp khớp tốt không đủ: {len(good_matches)} (yêu cầu: {min_matches})")
    
    # Bước 4: Tìm và vẽ khung bao dựa trên Homography
    scene_with_box, mask, scene_corners = find_and_draw_bounding_box(scene_img, object_img, good_matches, keypoints1, keypoints2)
    
    # Bước 5: Hiển thị các cặp điểm khớp
    draw_matches_with_status(scene_img, object_img, keypoints1, keypoints2, good_matches, mask)
    
    return scene_with_box

# Ví dụ sử dụng
if __name__ == "__main__":
    try:
        # Các đường dẫn đến ảnh
        scene_path = "task3_scene.jpg"
        object_path = "task3_object.jpg"
        output_path = "scene_with_detected_object.jpg"
        
        # Thực hiện khớp ảnh
        result = match_image(scene_path, object_path, output_path)
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")