import os
import numpy as np
from skimage.feature import local_binary_pattern  # Thư viện để trích xuất đặc trưng LBPH
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # Thuật toán Random Forest cho phân loại
from sklearn.preprocessing import StandardScaler  # Công cụ chuẩn hóa dữ liệu
from sklearn.metrics import accuracy_score, classification_report  # Đánh giá mô hình
import kagglehub

# Hàm tải dữ liệu từ Kaggle
def load_fer2013_data():
    """
    Tải bộ dữ liệu FER2013 từ Kaggle sử dụng kagglehub.
    - dataset_path: Đường dẫn đến thư mục chứa dữ liệu sau khi tải.
    - train_path: Đường dẫn đến thư mục 'train' chứa ảnh huấn luyện.
    - test_path: Đường dẫn đến thư mục 'test' chứa ảnh kiểm tra.
    - Trả về: Tuple (train_path, test_path) để sử dụng sau này.
    """
    dataset_path = kagglehub.dataset_download("msambare/fer2013")  # Tải dataset FER2013 từ Kaggle
    train_path = os.path.join(dataset_path, "train")  # Lấy đường dẫn đến thư mục train
    test_path = os.path.join(dataset_path, "test")  # Lấy đường dẫn đến thư mục test
    return train_path, test_path

# Hàm tiền xử lý ảnh
def preprocess_image(img):
    """
    Tiền xử lý ảnh để chuẩn bị cho trích xuất đặc trưng.
    - img: Ảnh đầu vào (có thể là ảnh màu 3 kênh hoặc ảnh xám 1 kênh).
    - gray: Ảnh xám sau khi xử lý.
    - Phát hiện khuôn mặt bằng Haar Cascade và cắt vùng khuôn mặt, sau đó resize về kích thước 48x48.
    - Trả về: Ảnh xám đã xử lý.
    """
    # Kiểm tra số kênh của ảnh: 3 kênh (màu) hoặc 1 kênh (xám)
    if len(img.shape) == 3:  # Nếu là ảnh màu (BGR với 3 kênh)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển đổi từ BGR sang ảnh xám
    else:  # Nếu đã là ảnh xám (1 kênh)
        gray = img  # Sử dụng trực tiếp ảnh xám
    
    # Phát hiện khuôn mặt trong ảnh xám
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Tải bộ phân loại Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  # Phát hiện khuôn mặt, scaleFactor=1.3 giảm kích thước ảnh, minNeighbors=5 để lọc nhiễu
    if len(faces) > 0:  # Nếu phát hiện được ít nhất một khuôn mặt
        x, y, w, h = faces[0]  # Lấy tọa độ và kích thước của khuôn mặt đầu tiên
        gray = gray[y:y+h, x:x+w]  # Cắt vùng khuôn mặt
        gray = cv2.resize(gray, (48, 48))  # Resize về kích thước 48x48, phù hợp với FER2013
    return gray

# Hàm trích xuất đặc trưng LBPH
def extract_lbph_features(image, P=16, R=2, method='uniform', num_bins=256):
    """
    Trích xuất đặc trưng LBPH từ ảnh.
    - image: Ảnh đầu vào (thường là ảnh xám).
    - P: Số điểm lân cận (default=16), xác định số điểm xung quanh pixel trung tâm để so sánh.
    - R: Bán kính (default=2), khoảng cách từ pixel trung tâm đến các điểm lân cận.
    - method: Phương pháp tính LBPH (default='uniform'), giảm số lượng mẫu bằng cách sử dụng các mẫu đồng nhất.
    - num_bins: Số bin cho histogram (default=256), xác định độ chi tiết của histogram.
    - Trả về: Vector histogram LBPH đã chuẩn hóa bằng chuẩn L1.
    """
    # Kiểm tra số kênh và chuyển đổi sang ảnh xám nếu cần
    if len(image.shape) == 3:  # Nếu ảnh là màu (3 kênh)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển đổi sang ảnh xám
    
    # Tiền xử lý ảnh để chuẩn bị cho trích xuất đặc trưng
    image = preprocess_image(image)  # Áp dụng phát hiện khuôn mặt và resize
    
    # Trích xuất LBPH
    lbp = local_binary_pattern(image, P=P, R=R, method=method)  # Tạo bản đồ LBPH, P và R tăng độ chi tiết không gian
    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins), density=True)  # Tính histogram của LBPH, density=True để chuẩn hóa
    hist = hist / np.sum(hist)  # Chuẩn hóa histogram bằng chuẩn L1 để đảm bảo tổng bằng 1
    return hist

# Hàm xử lý dữ liệu
def process_dataset(data_path, P=16, R=2, method='uniform', num_bins=256, limit_per_class=2000):
    """
    Xử lý dữ liệu từ thư mục, trích xuất đặc trưng và nhãn.
    - data_path: Đường dẫn đến thư mục train hoặc test.
    - P, R, method, num_bins: Tham số cho trích xuất LBPH.
    - limit_per_class: Giới hạn số lượng ảnh mỗi lớp (default=2000) để tối ưu thời gian.
    - X: Danh sách vector đặc trưng LBPH.
    - y: Danh sách nhãn tương ứng.
    - Trả về: Tuple (X, y) dưới dạng mảng numpy.
    """
    X, y = [], []
    # Định nghĩa ánh xạ từ tên biểu cảm sang nhãn số
    emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    print(f"Các thư mục trong {data_path}: {os.listdir(data_path)}")  # In danh sách thư mục để kiểm tra cấu trúc
    for emotion in os.listdir(data_path):  # Duyệt qua các thư mục con
        emotion_lower = emotion.lower()  # Chuẩn hóa tên thư mục về chữ thường để tránh lỗi viết hoa
        emotion_dir = os.path.join(data_path, emotion)  # Tạo đường dẫn đến thư mục biểu cảm
        if os.path.isdir(emotion_dir) and emotion_lower in emotion_labels:  # Kiểm tra thư mục hợp lệ
            count = 0
            for img_name in os.listdir(emotion_dir):  # Duyệt qua các ảnh trong thư mục
                if count >= limit_per_class:  # Giới hạn số ảnh nếu vượt quá
                    break
                img_path = os.path.join(emotion_dir, img_name)  # Tạo đường dẫn đến ảnh
                img = cv2.imread(img_path)  # Đọc ảnh
                if img is not None:  # Kiểm tra ảnh có đọc được không
                    features = extract_lbph_features(img, P, R, method, num_bins)  # Trích xuất đặc trưng LBPH
                    X.append(features)  # Thêm đặc trưng vào danh sách
                    y.append(emotion_labels[emotion_lower])  # Thêm nhãn tương ứng
                    count += 1
        else:
            print(f"Bỏ qua thư mục không xác định: {emotion}")  # Báo cáo nếu thư mục không hợp lệ
    return np.array(X), np.array(y)  # Chuyển danh sách thành mảng numpy

# Hàm huấn luyện và đánh giá
def train_and_evaluate_model():
    """
    Huấn luyện mô hình Random Forest và đánh giá trên tập test.
    - Tải dữ liệu, trích xuất đặc trưng, chuẩn hóa, huấn luyện, và đánh giá.
    - Hiển thị độ chính xác, báo cáo phân loại, và một vài dự đoán.
    - Trả về: Tuple (accuracy, report) để tham khảo.
    """
    train_path, test_path = load_fer2013_data()  # Tải dữ liệu train và test
    print("Đang xử lý dữ liệu train...")  # Thông báo tiến trình
    X_train, y_train = process_dataset(train_path, limit_per_class=2000)  # Xử lý tập train
    print("Đang xử lý dữ liệu test...")  # Thông báo tiến trình
    X_test, y_test = process_dataset(test_path, limit_per_class=500)  # Xử lý tập test
    
    # Chuẩn hóa đặc trưng LBPH để đảm bảo giá trị đồng đều
    scaler = StandardScaler()  # Khởi tạo công cụ chuẩn hóa
    X_train_scaled = scaler.fit_transform(X_train)  # Chuẩn hóa tập train và học tham số
    X_test_scaled = scaler.transform(X_test)  # Chuẩn hóa tập test dựa trên tham số từ train
    
    # Huấn luyện mô hình Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,  # Số cây quyết định trong rừng, tăng để cải thiện độ chính xác
        max_depth=20,  # Độ sâu tối đa của mỗi cây, giới hạn để tránh overfitting
        min_samples_split=5,  # Số mẫu tối thiểu để phân chia node, kiểm soát độ phức tạp
        random_state=42  # Khởi tạo ngẫu nhiên cố định để tái hiện kết quả
    )
    rf_model.fit(X_train_scaled, y_train)  # Huấn luyện mô hình
    
    # Đánh giá mô hình
    y_pred = rf_model.predict(X_test_scaled)  # Dự đoán trên tập test
    accuracy = accuracy_score(y_test, y_pred)  # Tính độ chính xác
    report = classification_report(y_test, y_pred, target_names=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])  # Báo cáo chi tiết
    
    print(f"Độ chính xác: {accuracy:.4f}")  # In độ chính xác với 4 chữ số thập phân
    print("Báo cáo phân loại:")  # In tiêu đề báo cáo
    print(report)  # In chi tiết precision, recall, f1-score cho từng lớp
    
    # Hiển thị một vài dự đoán
    emotion_labels_reverse = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}  # Ánh xạ từ số sang tên biểu cảm
    for i in range(min(5, len(X_test))):  # Lặp tối đa 5 ảnh để hiển thị
        emotion_dir = os.path.join(test_path, emotion_labels_reverse[y_test[i]])  # Lấy đường dẫn đến thư mục biểu cảm thực tế
        img_files = os.listdir(emotion_dir)  # Lấy danh sách ảnh trong thư mục
        if img_files:  # Kiểm tra thư mục có ảnh không
            test_img_path = os.path.join(emotion_dir, img_files[0])  # Lấy đường dẫn ảnh đầu tiên
            test_img = cv2.imread(test_img_path)  # Đọc ảnh
            if test_img is not None:  # Kiểm tra ảnh có đọc được không
                pred_label = emotion_labels_reverse[y_pred[i]]  # Nhãn dự đoán
                true_label = emotion_labels_reverse[y_test[i]]  # Nhãn thực tế
                plt.figure(figsize=(5, 5))  # Tạo figure với kích thước 5x5 inch
                plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))  # Hiển thị ảnh, chuyển từ BGR sang RGB
                plt.title(f'Dự đoán: {pred_label}, Thực tế: {true_label}')  # Thêm tiêu đề với nhãn
                plt.axis('off')  # Ẩn trục tọa độ
                plt.show()  # Hiển thị ảnh
    
    return accuracy, report  # Trả về độ chính xác và báo cáo để tham khảo

if __name__ == "__main__":
    """
    Chạy chương trình chính.
    - Gọi hàm train_and_evaluate_model trong khối try-except để bắt lỗi.
    - In lỗi nếu xảy ra để hỗ trợ gỡ lỗi.
    """
    try:
        accuracy, report = train_and_evaluate_model()
    except Exception as e:
        print(f"Lỗi: {str(e)}")  # In lỗi chi tiết nếu có