import cv2
import mediapipe as mp
import math
import numpy as np
import time # Untuk mengukur waktu kalibrasi
from collections import deque # Untuk buffer nilai EAR 1 detik

# --- Fungsi Bantuan ---

# Fungsi untuk menghitung jarak Euclidean antara dua titik
def euclidean_distance(point1, point2):
    """
    Menghitung jarak Euclidean antara dua titik (x1, y1) dan (x2, y2).
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Fungsi untuk menghitung Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_landmarks_coords):
    """
    Menghitung Eye Aspect Ratio (EAR) berdasarkan koordinat landmark mata.
    eye_landmarks_coords diharapkan berisi 6 titik (P0 hingga P5)
    sesuai urutan umum perhitungan EAR.
    """
    # Mengasumsikan eye_landmarks_coords memiliki format:
    # [P0, P1, P2, P3, P4, P5]
    # P0 = (x_horizontal_kiri, y_horizontal_kiri)
    # P1 = (x_vertikal_atas_1, y_vertikal_atas_1)
    # P2 = (x_vertikal_atas_2, y_vertikal_atas_2)
    # P3 = (x_horizontal_kanan, y_horizontal_kanan)
    # P4 = (x_vertikal_bawah_2, y_vertikal_bawah_2)
    # P5 = (x_vertikal_bawah_1, y_vertikal_bawah_1)

    A = euclidean_distance(eye_landmarks_coords[1], eye_landmarks_coords[5]) # Jarak P2-P6 dari diagram EAR umum
    B = euclidean_distance(eye_landmarks_coords[2], eye_landmarks_coords[4]) # Jarak P3-P5 dari diagram EAR umum
    C = euclidean_distance(eye_landmarks_coords[0], eye_landmarks_coords[3]) # Jarak P1-P4 dari diagram EAR umum
    
    # Menghindari pembagian oleh nol jika C sangat kecil
    if C == 0:
        return 0.0 # Atau nilai yang sangat besar, tergantung penanganan
    
    ear = (A + B) / (2.0 * C)
    return ear

# --- Inisialisasi MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- Indeks Landmark Mata untuk MediaPipe ---
LEFT_EYE_IDXS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398] 
RIGHT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

LEFT_EYE_EAR_IDXS = [362, 386, 387, 398, 374, 373] 
RIGHT_EYE_EAR_IDXS = [33, 159, 158, 133, 145, 153]

# --- Thresholds dan Counter Deteksi Kantuk ---
EAR_CONSEC_FRAMES = 30 # Jumlah frame berturut-turut di mana EAR di bawah threshold untuk dianggap mengantuk.

COUNTER = 0         
DROWSY = False      

# --- Variabel untuk Kalibrasi EAR Dinamis ---
CALIBRATION_DURATION = 60 # Durasi kalibrasi dalam detik
calibration_start_time = time.time()
ear_values_for_calibration = [] # Menyimpan nilai EAR untuk periode kalibrasi 10 detik penuh
is_calibrating = True
dynamic_ear_threshold = None
DROWSINESS_FACTOR = 0.95 # Faktor untuk menentukan threshold kantuk dari EAR rata-rata (misal: 90% dari EAR saat mata terbuka)

# --- Variabel untuk EAR Averaging (1 detik) ---
# Menyimpan tuple (ear_value, timestamp) untuk rata-rata 1 detik terakhir
ear_buffer_1sec = deque() 
BUFFER_DURATION = 1.0 # Durasi buffer dalam detik (1 detik)

# --- BAGIAN BARU: Variabel untuk Deteksi Kedipan ---
BLINK_COUNTER = 0
EYE_CLOSED = False # Status untuk menandai apakah mata sedang dalam keadaan tertutup (untuk menghitung kedipan)
# --- AKHIR BAGIAN BARU ---

print("Memulai program deteksi kantuk dengan kalibrasi EAR dinamis dan visualisasi landmark.")
print(f"Kalibrasi akan berlangsung selama {CALIBRATION_DURATION} detik. Harap jaga mata tetap terbuka.")
print("Tekan 'ESC' untuk keluar.")

# --- Buka Webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak dapat membuka aliran video dari kamera.")
    exit()

# Flag untuk mengelola visibilitas jendela kalibrasi
calibration_window_open = False

# --- Loop Utama Deteksi ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Mengabaikan frame kamera yang kosong.")
        continue

    image = cv2.flip(image, 1) # Balik gambar secara horizontal
    h, w, _ = image.shape      # Dapatkan tinggi dan lebar gambar

    # MediaPipe membutuhkan input RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Tingkatkan kinerja
    
    results = face_mesh.process(image_rgb)

    # Izinkan penulisan pada gambar untuk OpenCV
    image_rgb.flags.writeable = True 
    
    frame_ear = 0.0 # EAR untuk frame saat ini (nilai mentah, sebelum rata-rata 1 detik)
    ear_calculated_in_frame_successfully = False # Flag untuk menandai apakah EAR berhasil dihitung di frame ini

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_ear_coords = []
            right_eye_ear_coords = []
            left_eye_all_coords = []
            right_eye_all_coords = []

            # Ekstrak koordinat landmark mata yang relevan untuk EAR
            for idx in LEFT_EYE_EAR_IDXS:
                lm = face_landmarks.landmark[idx]
                left_eye_ear_coords.append((int(lm.x * w), int(lm.y * h)))
            
            for idx in RIGHT_EYE_EAR_IDXS:
                lm = face_landmarks.landmark[idx]
                right_eye_ear_coords.append((int(lm.x * w), int(lm.y * h)))

            # Ekstrak semua koordinat landmark mata untuk visualisasi
            for idx in LEFT_EYE_IDXS:
                lm = face_landmarks.landmark[idx]
                left_eye_all_coords.append((int(lm.x * w), int(lm.y * h)))
            
            for idx in RIGHT_EYE_IDXS:
                lm = face_landmarks.landmark[idx]
                right_eye_all_coords.append((int(lm.x * w), int(lm.y * h)))

            # Hitung EAR untuk kedua mata
            left_ear = eye_aspect_ratio(left_eye_ear_coords)
            right_ear = eye_aspect_ratio(right_eye_ear_coords)

            frame_ear = (left_ear + right_ear) / 2.0
            ear_calculated_in_frame_successfully = True

            # --- Visualisasi Landmark Mata (di gambar RGB asli) ---
            # Gambar titik-titik landmark mata
            for (x_pt, y_pt) in left_eye_all_coords:
                cv2.circle(image, (x_pt, y_pt), 1, (0, 255, 255), -1) # Kuning
            for (x_pt, y_pt) in right_eye_all_coords:
                cv2.circle(image, (x_pt, y_pt), 1, (0, 255, 255), -1) # Kuning
            
            # Gambar kontur mata
            if len(left_eye_all_coords) > 0:
                left_eye_hull = cv2.convexHull(np.array(left_eye_all_coords))
                cv2.drawContours(image, [left_eye_hull], -1, (0, 255, 0), 1) # Hijau
            
            if len(right_eye_all_coords) > 0:
                right_eye_hull = cv2.convexHull(np.array(right_eye_all_coords))
                cv2.drawContours(image, [right_eye_hull], -1, (0, 255, 0), 1) # Hijau

    # --- Update Buffer EAR 1 Detik ---
    # Hanya tambahkan nilai EAR ke buffer jika berhasil dihitung di frame ini
    if ear_calculated_in_frame_successfully:
        ear_buffer_1sec.append((frame_ear, time.time()))
    
    # Hapus nilai EAR yang lebih lama dari 1 detik
    while ear_buffer_1sec and (time.time() - ear_buffer_1sec[0][1] > BUFFER_DURATION):
        ear_buffer_1sec.popleft()
    
    # Hitung 'current_ear' sebagai rata-rata dari buffer 1 detik
    averaged_ear_1sec = 0.0
    if len(ear_buffer_1sec) > 0:
        averaged_ear_1sec = np.mean([item[0] for item in ear_buffer_1sec])
    
    current_ear = averaged_ear_1sec # Ini adalah nilai EAR yang akan digunakan untuk semua logika

    # --- Logika Kalibrasi dan Deteksi Kantuk ---
    if is_calibrating:
        # Buat frame terpisah untuk tampilan kalibrasi
        calibration_display_frame = image.copy() # Gunakan salinan frame saat ini untuk konteks visual

        # Hanya tambahkan ke data kalibrasi jika EAR 1-detik rata-rata non-zero
        if current_ear > 0.0: # Filter out 0.0 EARs which often mean no face or error
            ear_values_for_calibration.append(current_ear) 
        
        elapsed_time = time.time() - calibration_start_time
        
        # Tampilkan progres kalibrasi di jendela baru
        cv2.putText(calibration_display_frame, f"KALIBRASI: {int(elapsed_time)}/{CALIBRATION_DURATION}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(calibration_display_frame, "HARAP JAGA MATA TETAP TERBUKA", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Tampilkan EAR rata-rata 1 detik saat ini
        cv2.putText(calibration_display_frame, f"Current EAR (1s Avg): {current_ear:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Tampilkan rata-rata keseluruhan EAR selama kalibrasi
        if len(ear_values_for_calibration) > 0:
            current_avg_ear_calib = np.mean(ear_values_for_calibration)
            cv2.putText(calibration_display_frame, f"Avg. EAR (Calib): {current_avg_ear_calib:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            cv2.putText(calibration_display_frame, "Avg. EAR (Calib): N/A", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow('EAR Calibration Progress', calibration_display_frame)
        calibration_window_open = True # Tandai jendela kalibrasi sebagai terbuka
        
        if elapsed_time >= CALIBRATION_DURATION:
            if len(ear_values_for_calibration) > 0:
                average_ear = np.mean(ear_values_for_calibration)
                dynamic_ear_threshold = average_ear * DROWSINESS_FACTOR
                is_calibrating = False
                print(f"Kalibrasi selesai. Rata-rata EAR: {average_ear:.2f}")
                print(f"Threshold Deteksi Kantuk Ditetapkan: {dynamic_ear_threshold:.2f}")
            else:
                print("Tidak ada nilai EAR yang terkumpul selama kalibrasi. Menggunakan threshold default (0.25).")
                dynamic_ear_threshold = 0.25 # Fallback default
                is_calibrating = False
            
            # Tutup jendela kalibrasi setelah selesai
            if calibration_window_open:
                cv2.destroyWindow('EAR Calibration Progress')
                calibration_window_open = False

    else: # Setelah kalibrasi selesai, lakukan deteksi
        # Pastikan jendela kalibrasi tertutup jika entah bagaimana masih terbuka
        if calibration_window_open:
            cv2.destroyWindow('EAR Calibration Progress')
            calibration_window_open = False
            
        if dynamic_ear_threshold is not None:
            # --- Logika Deteksi Kantuk ---
            if current_ear < dynamic_ear_threshold:
                COUNTER += 1 
                if COUNTER >= EAR_CONSEC_FRAMES:
                    if not DROWSY:
                        DROWSY = True
                    cv2.putText(image, "STATUS: MENGANTUK", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                DROWSY = False
                cv2.putText(image, "STATUS: NORMAL", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- BAGIAN BARU: Logika Deteksi Kedipan Mata ---
            # Jika EAR turun di bawah threshold DAN mata sebelumnya terbuka
            if current_ear < dynamic_ear_threshold and not EYE_CLOSED:
                EYE_CLOSED = True # Tandai mata sebagai tertutup

            # Jika EAR naik kembali di atas threshold DAN mata sebelumnya tertutup
            if current_ear >= dynamic_ear_threshold and EYE_CLOSED:
                BLINK_COUNTER += 1 # Tambah hitungan kedipan
                EYE_CLOSED = False # Reset status mata menjadi terbuka
            # --- AKHIR BAGIAN BARU ---

        else: 
            cv2.putText(image, "ERROR: THRESHOLD TIDAK ADA", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    # Teks status jika tidak ada wajah terdeteksi (di main window)
    if not results.multi_face_landmarks:
        COUNTER = 0
        DROWSY = False
        if is_calibrating:
             cv2.putText(image, f"KALIBRASI: {int(time.time() - calibration_start_time)}/{CALIBRATION_DURATION}s - MENUNGGU WAJAH", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(image, "STATUS: MENUNGGU WAJAH", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    # Tampilkan EAR (rata-rata 1 detik) dan Threshold di jendela utama
    cv2.putText(image, f"EAR (1s Avg): {current_ear:.2f}", (w - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if not is_calibrating and dynamic_ear_threshold is not None:
        cv2.putText(image, f"Threshold: {dynamic_ear_threshold:.2f}", (w - 220, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # --- BAGIAN BARU: Tampilkan jumlah kedipan ---
    cv2.putText(image, f"Kedipan: {BLINK_COUNTER}", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    # --- AKHIR BAGIAN BARU ---

    # --- Konversi ke Grayscale dan Tampilkan Jendela Utama ---
    gray_image_for_display = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Drowsiness Detection (Grayscale Output)', gray_image_for_display)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()