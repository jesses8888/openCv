import os
import shutil
import cv2
from deepface import DeepFace


source_base_dir = "data"
target_emotions = ["sad", "angry", "happy"]
dir_mapping = {"sad": "Sad", "angry": "Angry", "happy": "Happy"}

integrated_dir = "face_data"
output_base_dir = "face_data_ok"

os.makedirs(integrated_dir, exist_ok=True)
os.makedirs(output_base_dir, exist_ok=True)

print("=== 階段一：開始整合資料夾影像 ===")

for emotion in target_emotions:
    folder_name = dir_mapping[emotion]
    dir_path = os.path.join(source_base_dir, folder_name)

    if not os.path.exists(dir_path):
        print(f"找不到資料夾: {dir_path}，跳過。")
        continue

    for filename in os.listdir(dir_path):
        src_file_path = os.path.join(dir_path, filename)

        if os.path.isfile(src_file_path):
            new_filename = f"{folder_name}_{filename}"
            dst_file_path = os.path.join(integrated_dir, new_filename)
            shutil.copy(src_file_path, dst_file_path)

print(f"資料整合完成！所有影像已複製到 '{integrated_dir}' 資料夾中。\n")

print("=== 階段二：開始進行臉部辨識與分類 ===")

for filename in os.listdir(integrated_dir):
    img_path = os.path.join(integrated_dir, filename)

    if not os.path.isfile(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"無法讀取影像: {filename}，跳過。")
        continue

    try:
        result = DeepFace.analyze(
            img_path=img,
            actions=["emotion"],
            enforce_detection=False
        )

        all_emotions_score = result[0]["emotion"]
        dominant_emotion = result[0]["dominant_emotion"]

        if dominant_emotion not in target_emotions:
            filtered_scores = {k: v for k, v in all_emotions_score.items() if k in target_emotions}
            dominant_emotion = max(filtered_scores, key=filtered_scores.get)

    except Exception as e:
        dominant_emotion = "happy"

    display_text = dominant_emotion.upper()
    cv2.putText(img, display_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    final_folder_name = dir_mapping[dominant_emotion]
    emotion_target_dir = os.path.join(output_base_dir, final_folder_name)
    os.makedirs(emotion_target_dir, exist_ok=True)

    save_path = os.path.join(emotion_target_dir, f"result_{filename}")
    cv2.imwrite(save_path, img)
    print(f"影像 {filename} 辨識結果為 [{final_folder_name}] -> 已存至 {save_path}")

print("\n=== 所有影像辨識與分類程序結束 ===")