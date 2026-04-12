from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v12/DFS.yaml")
    data = r"ultralytics/cfg/datasets/HITUAV.yaml"
    model.train(
        data = data,
        epochs = 10,
        imgsz = 640,
        batch = 4,
        optimizer='AdamW'
    )

    eval_name = "DFS" 
    predict_name = "DFS" 
    best_model_path = YOLO(r"runs/detect/train/weights/best.pt")
    test_image_path = r"datasets/HITUAV/images/test"

    results = best_model_path.predict(
        source = test_image_path,
        save = True,
        show = False, 
        line_width = 1, 
        show_labels = True, 
        show_conf = True, 
        project = "runs/detect",
        name = predict_name
    )
    print(f"\n 可视化已完成!")
    print(f" 结果已保存至: runs/detect/{predict_name}/\n")

    metrics = best_model_path.val(
        data = data,
        save_json = False, 
        project = "runs/detect",
        name = eval_name
    )
    print(f"\n 评估结果已完成!")
    print(f" 评估结果已保存至: runs/detect/{eval_name}/")
