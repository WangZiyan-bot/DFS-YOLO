from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("ultralytics/cfg/models/v12/DFS.yaml")
    data = r"ultralytics/cfg/datasets/HITUAV.yaml"
    model.train(
        data = data,
        epochs = 100,
        imgsz = 640,
        batch = 4,
        optimizer='AdamW'
    )

    # ------------------开始测试---------------------------------
    eval_name = "DFS"     # 评估结果文件夹名
    predict_name = "DFS"  # 可视化结果文件夹名
    best_model_path = YOLO(r"runs/detect/train/weights/best.pt")
    test_image_path = r"datasets/HITUAV/images/test"

    results = best_model_path.predict(
        source = test_image_path,
        save = True,             # 保存检测结果图片
        show = False,            # 不弹窗显示
        line_width = 1,          # 边框线宽
        show_labels = True,      # 显示类别标签
        show_conf = True,        # 显示置信度分数
        project = "runs/detect",
        name = predict_name
    )
    print(f"\n 可视化已完成!")
    print(f" 结果已保存至: runs/detect/{predict_name}/\n")

    # ------------------开始验证---------------------------------
    metrics = best_model_path.val(
        data = data,
        split = 'test',          # 指定使用测试集
        save_json = False,       # 是否保存COCO格式 (可选)
        project = "runs/detect",
        name = eval_name
    )
    print(f"\n 评估结果已完成!")
    print(f" 评估结果已保存至: runs/detect/{eval_name}/")