import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from core.chessboard_detector import ChessboardDetector

# 棋子类别映射
dict_cate_names_reverse = {
    'K': '红帅',
    'A': '红士',
    'B': '红相',
    'N': '红马',
    'R': '红车',
    'C': '红炮',
    'P': '红兵',
    'k': '黑将',
    'a': '黑仕',
    'b': '黑象',
    'n': '黑傌',
    'r': '黑車',
    'c': '黑砲',
    'p': '黑卒',
    '.': '空点',
    'x': '遮挡'
}

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='中国象棋棋盘检测与棋子识别')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--pose_model', type=str, default='onnx/pose/4_v6-0301.onnx', help='关键点检测模型路径')
    parser.add_argument('--classifier_model', type=str, default='onnx/layout_recognition/nano_v3-0319.onnx', help='棋子识别模型路径')
    args = parser.parse_args()

    # 检查输入图片是否存在
    if not Path(args.image).exists():
        print(f"错误：图片 {args.image} 不存在")
        return

    # 检查模型文件是否存在
    if not Path(args.pose_model).exists():
        print(f"错误：关键点检测模型 {args.pose_model} 不存在")
        return
    
    if not Path(args.classifier_model).exists():
        print(f"错误：棋子识别模型 {args.classifier_model} 不存在")
        return

    # 初始化检测器
    detector = ChessboardDetector(
        pose_model_path=args.pose_model,
        full_classifier_model_path=args.classifier_model
    )

    # 读取图片
    image = cv2.imread(args.image)
    if image is None:
        print(f"错误：无法读取图片 {args.image}")
        return

    # 记录开始时间
    start_time = time.time()

    try:
        # 执行检测
        original_image_with_keypoints, transformed_image, cells_labels, scores, time_info = detector.pred_detect_board_and_classifier(image)

        if cells_labels is None:
            print("检测失败：无法识别棋盘布局")
            return

        # 处理换行符
        if isinstance(cells_labels, str):
            cells_labels = [list(row) for row in cells_labels.split('\n') if row.strip()]
        
        # 转换为中文显示
        annotation_arr_10_9_chinese = []
        for row in cells_labels:
            chinese_row = []
            for item in row:
                if item in dict_cate_names_reverse:
                    chinese_row.append(dict_cate_names_reverse[item])
                else:
                    chinese_row.append('未知')
            annotation_arr_10_9_chinese.append(chinese_row)

        # 计算总耗时
        total_time = time.time() - start_time

        # 打印结果
        print("\n=== 检测结果 ===")
        print(f"总耗时: {total_time:.2f}秒")
        print("\n棋盘布局:")
        for row in annotation_arr_10_9_chinese:
            print(" ".join(row))

        # 保存结果图片
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # 保存带关键点的原图
        cv2.imwrite(str(output_dir / "original_with_keypoints.jpg"), original_image_with_keypoints)
        # 保存变换后的棋盘图
        cv2.imwrite(str(output_dir / "transformed_board.jpg"), transformed_image)
        
        print(f"\n结果图片已保存到 {output_dir} 目录")

    except Exception as e:
        print(f"检测失败: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 