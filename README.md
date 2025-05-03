---
title: Chinese Chess Recognition
emoji: 🌖
colorFrom: yellow
colorTo: gray
pinned: false
license: mit
short_description: 中国象棋识别，棋谱识别
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 中国象棋棋盘检测与棋子识别

## 功能说明
本项目实现了中国象棋棋盘的自动检测和棋子识别功能，包括：
- 棋盘关键点检测
- 棋盘透视变换
- 棋子类别识别

## 环境要求
- Python 3.7+
- opencv-python
- onnxruntime

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法
### 命令行运行
```bash
# 基本用法
python main.py --image 图片路径

# 指定模型路径
python main.py --image 图片路径 --pose_model 自定义关键点模型路径 --classifier_model 自定义分类模型路径
```

### 参数说明
- `--image`：必需参数，指定要检测的图片路径
- `--pose_model`：可选参数，关键点检测模型路径，默认为 `onnx/pose/4_v6-0301.onnx`
- `--classifier_model`：可选参数，棋子识别模型路径，默认为 `onnx/layout_recognition/nano_v3-0319.onnx`

### 输出说明
- 控制台输出检测结果，包括：
  - 总耗时
  - 棋盘布局（用中文显示棋子名称）
- 在 `output` 目录下保存处理后的图片：
  - `original_with_keypoints.jpg`：带关键点的原图
  - `transformed_board.jpg`：变换后的棋盘图

## 模型说明
- 关键点检测模型：`onnx/pose/4_v6-0301.onnx`
- 棋子识别模型：`onnx/layout_recognition/nano_v3-0319.onnx`

## 注意事项
- 确保输入图片清晰可见
- 建议图片分辨率在 1024x1024 以内
- 模型支持识别遮挡和空点
