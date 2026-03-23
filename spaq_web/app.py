"""
SPAQ Web 后端服务
提供静态页面渲染和图像评分 API
"""

import os
import uuid  # 【新增】引入 UUID 库生成唯一文件名
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from inference import SPAQPredictor

app = Flask(__name__)

# 配置上传文件夹和允许的文件类型
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 限制最大上传大小为 32MB

# 确保上传目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 提前初始化预测引擎（单例），避免首次请求时加载过慢
predictor = SPAQPredictor()

def allowed_file(filename):
    """检查文件扩展名是否合法"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """接收图片并返回质量分数的 API 端点"""
    # 1. 检查请求中是否包含文件
    if 'image' not in request.files:
        return jsonify({'error': '未找到图片文件'}), 400
        
    file = request.files['image']
    
    # 2. 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
        
    # 3. 验证并保存文件
    if file and allowed_file(file.filename):
        # 【修改】获取文件后缀，并生成全局唯一的 UUID 文件名
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            # 此时并发请求 A 和 B 会保存为互不干扰的独立文件
            file.save(filepath)
            
            # 调用推理引擎，接收字典返回
            result = predictor.predict(filepath)
            
            # 【核心清理】评估完成后，立即删除磁盘上的临时文件，杜绝空间泄漏
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'success': True,
                'score': result['score'],
                'exif': result['exif'], # 将 EXIF 透传给前端
                'message': '评估完成'
            }), 200
            
        except Exception as e:
            # 【安全兜底】如果预测过程中发生异常崩溃，也要确保删掉临时文件
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'处理图片时发生错误: {str(e)}'}), 500
            
    return jsonify({'error': '不支持的文件格式，仅支持 JPG/JPEG/PNG'}), 400

if __name__ == '__main__':
    # 启动 Flask 服务，开启 debug 模式便于开发调试
    app.run(host='0.0.0.0', port=5000, debug=True)