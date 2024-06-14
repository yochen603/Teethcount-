from flask import Flask, request, jsonify
import os
import requests
from loadapi import xyplane, get_final_prediction
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

@app.route('/estimate', methods=['POST'])
def estimate_api():
    # 初始化开始时间
    start_time = time.time()
    url = request.form.get('url')
    filename = url.split('/')[-1]
    os.makedirs('temp', exist_ok=True)
    filepath = os.path.join('temp', filename)

    try:
        # 发送HTTP GET请求
        response = requests.get(url, stream=True)
        # 检查请求是否成功
        response.raise_for_status()

        # 保存文件
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        # 计算下载并保存文件所需的时间
        elapsed_time = time.time() - start_time
        # 处理图像和预测
        input_image = xyplane(filepath)
        predicted_label,load_time,predict_time = get_final_prediction(filepath, input_image)
        return jsonify({'filename': filename, 'teeth_count': predicted_label,
        'download_time': round(elapsed_time, 2),
        'load_time': round(load_time, 2),
        'predict_time': round(predict_time, 2),
         }) # 四舍五入至两位小数})

    except requests.exceptions.HTTPError as http_err:
        # 处理HTTP错误
        return jsonify({'error': 'HTTP error occurred: ' + str(http_err)}), 500
    except requests.exceptions.RequestException as req_err:
        # 处理请求错误
        return jsonify({'error': 'Request error occurred: ' + str(req_err)}), 500
    except Exception as e:
        # 处理其他异常
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)
