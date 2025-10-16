import os
import requests
from flask import Flask, request, jsonify

API_KEY = os.environ.get('API_KEY') 
API_URL = 'https://inner-medusa.genai.nchc.org.tw/v1/chat/completions'

# 初始化 Flask 應用
app = Flask(__name__)

# --- API 路由 ---
@app.route('/chat', methods=['POST'])
def chat_with_model():
    """
    接收使用者的文字，與 AI 模型對話，並回傳模型的答覆。
    """
    # 1. 檢查進來的請求是否為 JSON 格式
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # 2. 從收到的 JSON 中取出 'message' 欄位
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "Missing 'message' field in request"}), 400

    # 3. 準備要發送到 AI 模型 API 的標頭 (Headers)
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    # 4. 準備要發送到 AI 模型 API 的資料 (Payload)
    #    將使用者的訊息放進去
    payload = {
        "max_tokens": 1024,
        "messages": [
            {
                "content": "You are a helpful assistant. 你是一個樂於助人的助手。",
                "role": "system"
            },
            {
                "content": user_message, # 使用者傳來的訊息放在這裡
                "role": "user"
            }
        ],
        "model": "Llama3-TAIDE-LX-70B-Chat",
        "temperature": 0.2,
        "top_p": 0.92
    }

    try:
        # 5. 發送 POST 請求到真正的 AI API
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()  # 如果 API 回傳錯誤 (如 4xx, 5xx)，會在此拋出例外

        # 6. 解析 AI API 的回傳結果
        api_response_data = response.json()
        
        # 7. 從複雜的 JSON 結構中，取出我們真正需要的文字內容
        #    根據您的範例，路徑是 ['choices'][0]['message']['content']
        model_reply = api_response_data['choices'][0]['message']['content']

        # 8. 將乾淨的文字回傳給使用者
        return jsonify({"reply": model_reply})

    except requests.exceptions.RequestException as e:
        # 處理網路連線或 API 的錯誤
        return jsonify({"error": f"API request failed: {e}"}), 500
    except (KeyError, IndexError) as e:
        # 處理 API 回傳的 JSON 格式不符合預期的問題
        return jsonify({"error": f"Failed to parse API response: {e}"}), 500


# --- 啟動伺服器 ---
if __name__ == '__main__':
    # 讓伺服器監聽在所有網路介面 (0.0.0.0) 的 5000 port
    # debug=True 讓你在修改程式碼後不用重啟伺服器，但正式環境建議關閉
    app.run(host='0.0.0.0', port=5000, debug=True)
