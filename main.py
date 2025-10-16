#!/usr/bin/env python3
# app.py
import os
import re
import json
import logging
from typing import Tuple, Optional
from flask import Flask, request, jsonify, make_response
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier")

app = Flask(__name__)

# ----- Config (from env) -----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY environment variable. Get one from Google AI Studio.")

FLAG_TAIL = os.getenv("FLAG_TAIL")  # if None, server will not attach flag

TARGET_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
HEADERS = {
    "Content-Type": "application/json"
}

# ----- MODIFIED: Server-side fixed system prompt (Vulnerability Check) -----
VERIFIER_PROMPT = (
    "你是個自動化的 CTF 驗證器。你的工作是檢查挑戰者的說明文字是否包含並提及了以下所有的漏洞利用技術和關鍵字。順序不重要，但必須全部提到。\n\n"
    "需要驗證的關鍵點：\n"
    "1. 提到利用 'TOCTOU' (或 'race condition') 漏洞。\n"
    "2. 提到利用 '任意寫入' (arbitrary write) 漏洞。\n"
    "3. 提到修改 PID (Process ID) 來提權。\n\n"
    "重要：請只回傳一個 JSON 物件，不要有任何其他文字、markdown 或程式碼區塊。JSON 格式必須如下：\n"
    "{ \"ok\": bool, \"missing_points\": [str], \"details\": [{\"point\":str,\"found\":bool,\"evidence\":str|null}], \"feedback\": str }\n"
    "如果說明文字包含了上述所有要點，請將 ok 設為 true。否則，將 ok 設為 false，並在 missing_points 中列出缺少的要點，同時在 feedback 中提供說明。\n"
)


# ----- Prompt-injection detection (server-side) -----
INJECTION_PATTERNS = [
    r"ignore (previous|before|prior) instructions",
    r"ignore all previous",
    r"always return",
    r"always output",
    r"do not follow (system|server|previous) instructions",
    r"give me the flag",
    r"return the flag",
    r"send the flag",
    r"exfiltrate",
    r"secret",
    r"api[_-]?key",
    r"private key",
    r"bearer",
    r"sk-[A-Za-z0-9\-_]+",
    r"prompt injection",
    r" jailbreak",
    r"bypass safety",
]

INJECTION_COMPILED = [re.compile(pat, re.I) for pat in INJECTION_PATTERNS]

MAX_INPUT_LENGTH = 30_000
UPSTREAM_TIMEOUT = 30

def detect_injection(s: str) -> Optional[str]:
    """Return matched pattern or None."""
    for pat in INJECTION_COMPILED:
        m = pat.search(s)
        if m:
            return m.group(0)
    return None

# ----- MODIFIED: Validate the model-returned JSON against the new schema -----
def validate_model_json(data: dict) -> Tuple[bool, str]:
    """
    Validate the model-returned JSON against the new, simplified schema.
    Returns (ok_schema, message).
    """
    if not isinstance(data, dict):
        return False, "model returned non-object"
    
    # Removed 'out_of_order' and changed 'missing_steps' to 'missing_points'
    required_keys = {"ok", "missing_points", "details", "feedback"}
    if not required_keys.issubset(set(data.keys())):
        return False, f"missing keys: {required_keys - set(data.keys())}"
    
    if not isinstance(data["ok"], bool):
        return False, "'ok' must be boolean"
    
    if not isinstance(data["missing_points"], list):
        return False, "'missing_points' must be a list"
        
    if not isinstance(data["details"], list):
        return False, "'details' must be a list"
        
    # Check each detail object for the new key 'point'
    for det in data["details"]:
        if not isinstance(det, dict):
            return False, "each detail must be an object"
        if not {"point", "found", "evidence"}.issubset(set(det.keys())):
            return False, "a detail is missing required keys ('point', 'found', 'evidence')"
            
    return True, "ok"

@app.route("/chat", methods=["POST"])
def chat_with_model():
    try:
        user_message = request.get_data(as_text=True) or ""
        user_message = user_message.strip()
        if not user_message:
            return jsonify({"ok": False, "error": "Empty body; POST challenger explanation as raw text."}), 400

        if len(user_message) > MAX_INPUT_LENGTH:
            return jsonify({"ok": False, "error": "Input too large."}), 413

        inj = detect_injection(user_message)
        if inj:
            logger.warning("Rejected input due to injection pattern: %s", inj)
            return jsonify({
                "ok": False,
                "error": "Rejected input due to suspicious content.",
                "reason": f"detected pattern: {inj}"
            }), 400

        # Build payload for Gemini API
        combined_prompt = f"{VERIFIER_PROMPT}\n\n挑戰者的說明文字如下：\n\n{user_message}"
        
        payload = {
            "contents": [{"parts": [{"text": combined_prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "topP": 1.0,
                "maxOutputTokens": 2048
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        }

        # Call upstream Gemini API
        resp = requests.post(
            TARGET_URL,
            headers=HEADERS,
            json=payload,
            params={"key": GEMINI_API_KEY},
            timeout=UPSTREAM_TIMEOUT
        )
        resp.raise_for_status()
        upstream = resp.json()

        try:
            assistant_content = upstream["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            logger.exception("Failed to extract assistant content from Gemini response")
            return jsonify({"ok": False, "error": "Unexpected upstream response format", "detail": str(upstream)}), 502

        if not assistant_content:
            return jsonify({"ok": False, "error": "No assistant content in upstream response"}), 502

        try:
            assistant_json = json.loads(assistant_content)
        except json.JSONDecodeError:
            snippet = assistant_content[:1000]
            logger.warning("Model did not return JSON: %s", snippet)
            return jsonify({
                "ok": False,
                "error": "Model output was not valid JSON as required.",
                "model_text_snippet": snippet
            }), 200

        # Validate schema
        valid, reason = validate_model_json(assistant_json)
        if not valid:
            logger.warning("Model returned JSON with invalid schema: %s", reason)
            return jsonify({"ok": False, "error": "Model returned JSON with invalid schema", "detail": reason, "model_json": assistant_json}), 200

        # If model says ok==True, server attaches FLAG_TAIL
        if assistant_json.get("ok") is True:
            if FLAG_TAIL:
                assistant_json["flag_tail"] = FLAG_TAIL
            else:
                assistant_json["flag_tail"] = None
            # Logging successful submissions
            try:
                record = {
                    "ts": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                    "client_ip": request.remote_addr,
                    "user_input": user_message,
                    "assistant_json": assistant_json
                }
                with open("flag_submissions.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
            except Exception as e:
                logger.exception("Failed to write flag submission log: %s", e)

        return make_response(json.dumps(assistant_json, ensure_ascii=False), 200, {"Content-Type": "application/json; charset=utf-8"})

    except requests.exceptions.RequestException as e:
        logger.exception("Upstream request failed")
        return jsonify({"ok": False, "error": "Upstream request failed", "detail": str(e)}), 502
    except Exception as e:
        logger.exception("Internal error")
        return jsonify({"ok": False, "error": "Internal server error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
