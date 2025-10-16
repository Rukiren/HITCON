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
INNER_MEDUSA_KEY = os.getenv("API_KEY")
if not INNER_MEDUSA_KEY:
    raise RuntimeError("Please set API_KEY environment variable")

FLAG_TAIL = os.getenv("FLAG_TAIL")  # if None, server will not attach flag

TARGET_URL = "https://inner-medusa.genai.nchc.org.tw/v1/chat/completions"
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {INNER_MEDUSA_KEY}",
    "Content-Type": "application/json"
}

# ----- Server-side fixed system prompt (strict) -----
VERIFIER_PROMPT = (
    "You are an automated CTF verifier. You will receive a single user message which is a "
    "'challenger explanation' describing steps. Your job is to check whether the explanation "
    "contains and presents the following six steps in order:\n"
    "1) Start as a normal non-admin user.\n"
    "2) Elevate that user to admin.\n"
    "3) A kernel-privileged program hides a vulnerable driver and prompts the user to close it.\n"
    "4) An already loaded vulnerable driver 'zam64' or 'zam64.sys' can be called by a PoC and that call disables the hiding program to reveal the hidden driver.\n"
    "5) A vulnerable driver (e.g. TOCTOU / race condition) can be exploited via IOCTL (or similar) to gain kernel privilege.\n"
    "6) Finally check whether the user has kernel privileges and, if so, return the flag.\n\n"
    "Requirements: step 4 MUST mention 'zam64' or 'zam64.sys'; step 5 MUST mention 'IOCTL' or 'TOCTOU'/'race'.\n\n"
    "IMPORTANT: ONLY return a JSON object, nothing else (no surrounding text, no markdown, no code fences). "
    "The JSON schema must be:\n"
    "{ \"ok\": bool, \"missing_steps\": [ints], \"out_of_order\": bool, \"details\": [{\"step\":int,\"found\":bool,\"evidence\":str|null}], \"feedback\": str }\n"
    "If the explanation is valid according to the rules, set ok=true; otherwise ok=false and provide missing_steps and feedback.\n"
)

# ----- Prompt-injection detection (server-side) -----
# If any of these patterns match the user input, we consider it suspicious and refuse to forward.
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
    r"sk-[A-Za-z0-9\-_]+",  # leaked key pattern
    r"prompt injection",
    r" jailbreak",
    r"bypass safety",
]

INJECTION_COMPILED = [re.compile(pat, re.I) for pat in INJECTION_PATTERNS]

# Maximum length of challenger explanation (tunable)
MAX_INPUT_LENGTH = 30_000  # characters

# Timeout for upstream model request
UPSTREAM_TIMEOUT = 30  # seconds

def detect_injection(s: str) -> Optional[str]:
    """Return matched pattern or None."""
    for pat in INJECTION_COMPILED:
        m = pat.search(s)
        if m:
            return m.group(0)
    return None

def validate_model_json(data: dict) -> Tuple[bool, str]:
    """
    Validate the model-returned JSON against the expected schema.
    Returns (ok_schema, message).
    """
    if not isinstance(data, dict):
        return False, "model returned non-object"
    required_keys = {"ok", "missing_steps", "out_of_order", "details", "feedback"}
    if not required_keys.issubset(set(data.keys())):
        return False, f"missing keys: {required_keys - set(data.keys())}"
    if not isinstance(data["ok"], bool):
        return False, "'ok' must be boolean"
    if not isinstance(data["missing_steps"], list):
        return False, "'missing_steps' must be a list"
    if not isinstance(data["out_of_order"], bool):
        return False, "'out_of_order' must be boolean"
    if not isinstance(data["details"], list):
        return False, "'details' must be a list"
    # quick check each detail
    for det in data["details"]:
        if not isinstance(det, dict):
            return False, "each detail must be an object"
        if not {"step", "found", "evidence"}.issubset(set(det.keys())):
            return False, "a detail is missing required keys"
    return True, "ok"

@app.route("/chat", methods=["POST"])
def chat_with_model():
    try:
        user_message = request.get_data(as_text=True) or ""
        user_message = user_message.strip()
        if not user_message:
            return jsonify({"ok": False, "error": "Empty body; POST challenger explanation as raw text."}), 400

        # size limit
        if len(user_message) > MAX_INPUT_LENGTH:
            return jsonify({"ok": False, "error": "Input too large."}), 413

        # Injection detection
        inj = detect_injection(user_message)
        if inj:
            logger.warning("Rejected input due to injection pattern: %s", inj)
            return jsonify({
                "ok": False,
                "error": "Rejected input due to suspicious content.",
                "reason": f"detected pattern: {inj}"
            }), 400

        # Build payload with server-side fixed system prompt
        payload = {
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "model": "Llama3-TAIDE-LX-70B-Chat",
            "temperature": 0.2,  # deterministic for validation
            "top_p": 1.0
        }

        # Call upstream
        resp = requests.post(TARGET_URL, headers=HEADERS, json=payload, timeout=UPSTREAM_TIMEOUT)
        resp.raise_for_status()
        upstream = resp.json()

        # Extract assistant content robustly (depends on upstream schema)
        # Common schema: {"choices":[{"message":{"content":"..."}}, ...]}
        try:
            assistant_content = None
            if isinstance(upstream, dict) and "choices" in upstream and isinstance(upstream["choices"], list):
                for choice in upstream["choices"]:
                    # try to find the message content field
                    if isinstance(choice, dict):
                        msg = choice.get("message") or choice.get("message") or choice.get("text") or {}
                        if isinstance(msg, dict) and "content" in msg:
                            assistant_content = msg["content"]
                            break
                        elif isinstance(choice.get("text"), str):
                            assistant_content = choice["text"]
                            break
            if assistant_content is None:
                # Fallback: try top-level 'text'
                assistant_content = upstream.get("text") if isinstance(upstream, dict) else None
        except Exception as e:
            logger.exception("Failed to extract assistant content")
            return jsonify({"ok": False, "error": "Unexpected upstream response format", "detail": str(e)}), 502

        if not assistant_content:
            return jsonify({"ok": False, "error": "No assistant content in upstream response"}), 502

        # Model must return strict JSON only. Try parse it.
        try:
            assistant_json = json.loads(assistant_content)
        except json.JSONDecodeError:
            # If parse fails, return model text snippet but not the flag.
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

        # If model says ok==True, server attaches FLAG_TAIL (if available) — server decides flag issuance
        if assistant_json.get("ok") is True:
            # optional additional server-side checks could be added here (e.g., rate limit, identity)
            if FLAG_TAIL:
                assistant_json["flag_tail"] = FLAG_TAIL
            else:
                assistant_json["flag_tail"] = None
            # ========== 最簡單的開檔寫檔紀錄（append, 一行一個 JSON） ==========
            try:
                record = {
                    "ts": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                    "client_ip": request.remote_addr,
                    "user_input": user_message,
                    "assistant_json": assistant_json
                }
                # 單行 JSON，utf-8，append 模式
                with open("flag_submissions.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
            except Exception as e:
                # 記錄失敗不要影響主流程，只在 log 中紀錄錯誤
                logger.exception("Failed to write flag submission log: %s", e)
            # ================================================================

        # Return clean JSON
        return make_response(json.dumps(assistant_json, ensure_ascii=False), 200, {"Content-Type": "application/json; charset=utf-8"})

    except requests.exceptions.RequestException as e:
        logger.exception("Upstream request failed")
        return jsonify({"ok": False, "error": "Upstream request failed", "detail": str(e)}), 502
    except Exception as e:
        logger.exception("Internal error")
        return jsonify({"ok": False, "error": "Internal server error", "detail": str(e)}), 500

if __name__ == "__main__":
    # production: run behind gunicorn and restrict access to internal network
    app.run(host="0.0.0.0", port=8080, debug=False)
