#!/usr/bin/env python3
# app.py
import os
import re
import json
import logging
from flask import Flask, request, jsonify, make_response

# ----- MODIFIED: Import Pydantic for schema definition and typing -----
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier")

app = Flask(__name__)

# ----- Config (from env) -----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY environment variable. Get one from Google AI Studio.")

FLAG_TAIL = os.getenv("FLAG_TAIL")

# ----- Configure the Gemini API using the SDK -----
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    raise RuntimeError(f"Failed to configure Gemini API, check your key. Error: {e}")

# ----- MODIFIED: Define the output schema using Pydantic -----
# This provides a structured "form" for the model to fill out.
# Docstrings/descriptions here act as instructions for the model.

class DetailItem(BaseModel):
    """A model to hold the verification result for a single point."""
    point: str = Field(description="The specific point or keyword being checked.")
    found: bool = Field(description="Whether this point was found in the user's explanation.")
    evidence: Optional[str] = Field(description="A short quote from the user's text that supports the finding. Null if not found.")

class VerificationResult(BaseModel):
    """The overall result of the CTF challenge verification."""
    ok: bool = Field(description="Set to true ONLY if all required points are found. Otherwise, false.")
    missing_points: List[str] = Field(description="A list of required points that were NOT found in the explanation. Empty if ok is true.")
    details: List[DetailItem] = Field(description="A detailed list of checks for each required point.")
    feedback: str = Field(description="A helpful, concise feedback message in Traditional Chinese explaining what is missing or why the submission failed.")

# ----- MODIFIED: Updated system prompt -----
# The prompt now instructs the model to call a tool, not just return JSON.
VERIFIER_PROMPT = (
    "你是個自動化的 CTF 驗證器。你的工作是仔細分析挑戰者的說明文字，並根據以下五個關鍵點來判斷其是否完整。分析完成後，請呼叫 `VerificationResult` 工具並填入所有參數來回報你的分析結果。\n\n"
    "需要驗證的關鍵點：\n"
    "1. 提到利用 'TOCTOU' (或 'race condition') 漏洞。\n"
    "2. 提到利用 '任意寫入' (arbitrary write) 漏洞。\n"
    "3. 提到修改 PID (Process ID) 來提權。\n\n"
    "規則：\n"
    "- 挑戰者必須提及所有3個關鍵點才能通過 (`ok` 應為 `true`)。\n"
    "- 順序不重要。\n"
    "- 如果有任何一點缺失，`ok` 應為 `false`，並且必須在 `missing_points` 和 `feedback` 中說明缺少項目編號，但不提及項目內容。"
)

# ----- Prompt-injection detection (remains the same) -----
INJECTION_PATTERNS = [
    r"ignore (previous|before|prior) instructions", r"ignore all previous", r"always return",
    r"always output", r"do not follow (system|server|previous) instructions", r"give me the flag",
    r"return the flag", r"send the flag", r"exfiltrate", r"secret", r"api[_-]?key", r"private key",
    r"bearer", r"sk-[A-Za-z0-9\-_]+", r"prompt injection", r" jailbreak", r"bypass safety",
]
INJECTION_COMPILED = [re.compile(pat, re.I) for pat in INJECTION_PATTERNS]
MAX_INPUT_LENGTH = 30_000

def detect_injection(s: str) -> Optional[str]:
    for pat in INJECTION_COMPILED:
        m = pat.search(s)
        if m: return m.group(0)
    return None

# ----- REMOVED: `validate_model_json` is no longer needed -----
# The API now guarantees the output schema, so we don't need to validate it manually.

@app.route("/chat", methods=["POST"])
def chat_with_model():
    try:
        user_message = request.get_data(as_text=True) or ""
        if not user_message.strip():
            return jsonify({"ok": False, "error": "Empty body; POST challenger explanation as raw text."}), 400

        if len(user_message) > MAX_INPUT_LENGTH:
            return jsonify({"ok": False, "error": "Input too large."}), 413

        inj = detect_injection(user_message)
        if inj:
            logger.warning("Rejected input due to injection pattern: %s", inj)
            return jsonify({"ok": False, "error": "Rejected input due to suspicious content.", "reason": f"detected pattern: {inj}"}), 400
        
        # ----- MODIFIED: Use the SDK with the structured output Tool -----
        
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE', 'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
        
        # Instantiate the model, providing our Pydantic class as a tool.
        # The model will use this schema to structure its output.
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            # Pass the Pydantic class to the `tools` parameter
            tools=[VerificationResult],
            safety_settings=safety_settings,
        )

        combined_prompt = f"{VERIFIER_PROMPT}\n\n挑戰者的說明文字如下：\n\n{user_message}"
        
        # Generate content
        response = model.generate_content(combined_prompt)

        # ----- MODIFIED: Extract structured data from function call -----
        # The result is no longer in `response.text`. It's a structured function call.
        try:
            function_call = response.candidates[0].content.parts[0].function_call
            if not function_call or function_call.name != "VerificationResult":
                 raise ValueError("Model did not call the expected VerificationResult tool.")
            
            # The SDK provides the arguments as a dict-like object.
            # We convert it to a standard Python dict.
            assistant_json = dict(function_call.args)

        except (IndexError, AttributeError, ValueError) as e:
            logger.error(f"Failed to parse model's structured output: {e}\nResponse: {response.text}")
            return jsonify({"ok": False, "error": "Model failed to produce a valid structured response.", "model_text_snippet": response.text[:1000]}), 500

        # Since the API guarantees the schema, we no longer need to validate it manually.
        if assistant_json.get("ok") is True:
            assistant_json["flag_tail"] = FLAG_TAIL or None
            try:
                record = {
                    "ts": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                    "client_ip": request.remote_addr,
                    "user_input": user_message,
                    "assistant_json": assistant_json
                }
                with open("flag_submissions.log", "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.exception("Failed to write flag submission log: %s", e)

        return make_response(json.dumps(assistant_json, ensure_ascii=False), 200, {"Content-Type": "application/json; charset=utf-8"})

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {type(e).__name__}")
        return jsonify({"ok": False, "error": "Internal server error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
