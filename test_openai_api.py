from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def config_model(path: str = "solution/config.json") -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("model") or "gpt-5.4-nano"
    except Exception:
        return "gpt-5.4-nano"


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in environment/.env")
        return 1

    model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or config_model()
    base_url = (
        os.getenv("LOCAL_BASE_URL")
        or os.getenv("LLM_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "http://localhost:20128/v1"
    )
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Tra loi ngan gon: 2 + 2 bang bao nhieu?"}
        ],
        "max_tokens": 50,
        "temperature": 0,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "observathon-api-test/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {body}")
        return 1
    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}")
        return 1

    text = ""
    choices = data.get("choices") or []
    if choices:
        text = ((choices[0].get("message") or {}).get("content") or "").strip()

    print(f"OK model={model} url={url}")
    print(text or json.dumps(data, ensure_ascii=False)[:1000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
