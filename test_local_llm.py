from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


BASE_URL = "http://localhost:20128/v1"
MODEL = "cx/gpt-5.4"


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in environment or .env")
        return 1

    url = BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Tra loi ngan gon bang tieng Viet: 2 + 2 bang may?"},
        ],
        "temperature": 0,
        "max_tokens": 80,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {body}")
        return 1
    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}")
        return 1

    choices = data.get("choices") or []
    answer = ""
    if choices:
        answer = ((choices[0].get("message") or {}).get("content") or "").strip()

    print(f"OK url={url}")
    print(f"OK model={MODEL}")
    print(answer or json.dumps(data, ensure_ascii=False)[:1000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
