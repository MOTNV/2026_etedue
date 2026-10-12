"""Wikimedia Commons 사진 내려받기 (출처·라이선스 기록 포함)."""
import json
import re
import time
import urllib.parse
import urllib.request

from common import DATA

RAW = DATA / "raw"
UA = {"User-Agent": "scanandgo-edu/0.1 (student project)"}


def fetch(url):
    for i in range(6):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception:
            time.sleep(5 * (i + 1))  # 429(요청 과다)이면 점점 길게 쉰다
    raise RuntimeError(f"다운로드 실패: {url}")


def download(sources, width=960):
    """{저장이름: Commons 제목} -> data/raw/<저장이름>.jpg, 출처는 data/raw/credits.json에 누적"""
    RAW.mkdir(parents=True, exist_ok=True)
    meta_file = RAW / "credits.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    for key, title in sources.items():
        if (RAW / f"{key}.jpg").exists() and key in meta:
            continue
        q = urllib.parse.quote("File:" + title)
        api = ("https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&format=json"
               f"&iiprop=url|extmetadata&iiurlwidth={width}&titles={q}")
        info = list(json.loads(fetch(api))["query"]["pages"].values())[0]["imageinfo"][0]
        m = info["extmetadata"]
        (RAW / f"{key}.jpg").write_bytes(fetch(info["thumburl"]))
        meta[key] = {
            "title": title,
            "author": re.sub("<[^>]+>", "", m.get("Artist", {}).get("value", "?")).strip(),
            "license": m.get("LicenseShortName", {}).get("value", "?"),
            "url": info["descriptionurl"],
        }
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("다운로드:", title, flush=True)
        time.sleep(3)
    return meta


def write_credits(meta, path):
    lines = ["# 사진 출처", "",
             "실습용 사진은 모두 Wikimedia Commons의 자유 라이선스 이미지를 잘라 쓰거나 합성했습니다.",
             "`data/samples/`, `data/train/` 이미지와 결과 이미지들은 아래 사진을 합성·가공한 2차 저작물이며 원 라이선스를 따릅니다.", "",
             "| 파일 | 원본 | 저작자 | 라이선스 |", "|---|---|---|---|"]
    for key, m in sorted(meta.items()):
        lines.append(f"| `{key}.jpg` | [{m['title']}]({m['url']}) | {m['author']} | {m['license']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
