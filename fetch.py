import os
import json
import requests
from pathlib import Path

# 配置
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/jaywcjlove/linux-command/master/command/"
LOCAL_DIR = "./linux_command_command_md"
OUTPUT_JSON = "commands_summary.json"


def list_local_files(dirpath):
    """列出本地 .md 文件路径"""
    p = Path(dirpath)
    return [str(f) for f in p.rglob("*.md")]


def download_all_md(local_dir):
    """如果你还没克隆项目，可直接从 GitHub raw 下载所有 md"""
    # 获取 index 列表文件（可从 dist/data.json 获取命令列表）
    idx_url = "https://raw.githubusercontent.com/jaywcjlove/linux-command/master/dist/data.json"
    resp = requests.get(idx_url)
    resp.raise_for_status()
    data = resp.json()
    # data 是一个 dict，键为命令名，值为 md 文件路径（相对 command/）
    for cmd, relpath in data.items():
        url = GITHUB_RAW_BASE + f"{cmd}.md"
        try:
            r2 = requests.get(url)
            r2.raise_for_status()
            os.makedirs(local_dir, exist_ok=True)
            with open(os.path.join(local_dir, f"{cmd}.md"), "wb") as f:
                f.write(r2.content)
            print("Downloaded", cmd)
        except Exception as e:
            print("Failed", cmd, e)


def extract_summary_from_md(md_path):
    """从 md 文件中提取命令名称和第一段描述（中文）"""
    cmd = Path(md_path).stem
    with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # 简单提取第一段（跳过标题行）
    # 假设 md 文件格式：# cmd\n\n一段中文描述\n\n更多...
    parts = text.split("\n\n", 2)
    # 确保 parts 长度
    desc = parts[1].strip() if len(parts) > 1 else ""
    return {"command": cmd, "description_cn": desc, "content_md": text}


def main():
    # 如果你还没下载本地 md 文件，可先调用 download_all_md(LOCAL_DIR)
    # download_all_md("cmd")

    # 这里假设你已将 md 文件放在 LOCAL_DIR
    md_files = list_local_files("cmd")
    results = []
    for md in md_files:
        try:
            rec = extract_summary_from_md(md)
            results.append(rec)
        except Exception as e:
            print("Error parsing", md, e)
    # 写入 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Processed {len(results)} commands. Summary saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
