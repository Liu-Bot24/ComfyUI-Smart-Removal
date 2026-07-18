from __future__ import annotations

import argparse
import sys

import argostranslate.package
import argostranslate.translate


def has_zh_en_translation() -> bool:
    languages = argostranslate.translate.get_installed_languages()
    chinese = next((item for item in languages if item.code == "zh"), None)
    english = next((item for item in languages if item.code == "en"), None)
    if chinese is None or english is None:
        return False
    try:
        chinese.get_translation(english)
    except Exception:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="安装或检查 Argos 中文→英文语言模型。"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查是否已安装，不联网、不修改环境。",
    )
    args = parser.parse_args()

    if has_zh_en_translation():
        print("Argos 中文→英文语言模型已经安装，无需重复安装。")
        return 0

    if args.check:
        print("Argos 中文→英文语言模型尚未安装。", file=sys.stderr)
        return 1

    print("正在从 Argos 官方语言包索引查找中文→英文模型……")
    argostranslate.package.update_package_index()
    candidates = [
        item
        for item in argostranslate.package.get_available_packages()
        if item.from_code == "zh" and item.to_code == "en"
    ]
    if not candidates:
        print("未在 Argos 官方索引中找到中文→英文语言包。", file=sys.stderr)
        return 1

    package = candidates[0]
    print(f"正在下载 {package}……")
    download_path = package.download()
    argostranslate.package.install_from_path(download_path)

    if not has_zh_en_translation():
        print("语言包装入后仍无法建立中文→英文翻译。", file=sys.stderr)
        return 1

    print("Argos 中文→英文语言模型安装完成。请重启 ComfyUI。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
