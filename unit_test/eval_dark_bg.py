import os
import sys
import json
import time
import base64
import shutil
import argparse
import urllib.request


def check_dark_bg(image_path, base_url, timeout=30):
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    url = base_url + '/dark_bg_check'
    input_dict = {
        "recordID": "eval-" + os.path.basename(image_path),
        "sessionID": "eval-session",
        "msgID": "eval-msg",
        "imgData": img_b64,
        "imgType": "JPG"
    }
    json_data = json.dumps(input_dict).encode('utf-8')

    request = urllib.request.Request(url=url)
    request.add_header("Content-Type", "application/json")
    request.add_header("Content-Length", len(json_data))
    res = urllib.request.urlopen(url=request, data=json_data, timeout=timeout)
    result = json.load(res)

    is_dark_bg = result.get('is_dark_bg', False)
    dark_score = result.get('dark_score', 0)
    bg_ratio = result.get('bg_ratio', 0)
    resp_code = result.get('resp_code', -1)
    return is_dark_bg, dark_score, bg_ratio, resp_code


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate dark background detection')
    parser.add_argument('--input-dir', required=True, help='Source image directory (folder a)')
    parser.add_argument('--output-dir', required=True, help='Output root directory (a1/a2/a3 will be created here)')
    parser.add_argument('--base-url', default='http://localhost:37709', help='Server base URL')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--log-file', default=None, help='Save results to a txt file')
    args = parser.parse_args()

    if args.log_file:
        log_f = open(args.log_file, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_f)

    input_dir = args.input_dir
    base_url = args.base_url.rstrip('/')
    dir_dark = os.path.join(args.output_dir, 'a1')
    dir_normal = os.path.join(args.output_dir, 'a2')

    dir_noface = os.path.join(args.output_dir, 'a3')
    os.makedirs(dir_dark, exist_ok=True)
    os.makedirs(dir_normal, exist_ok=True)
    os.makedirs(dir_noface, exist_ok=True)

    jpg_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ])
    total = len(jpg_files)
    print(f"Found {total} JPG images in {input_dir}\n")

    dark_count = 0
    normal_count = 0
    noface_count = 0
    failed = []

    for idx, fname in enumerate(jpg_files, 1):
        src_path = os.path.join(input_dir, fname)

        try:
            is_dark_bg, dark_score, bg_ratio, resp_code = check_dark_bg(
                src_path, base_url, timeout=args.timeout
            )
        except Exception as e:
            print(f"[{idx}/{total}] {fname} FAILED: {e}")
            failed.append(fname)
            continue

        if resp_code == 300:
            dst_dir = dir_noface
            noface_count += 1
            print(f"[{idx}/{total}] {fname} NOFACE | dark_score={dark_score:.4f} bg_ratio={bg_ratio:.4f}")
        elif is_dark_bg:
            dst_dir = dir_dark
            dark_count += 1
            print(f"[{idx}/{total}] {fname} DARK   | dark_score={dark_score:.4f} bg_ratio={bg_ratio:.4f}")
        else:
            dst_dir = dir_normal
            normal_count += 1

        shutil.copy2(src_path, os.path.join(dst_dir, fname))

        if idx % 50 == 0 or idx == total:
            print(f"  >>> [{idx}/{total}] DARK={dark_count} NORMAL={normal_count} NOFACE={noface_count} FAIL={len(failed)}")

        time.sleep(0.01)

    print(f"\n{'='*50}")
    print(f"Total: {total}")
    print(f"Dark background (a1): {dark_count}")
    print(f"Normal (a2):        {normal_count}")
    print(f"No face (a3):        {noface_count}")
    print(f"Failed:              {len(failed)}")
    if failed:
        print(f"\nFailed files:")
        for f in failed:
            print(f"  - {f}")


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


if __name__ == '__main__':
    main()
