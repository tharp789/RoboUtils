import os
import subprocess
from pathlib import Path

def merge_and_filter_instance(dir1: str, dir2: str, timestamp: str, output_dir: str):
    out_file = os.path.join(output_dir, f"{timestamp}.mcap")

    # Collect .mcap files
    mcaps1 = []
    mcaps2 = []
    if os.path.exists(dir1):
        mcaps1.extend(sorted(os.listdir(dir1)))
    full_mcaps1 = [os.path.join(dir1, f) for f in mcaps1]
    if os.path.exists(dir2):
        mcaps2.extend(sorted(os.listdir(dir2)))
    full_mcaps2 = [os.path.join(dir2, f) for f in mcaps2]

    if not mcaps1 or not mcaps2:
        print(f"[WARN] No mcaps found for {timestamp}")
        return

    # Merge into a temp file
    full_mcaps = full_mcaps1 + full_mcaps2
    merged_tmp = os.path.join(output_dir, f"{timestamp}_tmp.mcap")
    merge_cmd = ["mcap", "merge", "--allow-duplicate-metadata","-o", str(merged_tmp)] + [str(f) for f in full_mcaps]
    try:
        subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Command failed:", e.cmd)
        print("Exit code:", e.returncode)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)  # usually where the error message is


    # Filter into final file
    filter_cmd = ["mcap", "filter", "-o", str(out_file), str(merged_tmp)]
    subprocess.run(filter_cmd, check=True)
    os.remove(merged_tmp)

    print(f"[OK] Wrote {out_file}")

def merge_and_filter_mcap(parent_dir, divided_data_dir, prepend1, prepend2, output_dir):
    split_data_dir = os.path.join(parent_dir, divided_data_dir)
    # find a list of folders in split_data_dir that start with prepend1 and prepend2
    files1 = [d for d in os.listdir(split_data_dir) if d.startswith(prepend1)]
    files2 = [d for d in os.listdir(split_data_dir) if d.startswith(prepend2)]
    timestamps2 = [tuple(d.split("_")[-2:]) for d in files2]
    timestamps1 = [tuple(d.split("_")[-2:]) for d in files1]

    # Convert to sets to find common timestamps
    set1 = set(timestamps1)
    set2 = set(timestamps2)
    common = sorted(set1 & set2)
    output_dir = os.path.join(parent_dir, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"[INFO] Found {len(common)} matching timestamps: {common}")
    for ts in common:
        ts_str = "_".join(ts)  # Convert tuple back to string
        merge_and_filter_instance(os.path.join(split_data_dir, prepend1) + "_" + ts_str, os.path.join(split_data_dir, prepend2) + "_" + ts_str, ts_str, output_dir)

def main():
    parent_dir = "/home/tyler/Documents/MSR/field_tests/250828_afca/"
    divided_data_dir = "split_data"
    prepend1 = "tracking_data"
    prepend2 = "wire_data"
    output_dir = "final_data"
    merge_and_filter_mcap(parent_dir, divided_data_dir, prepend1, prepend2, output_dir)


if __name__ == "__main__":
    main()
