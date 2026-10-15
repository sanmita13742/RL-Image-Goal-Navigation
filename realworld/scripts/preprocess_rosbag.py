import sys
import bisect
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    print("Error: 'rosbags' library is required. Install via: pip install rosbags")
    sys.exit(1)

def main():
    bag_dir = Path(r"D:\MINav_Dataset_20260921_154717")
    out_dir = Path(r"D:\MINav_Dataset_20260921_154717_processed")
    
    if not bag_dir.exists():
        print(f"Error: Rosbag directory not found: {bag_dir}")
        sys.exit(1)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "data.csv"
    
    typestore = get_typestore(Stores.ROS2_FOXY)
    
    print(f"Reading rosbag: {bag_dir}")
    print("Pass 1: Extracting all /cmd_vel messages into RAM...")
    
    cmd_times = []
    cmd_data = []
    
    try:
        with AnyReader([bag_dir], default_typestore=typestore) as reader:
            cmd_conns = [c for c in reader.connections if c.topic == '/cmd_vel']
            if not cmd_conns:
                print("Error: No /cmd_vel topic found in rosbag!")
                sys.exit(1)
                
            for connection, timestamp, rawdata in reader.messages(connections=cmd_conns):
                msg = reader.deserialize(rawdata, connection.msgtype)
                cmd_times.append(timestamp)  # timestamp is strictly in nanoseconds
                cmd_data.append((msg.linear.x, msg.linear.y, msg.angular.z))
                
            print(f"Loaded {len(cmd_times)} /cmd_vel commands.")
            
            print("Pass 2: Streaming /camera/color/image_raw and associating commands...")
            img_conns = [c for c in reader.connections if c.topic == '/camera/color/image_raw']
            if not img_conns:
                print("Error: No /camera/color/image_raw topic found in rosbag!")
                sys.exit(1)
                
            last_saved_time_ns = 0
            # 10 Hz maximum sampling rate = 100,000,000 ns
            sample_interval_ns = 100_000_000
            
            saved_count = 0
            skipped_no_cmd = 0
            total_images_seen = 0
            
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["timestamp", "image", "vx", "vy", "wz"])
            
            start_time_real = time.time()
            
            for connection, timestamp, rawdata in reader.messages(connections=img_conns):
                total_images_seen += 1
                
                # Check if it has been 100ms since the last saved image
                if (timestamp - last_saved_time_ns) >= sample_interval_ns:
                    # Find active command: latest command with cmd_time <= img_time
                    idx = bisect.bisect_right(cmd_times, timestamp) - 1
                    
                    if idx < 0:
                        skipped_no_cmd += 1
                        continue
                    
                    vx, vy, wz = cmd_data[idx]
                    
                    # Deserialize image
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    
                    # Ensure rgb8 format
                    if msg.encoding != "rgb8":
                        print(f"Warning: Unexpected encoding {msg.encoding} at timestamp {timestamp}")
                        
                    # Create numpy array
                    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    
                    # Save JPEG
                    img_filename = f"{saved_count:06d}.jpg"
                    img_path = images_dir / img_filename
                    Image.fromarray(frame).save(img_path, format="JPEG", quality=90)
                    
                    # Write CSV row
                    csv_writer.writerow([timestamp, f"images/{img_filename}", f"{vx:.6f}", f"{vy:.6f}", f"{wz:.6f}"])
                    
                    saved_count += 1
                    last_saved_time_ns = timestamp
                    
                # Progress logging
                if total_images_seen % 5000 == 0:
                    elapsed = time.time() - start_time_real
                    print(f"  Processed {total_images_seen} images | Saved {saved_count} | Elapsed: {elapsed:.1f}s")
                    
            csv_file.close()
            
            print("=" * 50)
            print("Conversion Complete! Running Validation...")
            print("=" * 50)
            
            # Validation
            saved_jpegs = sorted(list(images_dir.glob("*.jpg")))
            print(f"Number of JPEGs on disk : {len(saved_jpegs)}")
            
            with open(csv_path, 'r') as f:
                rows = list(csv.reader(f))
            header = rows[0]
            data_rows = rows[1:]
            print(f"Number of CSV data rows : {len(data_rows)}")
            
            if len(saved_jpegs) != len(data_rows):
                print("❌ ERROR: Mismatch between number of JPEGs and CSV rows!")
            else:
                print("✅ JPEG count matches CSV rows.")
                
            missing_files = 0
            duplicate_ts = 0
            mono_increasing = True
            
            prev_ts = -1
            
            for i, row in enumerate(data_rows):
                ts = int(row[0])
                img_path = out_dir / row[1]
                
                if not img_path.exists():
                    missing_files += 1
                    
                if ts <= prev_ts:
                    if ts == prev_ts:
                        duplicate_ts += 1
                    else:
                        mono_increasing = False
                prev_ts = ts
                
            if missing_files > 0:
                print(f"❌ ERROR: {missing_files} images specified in CSV are missing on disk!")
            else:
                print("✅ All images in CSV exist on disk.")
                
            if not mono_increasing:
                print("❌ ERROR: Timestamps are not monotonically increasing!")
            else:
                print("✅ Timestamps are strictly monotonically increasing.")
                
            if duplicate_ts > 0:
                print(f"❌ ERROR: {duplicate_ts} duplicate timestamps found!")
            else:
                print("✅ No duplicate timestamps.")
                
            # Compute actual sampling interval
            if len(data_rows) > 1:
                first_ts = int(data_rows[0][0])
                last_ts = int(data_rows[-1][0])
                duration_s = (last_ts - first_ts) / 1e9
                actual_rate = (len(data_rows) - 1) / duration_s
                print(f"Actual sampling rate    : {actual_rate:.2f} Hz")
            
            print(f"Skipped (no prior cmd)  : {skipped_no_cmd} images")
            
            # Get size
            total_size_bytes = sum(f.stat().st_size for f in images_dir.glob("*.jpg"))
            total_size_bytes += csv_path.stat().st_size
            print(f"Total processed size    : {total_size_bytes / (1024*1024):.2f} MB")
            print("=" * 50)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fatal Error: {e}")

if __name__ == "__main__":
    main()
