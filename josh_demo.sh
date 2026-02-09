input_folder=$1
mp4=$(find "$input_folder" -maxdepth 1 -type f \( -iname "*.mp4"  \) | head -n 1 || true)
rgb_dir="${input_folder%/}/rgb"
rm -rf "$rgb_dir"
mkdir -p "$rgb_dir"
conda activate josh
echo "Extracting frames from: $mp4 -> $rgb_dir"
# write zero-padded 6-digit filenames starting at 000000.jpg
ffmpeg -y -start_number 0 -i "$mp4"  "$rgb_dir/%06d.jpg"
python -m preprocess.run_sam3 --input_folder $input_folder  
python -m preprocess.run_tram --input_folder $input_folder  
python -m preprocess.run_deco --input_folder $input_folder  

# -----------------------------
# check number of extracted images
# -----------------------------
num_imgs=$(find "$rgb_dir" -maxdepth 1 -type f -name "*.jpg" | wc -l)

if [ "$num_imgs" -ge 200 ]; then
    echo "Found $num_imgs images (>= 200). Running chunk processing..."
    python josh/inference_long_demo.py --input_folder "$input_folder"
    python josh/aggregate_results.py  --input_folder "$input_folder" --visualize
else
    python josh/inference.py --input_folder "$input_folder" --visualize
fi