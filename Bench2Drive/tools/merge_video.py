import subprocess
import os
import json

def scale_and_merge_videos(input_video_1, input_video_2, output_video):
    # Step 1: Scale the first video (1600x900 -> 910x512)
    scaled_video_1 = folder+"scaled_video_1.mp4"
    scale_command = [
        'ffmpeg', 
        '-i', input_video_1, 
        '-vf', 'scale=512:288', 
        '-c:a', 'copy',  # Keep audio if available, or remove if not needed
        scaled_video_1
    ]
    
    # Execute the scaling command
    subprocess.run(scale_command, check=True)

    # Step 2: Merge scaled video with second video (512x512) using vstack (vertical stacking)
    merged_video = output_video
    merge_command = [
        'ffmpeg',
        '-i', scaled_video_1,
        '-i', input_video_2,
        '-filter_complex', '[0][1]vstack=inputs=2',  # Vertical stacking of two videos
        '-c:v', 'libx264', 
        '-preset', 'fast', 
        '-crf', '23',  # Adjust quality (lower is better quality)
        '-c:a', 'copy',  # Audio stream copy (or you can discard if not needed)
        merged_video
    ]
    # Execute the merging command
    subprocess.run(merge_command, check=True)

    # final_scaled_video = "final_scaled_video.mp4"
    # scale_final_command = [
    #     'ffmpeg',
    #     '-i', merged_video,
    #     '-vf', 'scale=1600:900',  # Scale the video to 80% of its original width and height
    #     '-c:v', 'libx264',
    #     '-preset', 'fast',
    #     '-crf', '23',
    #     '-c:a', 'copy',  # Keep the original audio
    #     final_scaled_video
    # ]
    
    # subprocess.run(scale_final_command, check=True)

    # Optionally, remove the temporary scaled video file
    subprocess.run(['rm', scaled_video_1], check=True)
    # subprocess.run(['rm', merged_video], check=True)

    print(f"Successfully created the merged video: {output_video}")

# Example Usage
images_folder = '/home/ke/PAD/exp/b2d_result/eval_vis_s16_h02_d7/'
file_path=images_folder+'eval.json'
with open(file_path) as file:
    data = json.load(file)
records = data['_checkpoint']['records']

for folder in os.listdir(images_folder):
    case_folder=images_folder+folder
    if os.path.isdir(case_folder):
        route_id=folder.split('_')[2]

        for record in records:
            if route_id == record['route_id'].split('_')[1]:
                score=record['scores']['score_composed']

        if score==100:
            print(folder)
            input_video_1 = case_folder+".mp4"
            input_video_2 = case_folder+"bev.mp4"
            output_video = folder+".mp4"  # Path for the output merged video

            scale_and_merge_videos(input_video_1, input_video_2, output_video)
