# python outdreamer/sample/sample_outpaint.py \
#     --origin_video "examples/val_beach/video_val_beach.txt" \
#     --model_path /path/to/checkpoint-xxx/model \
#     --num_frames 29 \
#     --height 480 \
#     --width 640 \
#     --cache_dir "./cache_dir" \
#     --text_encoder_name google/mt5-xxl \
#     --text_prompt "examples/val_beach/prompt_val_beach.txt" \
#     --ae CausalVAEModel_D4_4x8x8 \
#     --ae_path "/path/to/causalvideovae" \
#     --save_video_path "./results/beach-output.mp4" \
#     --guidance_scale 3 \
#     --num_sampling_steps 100 \
#     --enable_tiling \
#     --max_sequence_length 512 \
#     --sample_method EulerAncestralDiscrete \
#     --given_frame_num 3 \
#     --mask_ratio_l 0.33 \
#     --mask_ratio_r 0.33 \
#     --mask_ratio_u 0.0 \
#     --mask_ratio_d 0.0

python outdreamer/sample/sample_outpaint.py \
    --origin_video "examples/val_road_long/video_val_road.txt" \
    --model_path /path/to/checkpoint-xxx/model \
    --num_frames 29 \
    --height 480 \
    --width 640 \
    --cache_dir "./cache_dir" \
    --text_encoder_name google/mt5-xxl \
    --text_prompt examples/val_road_long/prompt_val_road.txt \
    --ae CausalVAEModel_D4_4x8x8 \
    --ae_path "/path/to/causalvideovae" \
    --save_video_path "./results/road-output.mp4" \
    --guidance_scale 3 \
    --num_sampling_steps 100 \
    --enable_tiling \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --given_frame_num 3 \
    --mask_ratio_l 0.25 \
    --mask_ratio_r 0.25 \
    --mask_ratio_u 0.0 \
    --mask_ratio_d 0.0