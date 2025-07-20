python generate.py \
--root_dir /workspace/datasetvol/mvhuman_data/mv_captures \
--prompt_file /workspace/datasetvol/pose_prompts_simple.txt \
--output_dir /workspace/datasetvol/mvhuman_data/inconsistent_images \
--num_steps 25 \
--quantize_8bit \
--num_samples 1 \
--guidance_scale 3.5 \
--infusenet_guidance_start 0.0 \
--infusenet_conditioning_scale 1.0 \
--model_version sim_stage1 \
--width 576 \
--height 576 \
--step_size 20 \
--enable_realism_lora

# num_samples is how many PER TIMESTEP
# STEP SIZE is the interval of timesteps to skip.