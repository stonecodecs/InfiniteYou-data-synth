# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
import time
import random
import glob
from pathlib import Path
from prompt_sampler import PromptSampler
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as T

from pipelines.pipeline_infu_flux import InfUFluxPipeline

def get_pod_index():
    """Get the pod index from environment variables."""
    # Try different ways to get pod index
    # hostname = os.environ.get('HOSTNAME', '')
    return int(os.environ.get('JOB_COMPLETION_INDEX', 0))
    
    # # Extract index from hostname (e.g., "infgen-abc123" -> 0)
    # if 'infgen' in hostname:
    #     # For Kubernetes jobs, hostname format is typically job-name-randomstring
    #     # We'll use the last part to determine index
    #     parts = hostname.split('-')
    #     if len(parts) >= 3:
    #         # Use hash of the random part to get consistent index
    #         random_part = parts[-1]
    #         return hash(random_part) % int(os.environ.get('JOB_PARALLELISM', 1))
    
    # # Fallback: use HOSTNAME hash
    # return hash(hostname) % int(os.environ.get('JOB_PARALLELISM', 1))

def get_assigned_subjects(root_dir, pod_index, total_pods):
    """Get the list of subjects assigned to this pod based on index."""
    all_subjects = []
    for item in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, item)
        if os.path.isdir(subject_path):
            all_subjects.append(item)
    
    # Sort for consistent partitioning
    all_subjects = sorted(all_subjects)
    
    # Assign subjects to pods using modulo
    assigned_subjects = []
    for i, subject in enumerate(all_subjects):
        if i % total_pods == pod_index:
            assigned_subjects.append(subject)
    
    return assigned_subjects

def stopwatch(func):
    def wrapper(*args, **kwargs):
        timer_name = kwargs.get('timer_name', None)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if timer_name is not None:
            print(f"{timer_name} took {end_time - start_time:.2f} seconds")
        else:
            print(f"Time taken: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def enhance_prompt(prompt, subject_prompt):
    """Enhance prompt with subject name and prompt json."""
    gender = subject_prompt.split(" ")[1] # apparently, this is always man, woman, or lady
    prompt = prompt.lower()
    if "person" in prompt:
        new_prompt = prompt.replace("person", "one " + gender)
    else: # in prompts, if not Person, then it starts with a verb
        new_prompt = "one " + gender + " " + prompt # then, attach the gender to the prompt
    return new_prompt

# if the output directory already exists with the desired number of samples, skip it.
def apply_over_all_subdirectories(root_dir, func, skip_existing=True, **kwargs):
    # root dir is mvhumandata/mv_captures here
    for subject in tqdm(os.listdir(root_dir), desc="Processing subjects", total=len(os.listdir(root_dir))):
        if os.path.isdir(os.path.join(root_dir, subject)):
            func(os.path.join(root_dir, subject), **kwargs) # this calls the subjects
    
def get_representative_image(subject_dir, front_camera='CC32871A059', start_index=5):
    """Get the most front-facing image from a subject directory."""
    images_dir = os.path.join(subject_dir, 'images_lr')
    front_camera_dir = os.path.join(images_dir, front_camera)
    if os.path.exists(front_camera_dir):
        try:
            # get the first timestep image available
            index = start_index
            filename = f"{index:04d}_img.jpg"
            img_path = os.path.join(front_camera_dir, filename)
            while not os.path.exists(img_path): # ! hack, but avoids infinite loop
                index += 5
                filename = f"{index:04d}_img.jpg"
                img_path = os.path.join(front_camera_dir, filename)
                # Change maximum value from 50 to 2105 to prevent not returning any image for pruned out objects
                if index > 2105: # if not found in first 10 images, this directory is probably empty and/or can't find face.
                    raise ValueError(f"No image found for {subject_dir}. Tried for 10 indices.")
            return img_path
        except Exception as e:
            print(f"Error listing directory {front_camera_dir}: {e}")
            return None
    else:
        raise ValueError(f"Camera {front_camera} not found in {subject_dir}.")

# ! not used
def get_image_files(subject_dir, extensions=['*.jpg', '*.jpeg', '*.png']):
    """Get all image files from a directory. This will be applied to cameras."""
    image_files = []
    images_dir = os.path.join(subject_dir, 'images_lr')
    camera_dirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    for camera_dir in camera_dirs:
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, camera_dir, ext)))
            image_files.extend(glob.glob(os.path.join(images_dir, camera_dir, ext.upper())))
    return sorted(image_files)

def load_prompt_json(prompt_json_file):
    """Load prompt json from a .json file."""
    with open(prompt_json_file, 'r') as f:
        return json.load(f)

def init_prompt_sampler(prompt_file):
    """Load prompts from a .txt file (one prompt per line)."""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file {prompt_file} not found")
    return PromptSampler(prompt_file)

def get_number_of_original_folder(image_path):
    """This function is to compute the exact number of images before pruning out"""
    default_step_size = 5 # By default, step size is 5
    image_list = sorted(os.listdir(os.path.dirname(image_path)))
    starting_index = int(image_list[0][:4])
    ending_index = int(image_list[-1][:4])
    num_images = (ending_index - starting_index) // default_step_size + 1
    return num_images

def process_subject(subject_dir, prompt_sampler, pipe, output_dir, num_samples=1, step_size=60, subject_prompt=None, **kwargs):
    """Process a subject directory."""
    # NOTE: because InfiniteYou expects faces, we will be using the most front-facing image.
    #       This refers to the first image of camera CC32871A059.
    #       Thus, we create multiple generations using this image only. 

    # image_files = get_image_files(subject_dir)
    # for image_file in image_files:
    #     process_single_image(pipe, image_file, prompt_sampler, **kwargs)

    seed = kwargs.get('seed', 0)
    step_size = kwargs.get('step_size', 60)
    try: 
        image_path = get_representative_image(subject_dir)
    except Exception as e:
        print(f"Error getting representative image: {e}")
        return

    # get number of images within image_path (camera)
    # not sure if this guarantees timestep order, but actually might not matter.
    # Fix return empty synthesis folder
    num_timesteps = get_number_of_original_folder(image_path) // step_size
    num_generations = num_samples * num_timesteps

    # ! hack, but process_single_image doesn't allow the kwarg key "step_size"
    kwargs.pop('step_size', None)
    
    # get prompts
    prompts = prompt_sampler.sample_prompt(num_samples=num_generations)
    subject_name = subject_dir.split('/')[-1]
    print("Subject: ", subject_name)

    # if subject already exists in output directory, we "concatenate" new samples to it.
    new_number = 1 # for save file name
    if os.path.exists(os.path.join(output_dir, subject_name)):
        print("Output directory already exists.")
        files = os.listdir(os.path.join(output_dir, subject_name))
        if len(files) > 0:
            highest_number = max([int(f.split('_')[0]) for f in files if f.endswith('.png')])
            new_number = highest_number + 1
            num_generations = num_generations - new_number # ! HACK to get constant samples per subject
            if num_generations <= 0:
                print(f"{subject_name} already completed for {num_samples * num_timesteps} samples.")
                return
            print(f"Continuing from image #: {new_number}")

    os.makedirs(os.path.join(output_dir, subject_name), exist_ok=True)

    # generate images
    no_faces = False
    rep_image_path = None # only check once
    for i, prompt in tqdm(enumerate(prompts), desc="Generating images", total=num_generations):
        image = None
        retry_count = 0 # for face rec
        seed = torch.seed() & 0xFFFFFFFF if seed == 0 else seed + i
        kwargs.update({"seed": seed})
        output_path = os.path.join(output_dir, subject_name, f"{(new_number+i):06d}_{subject_name}_img.png")
        if subject_prompt is not None:
            prompt = enhance_prompt(prompt, subject_prompt) # update prompt
        print(f"[{i}] Prompt: ", prompt)
        # generate and save

        # if can't find a face, iterate 10 more images, otherwise, break
        while image is None:
            try:
                if rep_image_path is None:
                    image_path = get_representative_image(subject_dir, start_index=retry_count * 5)
                image, time_elapsed, error = process_single_image(pipe, image_path, prompt, **kwargs)
                if image is not None: # if we found a face, stop checking for representative images
                    rep_image_path = image_path
                retry_count += 1
            except Exception as e:
                print(e)
                no_faces = True

        if no_faces:
            break # go next subject
        
        # Clear GPU memory between images
        torch.cuda.empty_cache()

        if image is None:
            print(f"✗ Failed: {error}")
            continue
        image.save(output_path)
        print(f"✓ Generated in {time_elapsed:.2f}s -> {output_path}")


def process_single_image(pipe, id_image_path, prompt, control_image_path=None, **kwargs):
    """Process a single image with the loaded pipeline."""
    try:
        # Load and process ID image
        id_image = Image.open(id_image_path).convert('RGB')

        # reduce input size of input image
        transform = T.Compose([
            T.CenterCrop(1500),
        ])

        id_image = transform(id_image)
        
        # Load control image if provided
        control_image = None
        if control_image_path and os.path.exists(control_image_path):
            control_image = Image.open(control_image_path).convert('RGB')
        
        # Generate image
        start_time = time.time()
        image = pipe(
            id_image=id_image,
            prompt=prompt,
            control_image=control_image,
            **kwargs
        )

        generation_time = time.time() - start_time # TODO: get time metrics out of this
        return image, generation_time, None
        
    except Exception as e:
        return None, 0, str(e)

def init_model(args):
    # Load pipeline (only once!)
    infu_model_path = os.path.join(args.model_dir, f'infu_flux_{args.infu_flux_version}', args.model_version)
    insightface_root_path = os.path.join(args.model_dir, 'supports', 'insightface')
    pipe = InfUFluxPipeline(
        base_model_path=args.base_model_path,
        infu_model_path=infu_model_path,
        insightface_root_path=insightface_root_path,
        infu_flux_version=args.infu_flux_version,
        model_version=args.model_version,
        quantize_8bit=args.quantize_8bit,
        cpu_offload=args.cpu_offload,
    )
    
    # Load LoRAs (optional)
    lora_dir = os.path.join(args.model_dir, 'supports', 'optional_loras')
    if not os.path.exists(lora_dir): 
        lora_dir = './models/InfiniteYou/supports/optional_loras'
    loras = []
    if args.enable_realism_lora:
        loras.append([os.path.join(lora_dir, 'flux_realism_lora.safetensors'), 'realism', 1.0])
    if args.enable_anti_blur_lora:
        loras.append([os.path.join(lora_dir, 'flux_anti_blur_lora.safetensors'), 'anti_blur', 1.0])
    pipe.load_loras(loras)
    return pipe

def main():
    parser = argparse.ArgumentParser(description='Process directory of images with prompts from file')
    parser.add_argument('--root_dir', required=True, help='Directory containing subjects.')
    parser.add_argument('--prompt_file', required=True, help='File containing prompts (text file)')
    parser.add_argument('--control_image', default=None, help='Control image path (optional)')
    parser.add_argument('--base_model_path', default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--model_dir', default='./models/InfiniteYou')
    parser.add_argument('--infu_flux_version', default='v1.0')
    parser.add_argument('--model_version', default='sim_stage1')
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--guidance_scale', default=3.5, type=float)
    parser.add_argument('--num_steps', default=30, type=int)
    parser.add_argument('--infusenet_conditioning_scale', default=1.0, type=float)
    parser.add_argument('--infusenet_guidance_start', default=0.0, type=float)
    parser.add_argument('--infusenet_guidance_end', default=1.0, type=float)
    parser.add_argument('--enable_realism_lora', action='store_true')
    parser.add_argument('--enable_anti_blur_lora', action='store_true')
    parser.add_argument('--quantize_8bit', action='store_true')
    parser.add_argument('--cpu_offload', action='store_true')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Base seed (0 for random)')
    parser.add_argument('--max_images', type=int, help='Maximum number of images to process (default: all)')
    parser.add_argument('--width', type=int, default=576, help='Width of the output image')
    parser.add_argument('--height', type=int, default=576, help='Height of the output image')
    parser.add_argument('--skip_existing', action='store_true', help='Skip existing output directories')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate per subject')
    parser.add_argument('--step_size', type=int, default=60, help='Step size for frames to process.')
    parser.add_argument('--prompt_json', type=str,default="/workspace/datasetvol/mvhuman_data/text_description_48.json", help='Prompt json file')
    args = parser.parse_args()

    # Check arguments
    assert args.infu_flux_version == 'v1.0', 'Currently only supports InfiniteYou-FLUX v1.0'
    assert args.model_version in ['aes_stage2', 'sim_stage1'], 'Currently only supports model versions: aes_stage2 | sim_stage1'

    # Set cuda device
    torch.cuda.set_device(args.cuda_device)

    # Load prompts from file
    try:
        prompt_sampler = init_prompt_sampler(args.prompt_file)
        prompt_json = load_prompt_json(args.prompt_json)
        print(f"Loaded prompts from {args.prompt_file}")
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return

    # Load model
    print("Loading InfiniteYou pipeline...")
    start_load_time = time.time()
    pipe = init_model(args)
    load_time = time.time() - start_load_time
    print(f"Pipeline loaded in {load_time:.2f} seconds")

    # Create output directory
    # should contain all generated images,
    # then should copy over to the PVC on another thread.
    # this should initially be local to the machine, then should copy over to the PVC.
    # should be done in intervals such that we don't lose everything upon a crash.
    os.makedirs(args.output_dir, exist_ok=True)

    kwargs = {
        'pipe': pipe,
        'prompt_sampler': prompt_sampler,
        'control_image_path': args.control_image,
        'seed': args.seed,
        'guidance_scale': args.guidance_scale,
        'num_steps': args.num_steps,
        'infusenet_conditioning_scale': args.infusenet_conditioning_scale,
        'infusenet_guidance_start': args.infusenet_guidance_start,
        'infusenet_guidance_end': args.infusenet_guidance_end,
        'cpu_offload': args.cpu_offload,
        'width': args.width,
        'height': args.height,
        'output_dir': args.output_dir,
        'num_samples': args.num_samples,
        'step_size': args.step_size,
    }

    # Generate images for all subjects # ! - DEPRECATED
    # apply_over_all_subdirectories(args.root_dir, process_subject, **kwargs)

    # Generate images for a subset of subjects based on pod index
    pod_index = get_pod_index()
    total_pods = int(os.environ.get('JOB_PARALLELISM', 1))
    pod_id = os.environ.get('HOSTNAME', f'pod_{random.randint(1000, 9999)}')
    assigned_subjects = get_assigned_subjects(args.root_dir, pod_index, total_pods)
    print(f"Pod {pod_id} assigned subjects {assigned_subjects[:5]}...")

    for subject in tqdm(assigned_subjects, desc="Processing subjects", total=len(assigned_subjects)):
        prompt_for_subject = prompt_json[subject] if subject in prompt_json else None
        kwargs.update({'step_size': args.step_size}) # another hack to include step_size in kwargs (after popping within process_subject)
        kwargs.update({'subject_prompt': prompt_for_subject})
        process_subject(os.path.join(args.root_dir, subject), **kwargs)

    print(f"[InfiniteYou] Pod {pod_id} completed generations!")


if __name__ == "__main__":
    main() 