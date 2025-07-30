import torch
# import la
from diffusers import FluxPipeline
import time
import subprocess

def new_backup_count():
  backup_index = 0

  with open("generated/count", "br+") as f:
    stored_index = f.read(4)
    if stored_index != b'':
      backup_index = int.from_bytes(stored_index)

    backup_index+=1

    f.seek(0, 0)
    f.write(backup_index.to_bytes(4))

  return backup_index

def run_diffusers():
  pipe = FluxPipeline.from_pretrained("flux1_local/FLUX.1-schnell", torch_dtype=torch.bfloat16)

  # pipe.enable_model_cpu_offload() # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
  pixel_width = int(128 / 1.5)
  image_count = 1
  prompt = "A cat holding a sign that says hello world"
  a = new_backup_count()
  pixel_width = 16

  for k in range(6):

    for j in range(1, 6):
      start = time.time()
      image = pipe(
          prompt,
          guidance_scale=0.0,
          num_images_per_prompt=image_count,
          num_inference_steps=j,
          max_sequence_length=64,
          width=pixel_width,
          height=pixel_width,

          generator=torch.Generator("cpu").manual_seed(0)
      )
      end = time.time()
      print(f"[{pixel_width}, {j}] Total Time Elapsed: ", end - start)

      for i in range(image_count):
        image.images[i].save(f"generated/flux-schnell_{a}.{pixel_width}.{j}.{i}.png")

    pixel_width *= 2


def run_stable_diffusion():
  a = new_backup_count()

  pixel_width = 1024
  with open(f"log_{a}.txt", "w") as log_file:

    for _ in range(1):

        for j in range(1, 5):
          start = time.time()
          subprocess.call(["./stable-diffusion.cpp/stable-diffusion.cpp/build/bin/sd",
                          "--diffusion-model",  "./stable-diffusion.cpp/stable-diffusion.cpp/models/flux1-schnell-q8_0.gguf",
                          "--vae", "./stable-diffusion.cpp/stable-diffusion.cpp/models/ae.sft",
                          "--clip_l", "./stable-diffusion.cpp/stable-diffusion.cpp/models/clip_l.safetensors",
                          "--t5xxl", "./stable-diffusion.cpp/stable-diffusion.cpp/models/t5xxl_fp16.safetensors",
                          "-p", "a smart primate programming a computer",
                          "--cfg-scale", "1.0",
                          "--sampling-method", "euler",
                          "-v",
                          "--steps",  str(j),
                          "-o", f"generated/sd_flux-schnell_{a}.{pixel_width}.{j}.{0}.png",
                          "-H", str(pixel_width),
                          "-W", str(pixel_width)])

          end = time.time()

          log_file.write(f"{pixel_width}, {j}, {end - start}\n")
          log_file.flush()
          print(f"[{pixel_width}, {j}] Total Time Elapsed: ", end - start)

        pixel_width *= 2


def main():
  run_stable_diffusion()

if __name__ == "__main__":
  main()