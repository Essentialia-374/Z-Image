"""Z-Image PyTorch Native Inference."""

import os
import time
import warnings

import torch

warnings.filterwarnings("ignore")
from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate


def main():
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)  # True to verify with md5
    dtype = torch.bfloat16
    compile = False  # default False for compatibility
    output_path = "example.png"
    height = 1024
    width = 1024
    num_inference_steps = 8
    guidance_scale = 0.0
    seed = 42
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    prompt = (
        "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. "
        "Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. "
        "Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, "
        "silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
    )

    # Device selection priority: cuda -> tpu -> mps -> cpu
    if torch.cuda.is_available():
        device = "cuda"
        print("Chosen device: cuda")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            print("Chosen device: tpu")
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
                print("Chosen device: mps")
            else:
                device = "cpu"
                print("Chosen device: cpu")
                
    def log_vram_usage(label: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            reserved = torch.cuda.memory_reserved()
            allocated = torch.cuda.memory_allocated()
            print(
                f"[{label}] VRAM usage | total: {total / 1e9:.2f} GB, "
                f"free: {free / 1e9:.2f} GB, reserved: {reserved / 1e9:.2f} GB, "
                f"allocated: {allocated / 1e9:.2f} GB"
            )
        else:
            print(f"[{label}] VRAM usage not available (CUDA not detected)")
            
    # Load models
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile)
    AttentionBackend.print_available_backends()
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")
    log_vram_usage("post-model-load")

    # Gen an image
    start_time = time.time()
    images = generate(
        prompt=prompt,
        **components,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device).manual_seed(seed),
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    log_vram_usage("post-generation")
    images[0].save(output_path)

    ### !! For best speed performance, recommend to use `_flash_3` backend and set `compile=True`
    ### This would give you sub-second generation speed on Hopper GPU (H100/H200/H800) after warm-up


if __name__ == "__main__":
    main()
