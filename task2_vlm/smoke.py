"""Full-length CUDA smoke test for the connector and attention LoRA."""
from __future__ import annotations

import torch
from torch.amp import GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, SiglipVisionModel
from PIL import Image

from .prompts import build_example
from .development import per_example_nll
from .final_eval import greedy
from .train import optimizer_for
from .vlm import VisionLanguageModel


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Task 2 smoke requires a CUDA GPU")
    torch.manual_seed(6304); torch.cuda.manual_seed_all(6304)
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    vision = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", torch_dtype=torch.float16).to("cuda").eval()
    pixels = processor(images=Image.new("RGB", (224, 224)), return_tensors="pt")["pixel_values"].to("cuda", dtype=torch.float16)
    with torch.no_grad():
        patch_features = vision(pixel_values=pixels).last_hidden_state
    if tuple(patch_features.shape) != (1, 196, 768):
        raise RuntimeError(f"Unexpected SigLIP output: {tuple(patch_features.shape)}")
    print("SigLIP patch feature shape PASS", flush=True)
    del vision, pixels, patch_features, processor
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = VisionLanguageModel(model.to("cuda")).to("cuda")
    features = torch.randn(196, 768, dtype=torch.float16)
    rows = [build_example(tokenizer, "cat " * 100, "yes " * 80, 64, features) for _ in range(8)]
    scaler = GradScaler("cuda", init_scale=1024.0)
    torch.cuda.reset_peak_memory_stats()
    for stage in (1, 2):
        if stage == 2:
            model.enable_lora()
        model.train()
        optimizer = optimizer_for(model, stage)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            # Two 8-example microbatches exercise the prescribed accumulation.
            total = 0.0
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model.loss(rows) / 2
                scaler.scale(loss).backward()
                total += float(loss.detach())
            scaler.unscale_(optimizer)
            connector_grad = sum(float(p.grad.detach().abs().sum()) for p in model.connector.parameters() if p.grad is not None)
            adapter_grad = sum(float(p.grad.detach().abs().sum()) for p in model.backbone.parameters() if p.requires_grad and p.grad is not None)
            if connector_grad <= 0 or (stage == 2 and adapter_grad <= 0):
                raise RuntimeError(f"Missing connector or adapter gradients at stage {stage}")
            trainable = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
            scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() < scale:
                raise RuntimeError(f"Task 2 smoke skipped an optimizer step in stage {stage}")
            print(f"Task 2 smoke stage {stage} update {step + 1}: loss={total:.4f}", flush=True)
    eval_rows = [{"question": "Is there a cat?", "answer": "yes"}]
    sample_features = {1: features}
    for image_id in (1, None):
        score = per_example_nll(model, tokenizer, sample_features, eval_rows, [image_id], caption=False)
        if len(score) != 1 or not torch.isfinite(torch.tensor(score)).all():
            raise RuntimeError("Task 2 smoke produced invalid reference NLL")
        if len(greedy(model, tokenizer, sample_features, eval_rows, [image_id])) != 1:
            raise RuntimeError("Task 2 smoke generation returned incorrect output count")
    print(f"Task 2 GPU smoke PASS: peak allocated {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
