"""Exercise worst-length full-parameter training with initialized Adam state on the Kaggle GPU."""

import torch
from torch.amp import GradScaler
from transformers import AutoModelForCausalLM

from .train import add_l2_sp_gradients, memory_efficient_sft_loss
from .utils import parameter_groups, seed_everything


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The full-length memory probe requires a CUDA GPU")
    seed_everything(6304)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.float32, trust_remote_code=True
    )
    reference = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to("cuda").train()
    optimizer = torch.optim.AdamW(parameter_groups(model, 0.01), lr=2e-5,
                                  betas=(0.9, 0.999), eps=1e-8, foreach=False)
    scaler = GradScaler("cuda", init_scale=1024.0)
    ids = torch.randint(0, model.config.vocab_size, (2, 2048), device="cuda")
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
    torch.cuda.reset_peak_memory_stats()
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = memory_efficient_sft_loss(model, batch) / (2 * 2047)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        add_l2_sp_gradients(model, reference, 1e-3)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            raise RuntimeError(f"Memory probe optimizer update {update + 1} was skipped")
        print(f"Full-length memory probe update {update + 1}/2 completed", flush=True)
    torch.cuda.synchronize()
    print(f"Full-length memory probe PASS; peak allocated GPU GiB: {torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)


if __name__ == "__main__":
    main()
