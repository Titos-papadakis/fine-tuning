"""
QLoRA fine-tuning.

Runs end-to-end on a single free-tier Colab T4 (16GB):
  - 4-bit NF4 quantized base (bitsandbytes)
  - LoRA adapters via Unsloth (patched kernels: faster and materially lower VRAM
    than vanilla PEFT+TRL on the same hardware)
  - Unsloth gradient checkpointing ("unsloth" mode, not HF's default)
  - 8-bit AdamW optimizer state
  - small per-device batch + accumulation for an effective batch of 8 without
    ever materialising more than 2 sequences at once
  - completion-only loss masking, so the model is never trained to reproduce
    its own prompt

Config is validated (`ftspec.config.Config`) before a single weight is loaded,
and the dataset is expected to have passed `ftspec validate` first — a training
job is an expensive place to discover a truncated sample.
"""
from __future__ import annotations

from ftspec.config import Config
from ftspec.run import get_logger

log = get_logger("ftspec.train")

# Chat-template markers used to mask the prompt out of the loss.
RESPONSE_MARKERS = {
    "llama": ("<|start_header_id|>user<|end_header_id|>\n\n",
               "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "qwen": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
}


def detect_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "qwen" in lowered:
        return "qwen"
    if "llama" in lowered:
        return "llama"
    return ""


def run(cfg: Config, profile_name: str, max_steps: int = -1,
         resume: bool = False) -> dict:
    """Fine-tune and save adapters. Returns metrics for the run manifest."""
    # Heavy imports are deferred so `--help`, config validation and CI linting
    # do not require a CUDA toolchain to be installed.
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    m, lora, t = cfg.model, cfg.lora, cfg.training

    log.info("loading base model %s (4bit=%s)", m.base_model, m.load_in_4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=m.base_model,
        max_seq_length=m.max_seq_length,
        dtype=m.dtype,
        load_in_4bit=m.load_in_4bit,
    )

    log.info("attaching LoRA adapters (r=%d, alpha=%d, %d target modules)",
              lora.r, lora.lora_alpha, len(lora.target_modules))
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora.r,
        target_modules=lora.target_modules,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias,
        use_gradient_checkpointing=lora.use_gradient_checkpointing,
        random_state=lora.random_state,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("trainable parameters: %s / %s (%.3f%%)",
              f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    data_dir = cfg.data_dir(profile_name)
    log.info("loading corpus from %s", data_dir)
    dataset = load_dataset("json", data_files={
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "val.jsonl"),
    })

    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    # remove_columns drops `meta`, which exists for analysis and must not reach the trainer.
    dataset = dataset.map(format_chat, remove_columns=dataset["train"].column_names)

    sft_config = SFTConfig(
        output_dir=str(cfg.checkpoints_dir(profile_name)),
        num_train_epochs=t.num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        gradient_checkpointing=t.gradient_checkpointing,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=t.warmup_ratio,
        weight_decay=t.weight_decay,
        optim=t.optim,
        logging_steps=t.logging_steps,
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        seed=t.seed,
        packing=t.packing,
        report_to=t.report_to,
        max_seq_length=m.max_seq_length,
        dataset_text_field="text",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=sft_config,
    )

    if t.train_on_responses_only:
        family = detect_family(m.base_model)
        if family:
            from unsloth.chat_templates import train_on_responses_only
            instruction_part, response_part = RESPONSE_MARKERS[family]
            trainer = train_on_responses_only(
                trainer, instruction_part=instruction_part, response_part=response_part)
            log.info("completion-only loss enabled (%s chat template)", family)
        else:
            log.warning("unknown chat template for %s; training on the full sequence",
                         m.base_model)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log.info("GPU: %s, %.1f GB VRAM, effective batch size %d",
                  props.name, props.total_memory / 1024 ** 3, t.effective_batch_size)

    train_output = trainer.train(resume_from_checkpoint=resume or None)

    adapter_dir = cfg.adapter_dir(profile_name)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("saved LoRA adapter -> %s", adapter_dir)

    merged_dir = None
    if cfg.merge.save_merged_16bit:
        merged_dir = cfg.merged_dir(profile_name)
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        log.info("saved merged 16-bit model -> %s  (ready for `ftspec serve`)", merged_dir)

    peak_vram = (torch.cuda.max_memory_reserved() / 1024 ** 3) if torch.cuda.is_available() else 0.0
    return {
        "profile": cfg.profile,
        "train_runtime_s": round(train_output.metrics.get("train_runtime", 0.0), 1),
        "train_loss": round(train_output.metrics.get("train_loss", 0.0), 5),
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 4),
        "peak_vram_gb": round(peak_vram, 2),
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir) if merged_dir else None,
    }
