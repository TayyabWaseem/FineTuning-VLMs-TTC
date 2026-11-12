"""Qwen3 user/assistant formatting with content caps and visual insertion point."""
from __future__ import annotations


def build_example(tokenizer, question: str, answer: str, answer_cap: int,
                  features=None, include_eot: bool = True) -> dict:
    if answer_cap not in (16, 64):
        raise ValueError("Answer cap must be 16 (VQA) or 64 (caption)")
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"][:80]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"][:answer_cap]
    question = tokenizer.decode(question_ids, skip_special_tokens=False)
    answer = tokenizer.decode(answer_ids, skip_special_tokens=False)
    if not answer_ids:
        raise ValueError("Empty targets cannot be used for supervised loss")
    user = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(user + [{"role": "assistant", "content": answer}], tokenize=False,
                                         add_generation_prompt=False, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Question/answer chat template token alignment failed")
    user_prefix = tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    if prompt_ids[:len(user_prefix)] != user_prefix:
        raise RuntimeError("Unexpected Qwen user-message prefix; refusing misplaced visual tokens")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    user_end = next((i for i in range(len(user_prefix), len(prompt_ids)) if prompt_ids[i] == end), None)
    if user_end is None:
        raise RuntimeError("User end-of-turn token is missing")
    locations = [i for i in range(len(prompt_ids), len(full_ids)) if full_ids[i] == end]
    if not locations:
        raise RuntimeError("Assistant end-of-turn token is missing")
    response = full_ids[len(prompt_ids):locations[-1]][:answer_cap]
    prompt_ids = prompt_ids[:len(user_prefix)] + prompt_ids[len(user_prefix):user_end][:80] + prompt_ids[user_end:]
    full_ids = prompt_ids + response + full_ids[locations[-1]:]
    target_end = len(prompt_ids) + len(response) + int(include_eot)
    labels = [-100] * len(full_ids)
    labels[len(prompt_ids):target_end] = full_ids[len(prompt_ids):target_end]
    if not any(label != -100 for label in labels):
        raise RuntimeError("No supervised answer tokens")
    return {"input_ids": full_ids, "labels": labels, "visual_insert_index": len(user_prefix), "features": features}


def build_prompt(tokenizer, question: str, features=None) -> dict:
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"][:80]
    question = tokenizer.decode(question_ids, skip_special_tokens=False)
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    user_prefix = tokenizer("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    if ids[:len(user_prefix)] != user_prefix:
        raise RuntimeError("Unexpected Qwen user-message prefix")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    user_end = next((i for i in range(len(user_prefix), len(ids)) if ids[i] == end), None)
    if user_end is None:
        raise RuntimeError("User end-of-turn token is missing")
    ids = ids[:len(user_prefix)] + ids[len(user_prefix):user_end][:80] + ids[user_end:]
    return {"input_ids": ids, "labels": [-100] * len(ids), "visual_insert_index": len(user_prefix), "features": features}
