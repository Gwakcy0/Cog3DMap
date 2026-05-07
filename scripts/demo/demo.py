import torch
from PIL import Image
import requests
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Load model with memory-efficient settings
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir="./cache",
    torch_dtype=torch.bfloat16,  # Use bf16 for memory efficiency
    device_map="auto",  # Automatically distribute model across available devices
    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "What is the type of the Dog?"},
        ],
    }
]

# Prepare inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Move inputs to the same device as the model
inputs = inputs.to(model.device)

# Generate with memory-efficient settings
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,  # Greedy decoding to save memory
    )

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
