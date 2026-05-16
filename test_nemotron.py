import os
from openai import OpenAI

MODEL = "nvidia/nemotron-3-nano-30b-a3b"

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

with open("vadd.cu") as f:
    kernel = f.read()
with open("ncu_report.txt") as f:
    report = f.read()

SYSTEM = (
    "You are a CUDA performance expert. You receive a CUDA source file and "
    "Nsight Compute profiler output. Identify the most likely bottleneck from "
    "the profiler data, then output a rewritten kernel that addresses it. "
    "The rewrite must be functionally equivalent (same outputs for same inputs). "
    "Respond in exactly two parts: (1) a 3-5 sentence diagnosis, (2) a single "
    "fenced cuda code block with the full rewritten source file. Nothing else."
)

USER = (
    "KERNEL SOURCE (vadd.cu):\n"
    "```cuda\n"
    f"{kernel}\n"
    "```\n\n"
    "NSIGHT COMPUTE OUTPUT:\n"
    "```\n"
    f"{report}\n"
    "```\n\n"
    "Diagnose and rewrite."
)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ],
    temperature=0.2,
    max_tokens=4000,
)
print(resp.choices[0].message.content)
