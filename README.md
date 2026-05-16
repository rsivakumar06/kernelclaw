
## Why NemoClaw, not just OpenClaw?

dOpenClaw gives you an autonomous agent. NemoClaw gives you an autonomous agent
you can actually trust in production. Here's what NemoClaw adds:

1. **Per-binary network enforcement** — not just "allow this domain" but "only
   THIS binary can call THIS endpoint." In KernelClaw, only the OpenClaw binary
   can reach integrate.api.nvidia.com. A compromised or malicious rewrite cannot
   exfiltrate data even if it tries. OpenClaw alone cannot enforce this.

2. **Landlock + seccomp filesystem isolation** — the LLM reasoning component
   cannot write outside /sandbox/candidates/ even if Nemotron genewdrates code
   that tries to. OpenClaw has no equivalent enforcement layer.

3. **GPU passthrough with policy** — NemoClaw grants the sandbox access to the
   physical NVIDIA GPU while still enforcing all other restrictions. Running
   bare OpenClaw on a DGX Spark gives you GPU access but zero sandboxing.

4. **Audit log by default** — every ALLOW and DENY is captured automatically.
   With OpenClaw you would have to build this yourself.

In KernelClaw, the most dangerous operation is LLM-generated CUDA code.
NemoClaw ensures that code can be reasoned about, written to disk, and
benchmarked — but never executed with elevated privileges or used to reach
unauthorized systems. That guarantee is what makes KernelClaw safe to run
autonomously on real codebases.
