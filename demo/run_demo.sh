#!/bin/bash
echo "======================================"
echo "  KernelClaw Live Demo"
echo "======================================"
echo ""
echo "STEP 1: NemoClaw blocking dangerous actions"
echo "-------------------------------------------"
cat ~/kernelclaw/demo/blocked_actions.log
echo ""
echo "STEP 2: Per-binary network enforcement"
echo "---------------------------------------"
grep -A 6 "nvidia:" ~/kernelclaw/demo/policy_dump.txt
echo ""
echo "STEP 3: Live autonomous optimization"
echo "-------------------------------------"
source ~/kernelclaw/.venv/bin/activate
python3 ~/kernelclaw/agent.py --once ~/kernelclaw/kernels/naive_matmul.cu
echo ""
echo "STEP 4: Full audit log"
echo "----------------------"
python3 -c "
import json
runs=[json.loads(l) for l in open('/home/deepak/kernelclaw/runs.jsonl') if l.strip()]
verified=[r for r in runs if r['status']=='verified']
print('\n=== KERNELCLAW VERIFIED RESULTS ===')
for r in verified:
    print(f\"✅ {r['kernel_name']:20s} speedup={r['speedup']:6.2f}x  baseline={r['baseline_us']:8.0f}us  optimized={r['candidate_us']:8.0f}us\")
print(f'\n{len(verified)} successful optimizations')
"
