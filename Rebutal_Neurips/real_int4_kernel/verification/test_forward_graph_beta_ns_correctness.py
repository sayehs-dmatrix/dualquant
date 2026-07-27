import sys, torch
sys.path.insert(0, "/tmp/claude-0/-root-numrd/1e334428-137f-442b-9669-4bc5945263d0/scratchpad/svdquant_repo/build/lib.linux-x86_64-cpython-312")
import nunchaku_min

dev = torch.device("cuda:0")
torch.manual_seed(0)

def sqnr_db(ref, approx):
    sig = (ref.float() ** 2).sum()
    noise = ((approx.float() - ref.float()) ** 2).sum()
    return 10.0 * torch.log10(sig / noise).item()

N, K = 14336, 4096
W = (torch.randn(N, K) * 0.02).to(torch.bfloat16).cuda()
beta = (torch.rand(K) * 3 + 1).to(torch.bfloat16).cuda()

gemm = nunchaku_min.QuantizedGEMM()
gemm.init(K, N, False, True, 0)
gemm.load_weight(W)
gemm.load_smooth(torch.ones(K, dtype=torch.bfloat16, device=dev))

custom_stream = torch.cuda.Stream()
print(f"custom stream ptr: {custom_stream.cuda_stream}  (must be != 0)")

all_good = True

with torch.cuda.stream(custom_stream):
    print("=== Test 0: calling forward_graph_beta under the DEFAULT (legacy) stream should raise ===")
    raised = False
    try:
        with torch.cuda.stream(torch.cuda.default_stream()):
            _ = gemm.forward_graph_beta(torch.randn(1, K, dtype=torch.bfloat16, device=dev), beta)
    except RuntimeError as e:
        raised = True
        print(f"  raised as expected: {e}")
    if not raised:
        print("  FAILED: did not raise on legacy stream!")
        all_good = False

    print()
    print("=== Test 1: single-call correctness (forward_graph_beta vs forward vs ground truth) ===")
    x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
    x_scaled = (x * beta).contiguous()
    y_ref = gemm.forward(x_scaled)
    y_gb = gemm.forward_graph_beta(x, beta)
    torch.cuda.current_stream().synchronize()
    y_true = (x.float() * beta.float()) @ W.float().T
    print(f"SQNR(forward vs true)            = {sqnr_db(y_true, y_ref):.2f} dB")
    print(f"SQNR(forward_graph_beta vs true) = {sqnr_db(y_true, y_gb):.2f} dB")

    print()
    print("=== Test 2: repeated-call correctness (simulating decode loop, DIFFERENT input each call) ===")
    torch.manual_seed(123)
    xs = [torch.randn(1, K, dtype=torch.bfloat16, device=dev) for _ in range(20)]
    for i, xi in enumerate(xs):
        y_gb_i = gemm.forward_graph_beta(xi, beta).clone()
        torch.cuda.current_stream().synchronize()
        y_true_i = (xi.float() * beta.float()) @ W.float().T
        s = sqnr_db(y_true_i, y_gb_i)
        print(f"  call {i:2d}: SQNR(graph_beta)={s:6.2f}dB")
        if s < 10:
            all_good = False

    print()
    print("=== Test 3: vary M within the same instance (batch=1 then batch=16, then back, then 8) ===")
    x16 = torch.randn(16, K, dtype=torch.bfloat16, device=dev)
    y16_gb = gemm.forward_graph_beta(x16, beta).clone()
    torch.cuda.current_stream().synchronize()
    y16_true = (x16.float() * beta.float()) @ W.float().T
    s16 = sqnr_db(y16_true, y16_gb)
    print(f"  M=16 after M=1 calls: SQNR={s16:.2f}dB")
    if s16 < 10:
        all_good = False

    x1b = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
    y1b_gb = gemm.forward_graph_beta(x1b, beta).clone()
    torch.cuda.current_stream().synchronize()
    y1b_true = (x1b.float() * beta.float()) @ W.float().T
    s1b = sqnr_db(y1b_true, y1b_gb)
    print(f"  M=1 after M=16: SQNR={s1b:.2f}dB")
    if s1b < 10:
        all_good = False

    x8 = torch.randn(8, K, dtype=torch.bfloat16, device=dev)
    y8_gb = gemm.forward_graph_beta(x8, beta).clone()
    torch.cuda.current_stream().synchronize()
    y8_true = (x8.float() * beta.float()) @ W.float().T
    s8 = sqnr_db(y8_true, y8_gb)
    print(f"  M=8 after M=1: SQNR={s8:.2f}dB")
    if s8 < 10:
        all_good = False

    print()
    print("=== Test 4b: more graph_beta calls under the custom stream, for good measure ===")
    torch.manual_seed(77)
    for i in range(5):
        xi = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
        y_gb = gemm.forward_graph_beta(xi, beta).clone()
        torch.cuda.current_stream().synchronize()
        y_true_i = (xi.float() * beta.float()) @ W.float().T
        s = sqnr_db(y_true_i, y_gb.float())
        print(f"  call {i}: SQNR(graph_beta vs true)={s:.2f}dB")
        if s < 10:
            all_good = False

print()
print("=== Test 4: forward_beta on the DEFAULT stream (its actual production usage) ===")
print("    (forward_beta was never designed to run under a caller-supplied stream, so it's")
print("     tested here on the default stream, not inside the custom-stream block above)")
for i in range(5):
    xi = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
    y_b = gemm.forward_beta(xi, beta).clone()
    y_true_i = (xi.float() * beta.float()) @ W.float().T
    s = sqnr_db(y_true_i, y_b.float())
    print(f"  forward_beta call {i}: SQNR(vs true)={s:.2f}dB")
    if s < 10:
        all_good = False

print()
print("ALL GOOD" if all_good else "CORRECTNESS FAILURE DETECTED")
