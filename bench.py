import torch
import gemm_C
import time

seq_length = 40
h_size = 512
bs = 64 
input = torch.randn(seq_length, bs, h_size, dtype = torch.float16, device="cuda")
h = torch.zeros(bs, h_size, dtype = torch.float16, device="cuda")
w1 = torch.randn(h_size, h_size, dtype = torch.float16, device="cuda")
w2 = torch.randn(h_size, h_size, dtype = torch.float16, device="cuda")
out = gemm_C.gemm_stream(input, h, w1, w2, False)
torch.cuda.synchronize()
print("seq_length  h_size b_size time_streams(ms) time_nostreams(ms) speed_up") 
for h_size in [512, 1024, 2048]:
    for bs in [64, 128, 256]:
        input = torch.randn(seq_length, bs, h_size, dtype = torch.float16, device="cuda")
        h = torch.zeros(bs, h_size, dtype = torch.float16, device="cuda")
        w1 = torch.randn(h_size, h_size, dtype = torch.float16, device="cuda")
        w2 = torch.randn(h_size, h_size, dtype = torch.float16, device="cuda")
        out = gemm_C.gemm_stream(input, h, w1, w2, False)
        out = gemm_C.gemm_stream(input, h, w1, w2, True)
        torch.cuda.synchronize()
        start = time.time()
        out = gemm_C.gemm_stream(input, h, w1, w2, False)
        torch.cuda.synchronize()
        no_stream = time.time() - start
        start = time.time()
        out = gemm_C.gemm_stream(input, h, w1, w2, True)
        torch.cuda.synchronize()
        stream = time.time() - start
        print('{:5d} {:5d} {:5d} {:6.3f} {:6.3f} {:6.3f}'.format(seq_length, h_size, bs, stream*1000, no_stream*1000, no_stream/stream))

   


