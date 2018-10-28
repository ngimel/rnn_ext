#include <ATen/ATen.h>
#include "ATen/cuda/CUDAContext.h"
#include "cublas_v2.h"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/core/Half.h>

namespace at { namespace native {

    namespace {
    template<typename scalar_t>
    __global__ void add(scalar_t * in1, scalar_t * in2, scalar_t * out, int n){
         int tid = threadIdx.x+blockIdx.x*blockDim.x;
         for (int i=tid; i<n; i += blockDim.x*gridDim.x){
            out[i] = in1[i]+in2[i];
         }
    }
    
    }



    at::Tensor run_gemms(const Tensor& a, const Tensor& h, const Tensor& w1, const Tensor& w2, const bool use_streams){
       int64_t seq_length = a.size(0);
       int64_t batch_size = a.size(1);
       auto blasHandle = at::cuda::getCurrentCUDABlasHandle();
       cublasSetMathMode(blasHandle, CUBLAS_TENSOR_OP_MATH);
       cuda::CUDAStream s1=nullptr, s2=nullptr;
       at::cuda::CUDAEvent event;
       if (use_streams == true){
           s1 = at::cuda::createCUDAStream();
           s2 = at::cuda::createCUDAStream();
       }
//gemm: 
       int m = w1.size(1);
       int k = w1.size(0);
       int n = h.size(0);
       Tensor out1 = at::empty_like(h);
       Tensor out2 = at::empty_like(h);
       cublasStatus_t err;
       for (int t=0; t< seq_length; t++){
           auto in_a = a[t];
           cudaStream_t  ss1, ss2=nullptr;
           if (use_streams) {
             ss1  = s1.stream();
             cublasSetStream(blasHandle, ss1);    
           }
           //submit first gemm
           float alpha = 1.f;
           float beta = 0.f;
           err = cublasGemmEx(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   m, n, k, &alpha,
                                   w1.data<at::Half>(), CUDA_R_16F, m, in_a.data<at::Half>(), CUDA_R_16F,
                                   k, &beta, out1.data<at::Half>(), CUDA_R_16F, m,
                                   CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
           AT_CHECK(err==0, "error in cublas"); 
           if (use_streams) {
              event.record(s1);
              ss2 = s2.stream();
              cublasSetStream(blasHandle, ss2);
           } 
             
           //submit second gemm    
           err = cublasGemmEx(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   m, n, k, &alpha,
                                   w2.data<at::Half>(), CUDA_R_16F, m, h.data<at::Half>(), CUDA_R_16F,
                                   k, &beta, out2.data<at::Half>(), CUDA_R_16F, m,
                                   CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
           AT_CHECK(err==0, "error in cublas"); 
           if (use_streams) {
              s2.synchronize_with(event);
           }
        
          int numThreads = 256;
          int numBlocks = (h.numel()+numThreads -1)/numThreads;
          AT_DISPATCH_ALL_TYPES_AND_HALF(a.type(), "add", [&] {
           add<<<numBlocks, numThreads, 0, ss2>>>(
           out1.data<scalar_t>(), out2.data<scalar_t>(), h.data<scalar_t>() , h.numel());
 
          });
       }
      
       
       
       return h;
    }
}}
