#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <torch/torch.h>

namespace at { namespace native {

//at::Tensor revert_varlen_tensor(const Tensor& input, const Tensor& lengths);
at::Tensor run_gemms(const Tensor& a, const Tensor& h, const Tensor& w1, const Tensor& w2, const bool use_streams);



at::Tensor gemm_stream(const Tensor& a, const Tensor& h, const Tensor& w1, const Tensor w2, const bool use_streams){
   AT_CHECK(a.is_cuda() && a.type().scalarType() ==  ScalarType::Half, "only works for cuda Half tensors");  
   AT_CHECK(h.is_cuda() && h.type().scalarType() ==  ScalarType::Half, "only works for cuda Half tensors");  
   AT_CHECK(w1.is_cuda() && w1.type().scalarType() ==  ScalarType::Half, "only works for cuda Half tensors");  
   AT_CHECK(w2.is_cuda() && w2.type().scalarType() ==  ScalarType::Half, "only works for cuda Half tensors");  
   AT_CHECK(a.ndimension() == 3 && h.ndimension() == 2 && w1.ndimension() == 2 && w2.ndimension() == 2);
   at::Tensor out = run_gemms(a, h, w1, w2, use_streams);
   return out;
}   

}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("gemm_stream", &at::native::gemm_stream);
}
  
