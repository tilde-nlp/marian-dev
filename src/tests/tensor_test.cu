#include <iostream>
#include <fstream>

#include <cublas_v2.h>
#include <thrust/device_vector.h>

void Prod(cublasHandle_t handle,
          float* cdata, const float* adata, const float* bdata,
          size_t m, size_t k, size_t n) {

  float alpha = 1.0;
  float beta  = 0.0;

  size_t lda = k;
  size_t ldb = n;
  size_t ldc = n;

  cublasOperation_t opA = CUBLAS_OP_N;
  cublasOperation_t opB = CUBLAS_OP_N;

  cublasStatus_t stat;
  stat = cublasSgemm(handle, opB, opA, n, m, k,
                     &alpha, bdata, ldb, adata, lda, &beta, cdata, ldc);

  if(stat != CUBLAS_STATUS_SUCCESS)
    std::abort();
}

int main(int argc, char** argv) {

  cudaSetDevice(0);

  std::ifstream data("data.bin", std::ifstream::in);

  std::vector<float> hA(3072);
  data.read((char*)hA.data(), sizeof(float) * hA.size());

  std::vector<float> hB(3072 * 8);
  data.read((char*)hB.data(), sizeof(float) * hB.size());

  std::vector<float> hC(8);

  thrust::device_vector<float> dA(hA.size());
  thrust::device_vector<float> dB(hB.size());
  thrust::device_vector<float> dC(hC.size());

  thrust::copy(hA.begin(), hA.end(), dA.begin());
  thrust::copy(hB.begin(), hB.end(), dB.begin());

  cublasHandle_t handle;
  cublasCreate(&handle);

  const float* adata = thrust::raw_pointer_cast(dA.data());
  const float* bdata = thrust::raw_pointer_cast(dB.data());
  float* cdata = thrust::raw_pointer_cast(dC.data());

  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
  Prod(handle, cdata, adata, bdata, 1, 3072, 8);

  for(auto c : dC)
    std::cerr << c << " ";
  std::cerr << std::endl << std::endl;

  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  Prod(handle, cdata, adata, bdata, 1, 3072, 8);

  for(auto c : dC)
    std::cerr << c << " ";
  std::cerr << std::endl << std::endl;

  cublasDestroy(handle);
  return 0;
}
