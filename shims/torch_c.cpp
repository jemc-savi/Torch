// TODO: Remove this workaround for Savi C compilation when it's no longer needed.
#ifndef __GCC_ATOMIC_TEST_AND_SET_TRUEVAL
#define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
#endif

#include<torch/csrc/autograd/engine.h>
#include<torch/csrc/jit/frontend/tracer.h>
#include<torch/csrc/jit/runtime/graph_executor.h>
#include<torch/csrc/jit/passes/fixup_trace_scope_blocks.h>
#include<torch/csrc/jit/passes/normalize_ops.h>
#include<torch/csrc/jit/runtime/graph_executor.h>
#include<ATen/autocast_mode.h>
#include<torch/script.h>
#include<torch/csrc/jit/passes/tensorexpr_fuser.h>
#include<stdexcept>
#include<vector>

extern "C" {

using namespace std;

static inline torch::TensorImpl* export_tensor(torch::Tensor tensor) {
  return tensor.unsafeReleaseTensorImpl();
}
static inline torch::Tensor import_tensor(torch::TensorImpl* tensor) {
  c10::raw::intrusive_ptr::incref(tensor);
  return torch::Tensor(
    c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::reclaim(tensor)
  );
}
// TODO: add decref_tensor function to call from Savi finalizer

// On the Savi side we're representing this as a 32-byte struct, and that's the
// amount that we'll stack-allocate when we make a new instance in Savi.
// So we need to make sure that in C++ it's no larger than those 32 bytes.
static_assert(sizeof(torch::Scalar) <= 32, "wrong size");
static inline torch::Scalar import_scalar(void* scalar) {
  // TODO: Is there an easier way to import?
  torch::Scalar scalar_new;
  memcpy(&scalar_new, scalar, sizeof(torch::Scalar));
  return scalar_new;
}

torch::TensorImpl* torch_ffi_tensor_from_data(
  void* from_data,
  int64_t* dimension_list,
  size_t dimension_count,
  size_t element_size_in_bytes,
  int element_kind
) {
  torch::Tensor tensor = torch::zeros(
    torch::IntArrayRef(dimension_list, dimension_count),
    torch::ScalarType(element_kind)
  );
  void *tensor_data = tensor.data_ptr();
  memcpy(tensor_data, from_data, tensor.numel() * element_size_in_bytes);
  return export_tensor(tensor);
}

static size_t _tensor_into_data(
  const torch::Tensor& tensor,
  uint8_t* into_data,
  size_t into_data_offset
) {
  if (tensor.is_contiguous()) {
    size_t memcpy_size = tensor.numel() * tensor.element_size();
    memcpy(into_data + into_data_offset, tensor.data_ptr(), memcpy_size);
    return into_data_offset + memcpy_size;
  }

  auto dim_0_size = (size_t)tensor.sizes()[0];
  for (size_t i = 0; i < dim_0_size; i++) {
    into_data_offset =
      _tensor_into_data(tensor.select(0, i), into_data, into_data_offset);
  }

  return into_data_offset;
}

void torch_ffi_tensor_into_data(
  torch::TensorImpl* tensor_,
  uint8_t* into_data
) {
  torch::Tensor tensor = import_tensor(tensor_);
  _tensor_into_data(tensor, into_data, 0);
}

int64_t torch_ffi_tensor_itemsize(torch::TensorImpl* tensor) {
  return tensor->itemsize();
}

int64_t torch_ffi_tensor_numel(torch::TensorImpl* tensor) {
  return tensor->numel();
}

int64_t torch_ffi_tensor_ndimensions(torch::TensorImpl* tensor) {
  return tensor->dim();
}

torch::TensorImpl* torch_ffi_tensor_reshape(
  torch::TensorImpl* tensor,
  int64_t* dimension_list,
  size_t dimension_count
) {
  return export_tensor(import_tensor(tensor).reshape(
    torch::IntArrayRef(dimension_list, dimension_count)
  ));
}

torch::TensorImpl* torch_ffi_tensor_select(
  torch::TensorImpl* tensor,
  int64_t dimension,
  int64_t index
) {
  return export_tensor(import_tensor(tensor).select(dimension, index));
}

void torch_ffi_scalar_init_u8(torch::Scalar* scalar, uint8_t value) {
  *scalar = torch::Scalar(value);
}

void torch_ffi_scalar_init_i8(torch::Scalar* scalar, int8_t value) {
  *scalar = torch::Scalar(value);
}

void torch_ffi_scalar_init_i16(torch::Scalar* scalar, int16_t value) {
  *scalar = torch::Scalar(value);
}

void torch_ffi_scalar_init_i32(torch::Scalar* scalar, int32_t value) {
  *scalar = torch::Scalar(value);
}

void torch_ffi_scalar_init_i64(torch::Scalar* scalar, int64_t value) {
  *scalar = torch::Scalar(value);
}

void torch_ffi_scalar_init_f32(torch::Scalar* scalar, float value) {
  *scalar = torch::Scalar(value);
}

void torch_ffi_scalar_init_f64(torch::Scalar* scalar, double value) {
  *scalar = torch::Scalar(value);
}

void torch_ffi_scalar_init_bool(torch::Scalar* scalar, bool value) {
  *scalar = torch::Scalar(value);
}

torch::TensorImpl* torch_ffi_tensor_add_scalar(torch::TensorImpl* tensor, void* scalar, void* alpha) {
  return export_tensor(
    import_tensor(tensor).add(import_scalar(scalar), import_scalar(alpha))
  );
}

torch::TensorImpl* torch_ffi_tensor_mul_scalar(torch::TensorImpl* tensor, void* scalar) {
  return export_tensor(
    import_tensor(tensor).mul(import_scalar(scalar))
  );
}

torch::TensorImpl* torch_ffi_tensor_sub_scalar(torch::TensorImpl* tensor, void* scalar, void* alpha) {
  return export_tensor(
    import_tensor(tensor).sub(import_scalar(scalar), import_scalar(alpha))
  );
}

} // end of extern "C"
