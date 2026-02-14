#include "ops.h"
#include <stdio.h>
#include <string.h>

#define AT_DISPATCH_FLOATING_TYPES(dtype, name, ...)                           \
  switch (dtype) {                                                             \
  case DTYPE_FLOAT32: {                                                        \
    typedef float scalar_t;                                                    \
    __VA_ARGS__                                                                \
    break;                                                                     \
  }                                                                            \
  case DTYPE_FLOAT64: {                                                        \
    typedef double scalar_t;                                                   \
    __VA_ARGS__                                                                \
    break;                                                                     \
  }                                                                            \
  default:                                                                     \
    fprintf(stderr, "Unsupported dtype: %d\n", dtype);                         \
    break;                                                                     \
  }

static int check_broadcast_compatible(const Tensor *a, const Tensor *b) {
  if (!a || !b)
    return 0;

  if (a->ndim != b->ndim)
    return 0;

  for (int i = 0; i < a->ndim; ++i) {
    if (a->shape[i] != b->shape[i])
      return 0;
  }

  return 1;
}

void tensor_add_out(Tensor* out, Tensor* a, Tensor* b) {
  if (!out || !a || !b)
    return;

  size_t numel = tensor_numel(a);

  AT_DISPATCH_FLOATING_TYPES(a->storage->dtype, "add", {
    scalar_t *a_data = (scalar_t *)storage_data(a->storage);
    scalar_t *b_data = (scalar_t *)storage_data(b->storage);
    scalar_t *out_data = (scalar_t *)storage_data(out->storage);

    for (size_t i = 0; i < numel; i++) {
      out_data[i] = a_data[i] + b_data[i];
    }
  });
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
  if (!check_broadcast_compatible(a, b)) {
    fprintf(stderr, "Add: incompatible shapes\n");
    return NULL;
  }

  if (a->storage->device.type != DEVICE_CPU ||
      b->storage->device.type != DEVICE_CPU) {
    fprintf(stderr, "CPU operations only\n");
    return NULL;
  }

  Tensor* result =
      tensor_empty(a->shape, a->ndim, a->storage->dtype, a->storage->device);
  if (!result)
    return NULL;

  tensor_add_out(result, a, b);
  return result;
}


