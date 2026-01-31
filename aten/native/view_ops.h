#ifndef ATEN_VIEW_OPS_H
#define ATEN_VIEW_OPS_H

#include "../../c10/core/tensor.h"

Tensor* tensor_view(Tensor* self, size_t* new_shape, size_t new_ndim);
Tensor* tensor_contiguous(Tensor* self);
Tensor* tensor_reshape(Tensor* self, size_t* new_shape, size_t new_ndim);
Tensor* tensor_permute(Tensor* self, size_t* dims, size_t ndim);
#endif