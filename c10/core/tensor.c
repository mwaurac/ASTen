#include "tensor.h"
#include "storage.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void tensor_compute_contiguous_strides(size_t *strides, size_t *shape, size_t ndim) {
    if (ndim == 0) return;
    
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >=0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

Tensor* tensor_empty(size_t *shape, size_t ndim, DType dtype, Device device) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;
    
    size_t numel = 1;
    for (size_t i = 0; i < ndim; i++) {
        numel *= shape[i];
    }
    
    tensor->storage = storage_new(numel, dtype, device);
    if (!tensor->storage) {
        free(tensor);
        return NULL;
    }
    
    tensor->shape = (size_t*)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->shape || !tensor->strides) {
        storage_release(tensor->storage);
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    tensor_compute_contiguous_strides(tensor->strides, shape, ndim);
    
    tensor->ndim = ndim;
    tensor->offset = 0;
    tensor->requires_grad = 0;
    tensor->autograd = NULL;
    
    return tensor;
}

Tensor* tensor_zeros(size_t *shape, size_t ndim, DType dtype, Device device) {
    return tensor_empty(shape, ndim, dtype, device);
}

Tensor* tensor_ones(size_t *shape, size_t ndim, DType dtype, Device device) {
    Tensor* tensor = tensor_empty(shape, ndim, dtype, device);
    if (!tensor) return NULL;
    
    size_t numel = tensor_numel(tensor);
    
    void* data = storage_data(tensor->storage);
    
    if (device.type == DEVICE_CPU) {
        switch (dtype) {
            case DTYPE_FLOAT32: {
                float* ptr = (float*)data;
                for (size_t i = 0; i < numel; i++) ptr[i] = 1.0f;
            }
            case DTYPE_FLOAT64: {
                float* ptr = (float*)data;
                for (size_t i = 0; i < numel; i++) ptr[i] = 1.0;
            }
            case DTYPE_INT32: {
                int32_t* ptr = (int32_t*)data;
                for (size_t i = 0; i < numel; i++) ptr[i] = 1;
            }
            case DTYPE_INT64: {
                int64_t* ptr = (int64_t*)data;
                for (size_t i = 0; i < numel; i++) ptr[i] = 1;
            }
            default: break;
        }
    }
    return tensor;
}

void tensor_retain(Tensor *tensor) {
    if (tensor && tensor->storage) {
        storage_retain(tensor->storage);
    }
}

void tensor_release(Tensor *tensor) {
    if (!tensor) return;
    
    storage_release(tensor->storage);
    free(tensor->shape);
    free(tensor->strides);
    
    if (tensor->autograd) {
        if (tensor->autograd->grad) {
            tensor_release(tensor->autograd->grad);
        }
        free(tensor->autograd->next_functions);
        free(tensor->autograd);
    }
    free(tensor);
}

size_t tensor_numel(Tensor* tensor) {
    if (!tensor) return 0;
    size_t numel = 1;
    
    for (size_t i = 0; i < tensor->ndim; i++) {
        numel *= tensor->shape[i];
    }
    return numel;
}

int tensor_is_contiguous(Tensor *tensor) {
    if (!tensor) return 0;

    size_t expected_stride = 1;
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        if (tensor->strides[i] != expected_stride) {
            return 0;
        }
        expected_stride *= tensor->shape[i];
    }
    return 1;
}

size_t tensor_element_offset(Tensor *tensor, size_t *indeces) {
    size_t offset = tensor->offset;
    for (size_t i = 0; i < tensor->ndim; i++) {
        offset += indeces[i] * tensor->strides[i];
    }
    return offset;
}

void tensor_set_requires_grad(Tensor *tensor, int requires_grad) {
    if (!tensor) return;
    
    tensor->requires_grad = requires_grad;
    
    if (requires_grad && !tensor->autograd) {
        tensor->autograd = (AutogradMeta*)calloc(1, sizeof(AutogradMeta));
    }
}

Tensor* tensor_grad(Tensor* tensor) {
    if (!tensor || !tensor->autograd) return NULL;
    return tensor->autograd->grad;
}

void tensor_zero_grad(Tensor *tensor) {
    if (!tensor || !tensor->autograd || !tensor->autograd->grad) return;
    
    tensor_release(tensor->autograd->grad);
    tensor->autograd->grad = NULL;
}