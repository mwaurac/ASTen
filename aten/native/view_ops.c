#include "view_ops.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor* tensor_view(Tensor *self, size_t *new_shape, size_t new_ndim) {
    if (!self) return NULL;
    
    size_t old_numel = tensor_numel(self);
    size_t new_numel = 1;
    
    for (size_t i = 0; i < new_ndim; i++) {
        new_numel *= new_shape[i];
    }
    
    if (old_numel != new_numel) {
        fprintf(stderr, "View: shape mismatch, %zu vs %zu elements\n", old_numel, new_numel);
        return NULL;
    }
    
    if (!tensor_is_contiguous(self)) {
        fprintf(stderr, "View: tensor must be contiguous\n");
        return NULL;
    }
    
    Tensor* result = (Tensor*)malloc(sizeof(Tensor));
    if (!result) return NULL;
    
    result->storage = self->storage;
    storage_retain(result->storage);

    result->shape = (size_t*)malloc(new_ndim * sizeof(size_t));
    result->strides = (size_t*)malloc(new_ndim * sizeof(size_t));


    if (!result->shape ||!result->strides) {
        storage_release(result->storage);
        free(result->shape);
        free(result->strides);
        free(result);
        return NULL;
    }
    
    memcpy(result->shape, new_shape, new_ndim * sizeof(size_t));

    tensor_compute_contiguous_strides(result->strides, new_shape, new_ndim);

    result->ndim = new_ndim;
    result->offset = self->offset;
    result->requires_grad = self->requires_grad;
    result->autograd = NULL;

    return result;
}

Tensor* tensor_contiguous(Tensor* self) {
    if (!self) return NULL;

    if (tensor_is_contiguous(self)) {
        tensor_retain(self);
        return self;
    }

    fprintf(stderr, "View: contiguous tensor is not contiguous\n");
    Tensor* result = tensor_empty(self->shape, self->ndim, self->storage->dtype, self->storage->device);
    if (!result) return NULL;

    size_t numel = tensor_numel(self);
    void* src_data = storage_data(self->storage);
    void* dst_data = storage_data(result->storage);
    size_t elem_size = storage_element_size(self->storage);

    // TODO: maybe optimise
    size_t* indices = (size_t*)calloc(self->ndim, sizeof(size_t));

    for (size_t i = 0; i < numel; i++) {
        size_t src_offset = tensor_element_offset(self, indices);

        memcpy((char*)dst_data + i * elem_size,
               (char*)src_data + src_offset * elem_size,
               elem_size);

        // Increment indices
        for (int d = self->ndim - 1; d >= 0; d--) {
            indices[d]++;
            if (indices[d] < self->shape[d]) break;
            indices[d] = 0;
        }
    }
}

Tensor* tensor_reshape(Tensor* self, size_t* new_shape, size_t new_ndim) {
    if (!tensor_is_contiguous(self)) {
        Tensor* contiguous = tensor_contiguous(self);
        Tensor* result = tensor_view(contiguous, new_shape, new_ndim);
        tensor_release(contiguous);
        return result;
    }
    return tensor_view(self, new_shape, new_ndim);
}
