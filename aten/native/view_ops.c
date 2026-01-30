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