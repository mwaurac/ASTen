#ifndef C10_TENSOR_H
#define C10_TENSOR_H

#include "storage.h"

#include <stddef.h>

typedef struct Tensor Tensor;
typedef struct AutogradMeta AutogradMeta;

struct AutogradMeta {
    Tensor* grad;
    void (*grad_fn)(Tensor* self);
    Tensor** next_functions;
    int num_next;
};

struct Tensor {
    Storage* storage;
    size_t* shape;
    size_t* strides;
    size_t ndim;
    
    size_t offset;
    int requires_grad;
    AutogradMeta* autograd;
};

Tensor* tensor_empty(size_t* shape, size_t ndim, DType dtype, Device device);
Tensor* tensor_ones(size_t* shape, size_t ndim, DType dtype, Device device);
Tensor* tensor_zeros(size_t* shape, size_t ndim, DType dtype, Device device);
Tensor* tensor_from_data(void* data, size_t* shape, size_t ndim, DType dtype, Device);

void tensor_retain(Tensor* tensor);
void tensor_release(Tensor* tensor);

size_t tensor_numel(Tensor* tensor);
int tensor_is_contiguous(Tensor* tensor);
size_t tensor_element_offset(Tensor* tensor, size_t* indeces);

void tensor_set_requires_grad(Tensor* tensor, int requires_grad);
Tensor* tensor_grad(Tensor* tensor);
void tensor_zero_grad(Tensor* tensor);

void tensor_compute_contiguous_strides(size_t* strides, size_t* shape, size_t ndim);

#endif