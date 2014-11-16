#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#include "moving_median.h"

// Forward function declarations.
static PyObject *next_block(PyObject *self, PyObject *args); 

// Boilerplate: function list.
static PyMethodDef methods[] = {
  { "next_block", next_block, METH_VARARGS, ""},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

// Boilerplate: Module initialization.
PyMODINIT_FUNC initblockident_median_c(void) {
  (void) Py_InitModule("blockident_median_c", methods);
  import_array();
}

// next_block: Find the next block using a moving median filter to
// establish the baseline level.
static PyObject *next_block(PyObject *self, PyObject *args) {

  // Declare variables. 
  npy_int64 i_start, filt_len, pad_post, max_len;
  npy_float64 th;
  PyArrayObject *py_r, *py_return_inds;
  
  // Return values. 
  npy_int64 i0, i1;  // Returned in py_return_inds array.
  npy_float64 b = 0; // The returned offset.

  // Parse arguments.
  if(!PyArg_ParseTuple(args, "lllldO!O!",
                       &i_start,
                       &filt_len,
                       &pad_post,
                       &max_len,
                       &th,
                       &PyArray_Type, &py_r,
                       &PyArray_Type, &py_return_inds)) {
    return NULL;
  }

  // Get underlying arrays for numpy arrays.
  npy_float64 *r = (npy_float64*)PyArray_DATA(py_r);
  npy_int64 *return_inds = (npy_int64*)PyArray_DATA(py_return_inds);
  npy_int64 r_len = py_r->dimensions[0];

  // The moving-median filter.
  mm_handle *mm = mm_new(filt_len);
  
  // Initialize median filter.
  for(i0 = i_start; i0 < i_start + filt_len; ++i0) {
    mm_insert_init(mm, r[i0]);
  }
  
  // Find the starting index. 
  for(; i0 < r_len; ++i0) {
    mm_update(mm, r[i0]);
    b = mm_get_median(mm);
    if((r[i0] - b) > th) {
      break;
    }
  }
  
  mm_free(mm);
  
  // Find the ending index. 
  long n_below = 0; // Number of samples below threshold. 
  npy_int64 di_max = max_len + pad_post;
  
  for(i1 = i0 + 1;
      (i1 < r_len) &&  ((i1 - i0) < di_max) && (n_below < pad_post);
      ++i1) {
    if(r[i1] - b > th) {
      n_below = 0;
    } else {
      ++n_below;
    } 
  }
  
  // Move i1 back padding amount. 
  i1 -= pad_post;
  
  return_inds[0] = i0;
  return_inds[1] = i1;
  
  return PyFloat_FromDouble(b);
}
