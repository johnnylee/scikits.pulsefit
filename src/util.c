#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


// Forward function declarations.
static PyObject *add_pulses(PyObject *self, PyObject *args); 


// Boilerplate: function list.
static PyMethodDef methods[] = {
  { "add_pulses", add_pulses, METH_VARARGS, ""},
  { NULL, NULL, 0, NULL } /* Sentinel */
};


// Boilerplate: Module initialization.
PyMODINIT_FUNC initutil_c(void) {
  (void) Py_InitModule("util_c", methods);
  import_array();
}

// add_pulses computes the phi array.
static PyObject *add_pulses(PyObject *self, PyObject *args) {
    
  PyArrayObject *py_inds, *py_amps, *py_p, *py_out;

  if(!PyArg_ParseTuple(args, "O!O!O!O!",
                       &PyArray_Type, &py_inds,   
                       &PyArray_Type, &py_amps,   
                       &PyArray_Type, &py_p,
                       &PyArray_Type, &py_out  
                       )) {
    return NULL;
  }
  
  npy_float64 *inds = (npy_float64*)PyArray_DATA(py_inds);
  npy_float64 *amps = (npy_float64*)PyArray_DATA(py_amps);
  npy_float64 *p    = (npy_float64*)PyArray_DATA(py_p);
  npy_float64 *out  = (npy_float64*)PyArray_DATA(py_out);
  
  npy_int64 n_inds = py_inds->dimensions[0];
  npy_int64 n_p    = py_p->dimensions[0];
  npy_int64 n_out  = py_out->dimensions[0];

  npy_int64 i, j, imax, idx0;
  npy_float64 idx, amp, cr, cl;
  
  // Add pulses. 
  for(j = 0; j < n_inds; ++j) {
    idx = inds[j];
    amp = amps[j];
    
    idx0 = ceil(idx);
    imax = (n_p - 1) < (n_out - idx0) ? (n_p - 1) : (n_out - idx0); // min
    
    cr = idx0 - idx;
    cl = 1 - cr;
    
    for(i = 0; i < imax; ++i) {
      out[i + idx0] += amp * (cl * p[i] + cr * p[i+1]);
    }
  }

  Py_RETURN_NONE;
}


