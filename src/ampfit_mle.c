#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


// Forward function declarations.
static PyObject *compute_phi_array(PyObject *self, PyObject *args); 
static PyObject *compute_lambda_matrix(PyObject *self, PyObject *args); 


// Boilerplate: function list.
static PyMethodDef methods[] = {
  { "compute_phi_array", compute_phi_array, METH_VARARGS, ""},  
  { "compute_lambda_matrix", compute_lambda_matrix, METH_VARARGS, ""},
  { NULL, NULL, 0, NULL } /* Sentinel */
};


// Boilerplate: Module initialization.
PyMODINIT_FUNC initampfit_mle_c(void) {
  (void) Py_InitModule("ampfit_mle_c", methods);
  import_array();
}


/*****************************************************************************
 * Local helper functions and macros. 
 *****************************************************************************/

#define MIN(a,b) (((a)<(b))?(a):(b))


/*****************************************************************************
 * Computing the lambda matrix. 
 *****************************************************************************/

// calc_lam_ij computes a single entry in the lambda matrix. 
inline npy_float64 calc_lam_ij(npy_int64    i0,  // Index of first pulse.
                               npy_int64    i1,  // Index of second pulse. 
                               npy_int64    n_r, // Block data length.
                               npy_float64 *p,   // Pulse shape.
                               npy_int64    n_p  // Pulse length. 
                               ) {
  // Note that it's required that i1 >= i0. 
  npy_int64 di = i1 - i0;
  npy_int64 imax = MIN(n_p - di, n_r - i0 - di);

  npy_float64 lamij = 0;
  npy_int64 i;

  for(i = 0; i < imax; ++i) {
    lamij += p[i] * p[i + di];
  }

  return lamij; 
}

// compute_lambda_matrix computes the lambda matrix. 
static PyObject *compute_lambda_matrix(PyObject *self, PyObject *args) {
  
  npy_int64 n_r;
  PyArrayObject *py_p, *py_inds, *py_lam;

  if(!PyArg_ParseTuple(args, "lO!O!O!",
                       &n_r,                    // Length of block. 
                       &PyArray_Type, &py_p,    // Pulse shape. 
                       &PyArray_Type, &py_inds, // Pulse indices. 
                       &PyArray_Type, &py_lam   // Lambda array. 
                       )) {
    return NULL;
  }
  
  npy_float64 *p = (npy_float64*)PyArray_DATA(py_p);
  npy_int64 *inds = (npy_int64*)PyArray_DATA(py_inds);
  npy_float64 *lam = (npy_float64*)PyArray_DATA(py_lam);
  
  npy_int64 n_p = py_p->dimensions[0];
  npy_int64 n_inds = py_inds->dimensions[0];
  
  npy_int64 i, j;
  
  for(i = 0; i < n_inds; ++i) {
    for(j = i; j < n_inds; ++j) {
      lam[n_inds * i + j] = lam[n_inds * j + i] = 
        calc_lam_ij(inds[i], inds[j], n_r, p, n_p);
    }
  }
  
  Py_RETURN_NONE;
}


/*****************************************************************************
 * Computing the phi array.
 *****************************************************************************/

// calc_phi_i computes a single element of the phi array. 
inline npy_float64 calc_phi_i(npy_int64    idx, // Pulse index.  
                              npy_float64 *r,   // Raw data. 
                              npy_int64    n_r, // Raw data length
                              npy_float64 *p,   // Pulse shape.
                              npy_int64    n_p  // Pulse length. 
                              ) {
  npy_int64 i;
  npy_int64 imax = MIN(n_p, n_r - idx);
  npy_float64 phii = 0;
  
  for(i = 0; i < imax; ++i) {
    phii += p[i] * r[idx + i];
  }
  
  return phii; 
}

// compute_phi_array computes the phi array.
static PyObject *compute_phi_array(PyObject *self, PyObject *args) {
    
  PyArrayObject *py_r, *py_p, *py_inds, *py_phi;

  if(!PyArg_ParseTuple(args, "O!O!O!O!",
                       &PyArray_Type, &py_r,    // Raw data.
                       &PyArray_Type, &py_p,    // Pulse shape.
                       &PyArray_Type, &py_inds, // Pulse indices.
                       &PyArray_Type, &py_phi   // Pulse indices.
                       )) {
    return NULL;
  }
  
  npy_float64 *r = (npy_float64*)PyArray_DATA(py_r);
  npy_float64 *p = (npy_float64*)PyArray_DATA(py_p);
  npy_int64 *inds = (npy_int64*)PyArray_DATA(py_inds);
  npy_float64 *phi = (npy_float64*)PyArray_DATA(py_phi);
  
  npy_int64 n_r = py_r->dimensions[0];
  npy_int64 n_p = py_p->dimensions[0];
  npy_int64 n_inds = py_inds->dimensions[0];

  npy_int64 i;

  for(i = 0; i < n_inds; ++i) {
    phi[i] = calc_phi_i(inds[i], r, n_r, p, n_p);
  }
  Py_RETURN_NONE;
}
