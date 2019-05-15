#include <Python.h>
#include <math.h>

/* Practice using C extensions to speed up information entropy calculations since it is a bottleneck.
 * A small speedup was noticed but nothing too great unfortunately.
 */


static double get_entropy(double * p_values, int size){
    double sum = 0.0;

    for (int i = 0; i < size; i++){
        if (p_values[i] > 0.0){
            sum += p_values[i] * log2(p_values[i]);
        }
    }
    sum /= (double) size;
    sum -= log2(size);
    return -1.0 * sum;
}


static PyObject * entropy(PyObject *self, PyObject *args)
{
  PyObject* input;
  PyArg_ParseTuple(args, "O", &input);

  int size = PyList_Size(input);

  double list[size];

  for(int i = 0; i < size; i++) {
    list[i] = PyFloat_AS_DOUBLE(PyList_GET_ITEM(input, i));
  }
  return PyFloat_FromDouble(get_entropy(list, size));
}


// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "entropy", entropy, METH_VARARGS, "Calculates information entropy" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef metricModule = {
    PyModuleDef_HEAD_INIT,
    "metricModule",
    "Test Module",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_metricModule(void)
{
    return PyModule_Create(&metricModule);
}