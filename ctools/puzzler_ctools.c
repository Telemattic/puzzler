#define PY_SSIZE_T__CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject*
spam_system(PyObject* self, PyObject* args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyObject*
compute_nearest_point_image(PyObject* self, PyObject* args)
{
    int x0, y0, x1, y1;
    PyObject* pointsObject;
    if (!PyArg_ParseTuple(args, "((ii)(ii))O", &x0, &y0, &x1, &y1, &pointsObject))
        return NULL;

    int requirements = NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_ENSUREARRAY;
    pointsObject = PyArray_FromAny(pointsObject, PyArray_DescrFromType(NPY_INT32), 2, 2, requirements, NULL);
    if (pointsObject == NULL)
        return NULL;

    PyArrayObject* points = (PyArrayObject*) pointsObject;
    if (PyArray_DIM(points, 0) != 2) {
        PyErr_SetString(PyExc_ValueError, "points array must be [n,2]");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyMethodDef ctools_methods[] = {
    {"system", spam_system, METH_VARARGS,
     "Execute a shell command."},
    {"_compute_nearest_point_image", compute_nearest_point_image, METH_VARARGS,
     "Compute the nearest point image, internal implementation of python function."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef ctools_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "ctools",
    .m_methods = ctools_methods
};

PyMODINIT_FUNC
PyInit_ctools(void)
{
    return PyModuleDef_Init(&ctools_module);
}
