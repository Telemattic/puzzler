#define PY_SSIZE_T__CLEAN
#include <Python.h>

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
    PyObject* points;
    if (!PyArg_ParseTuple(args, "((ii)(ii))O", &x0, &y0, &x1, &y1, &points))
        return NULL;
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
