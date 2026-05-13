#define PY_SSIZE_T__CLEAN
#include <Python.h>

static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyMethodDef ctools_methods[] = {
    {"system", spam_system, METH_VARARGS,
     "Execute a shell command."},
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
