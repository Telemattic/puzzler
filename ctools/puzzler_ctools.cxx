#define PY_SSIZE_T__CLEAN

#ifdef _DEBUG
  #undef _DEBUG
  #include <Python.h>
  #include <memory>
  #define _DEBUG
#else
  #include <Python.h>
  #include <memory>
#endif

#include "nearest_point.h"
#include <numpy/arrayobject.h>

std::shared_ptr<PyObject> make_py_shared(PyObject* obj) {
    return std::shared_ptr<PyObject>(obj, [](PyObject* o) {
        Py_XDECREF(o);
    });
}

static PyObject*
py_compute_nearest_point_image(PyObject* self, PyObject* args)
{
    BBox bbox;
    PyObject* pointsObject;
    if (!PyArg_ParseTuple(args, "((ii)(ii))O", &bbox.ll.x, &bbox.ll.y, &bbox.ur.x, &bbox.ur.y, &pointsObject))
        return NULL;

    if (0 != PyArray_ImportNumPyAPI())
        return NULL;

    int requirements = NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED|NPY_ARRAY_ENSUREARRAY;
    pointsObject = PyArray_FromAny(pointsObject, PyArray_DescrFromType(NPY_INT32), 2, 2, requirements, NULL);
    if (!pointsObject)
        return NULL;

    // we now own a reference on pointsObject, Py_DECREF(pointsObject) is necessary before exiting
    auto pointsObject_ptr = make_py_shared(pointsObject);

    auto points = reinterpret_cast<PyArrayObject*>(pointsObject);
    if (PyArray_DIM(points, 1) != 2) {
        PyErr_SetString(PyExc_ValueError, "points array must be [n,2]");
        return NULL;
    }

    // these really amount to checking we got C-contiguous layout
    if (PyArray_STRIDE(points,0) != 8) {
        PyErr_SetString(PyExc_ValueError, "points array expected to have stride of 8");
        return NULL;
    }
    
    if (PyArray_STRIDE(points,1) != 4) {
        PyErr_SetString(PyExc_ValueError, "points array expected to have stride of 8");
        return NULL;
    }

    const auto width = bbox.ur.x - bbox.ll.x;
    const auto height = bbox.ur.y - bbox.ll.y;

    auto n_points = static_cast<int>(PyArray_DIM(points,0));
    auto points_data = reinterpret_cast<const Point*>(PyArray_DATA(points));

    const npy_intp dims[2] = {height, width};
    auto image_object = make_py_shared(PyArray_SimpleNew(2, dims, NPY_INT32));
    if (!image_object)
        return NULL;

    auto image_data = reinterpret_cast<int32*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(image_object.get())));
    std::fill(image_data, image_data + width*height, n_points);

    auto dist_object = make_py_shared(PyArray_SimpleNew(2, dims, NPY_FLOAT64));
    if (!dist_object)
        return NULL;

    auto dist_data = reinterpret_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dist_object.get())));

    compute_nearest_point_image(bbox, n_points, points_data, image_data, dist_data);

    return Py_BuildValue("OO", dist_object.get(), image_object.get());
}

static PyMethodDef puzzbin_methods[] = {
    {"compute_nearest_point_image", py_compute_nearest_point_image, METH_VARARGS,
     "Compute the nearest point image, internal implementation of python function."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef puzzbin_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "puzzbin",
    .m_methods = puzzbin_methods
};

extern "C" PyMODINIT_FUNC
PyInit_puzzbin(void)
{
    return PyModuleDef_Init(&puzzbin_module);
}
