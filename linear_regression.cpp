#include <iostream>
#include <cmath>
#include <vector>

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

#ifdef __APPLE__
#include <Python/Python.h>
#else
#include <Python.h>
#endif

// An Implementation of the Linear Regression Algorithm in C++.
// Accessible as a Python Module.

#define space " "
#define epoch int(1000)

double hypothesis(const std::vector<double>& b_val, double x){
    return b_val[0] + b_val[1] * x;
}

std::vector<double> regression(const std::vector<double>& x, const std::vector<double>& y,
                               int epochs, double learning_rate){
    if(!epochs) epochs = epoch;
    double _m = 0;
    double _b = 0;
    std::vector<double> _values(2);
    int N = x.size();

    for(int i = 0; i < epochs; i++){
        _values[0] = _b, _values[1] = _m;
        double dm = 0;
        double db = 0;
        double cost = 0;

        for(int j = 0; j < N; j++){
            double p = hypothesis(_values, x[j]);
            cost += pow(y[j] - p, 2);
            dm += (-2.0 / N) * (x[j] * (y[j] - p));
            db += (-2.0 / N) * (y[j] - p);
        }

        cost /= N;
        _m = _m - (learning_rate * dm);
        _b = _b - (learning_rate * db);

        if ((i + 1) % 100 == 0)
            std::cout << "Epoch: " << (i + 1) << " Cost: " << cost << std::endl;
    }
    std::vector<double> result(2);
    result[0] = _m, result[1] = _b;
    return result;
}

static PyObject * fit(PyObject * self, PyObject * args){
    PyObject *x;
    PyObject *y;
    double learning_rate;
    int epochs;
    int N;

    if(!PyArg_ParseTuple(args, "00di0i", &x, &y, &epochs, &learning_rate, &N)){
        return nullptr;
    }

    std::vector<double> _x(N), _y(N);
    for(int i = 0; i < N; i++){
        _x[i] = PyFloat_AsDouble(PyList_GetItem(x, (Py_ssize_t)i));
        _y[i] = PyFloat_AsDouble(PyList_GetItem(y, (Py_ssize_t)i));
    }

    std::vector<double> _result = regression(_x, _y, epochs, learning_rate);
    PyObject *result = PyTuple_New(2);
    for(int i = 0; i < 2; i++){
        PyTuple_SetItem(result, i, PyFloat_FromDouble(_result[i]));
    }

    return Py_BuildValue("s", result);
}

static PyMethodDef linreg_methods[] = {
        {"fit", (PyCFunction) fit, METH_VARARGS, "Linear Regression"},
        {nullptr}
};

#ifdef PY3K
static struct PyModuleDef linear_regression = {
        PyModuleDef_HEAD_INIT,
        "linear_regress"
        "Linear Regression",
        -1,
        linreg_methods
};

PyMODINIT_FUNC PyInit_linear_regression(void){
    return PyModule_Create(&linear_regression);
}
#else
PyMODINIT_FUNC linear_regression(){
    Py_InitModule3("linear_regress", linreg_methods, "Linear Regression");
}
#endif


