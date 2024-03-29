#line 1 "scipy/special/_logit.c.src"

/*
 *****************************************************************************
 **       This file was autogenerated from a template  DO NOT EDIT!!!!      **
 **       Changes should be made to the original source (.src) file         **
 *****************************************************************************
 */

#line 1
/*-*-c-*-*/

/*
 * ufuncs to compute logit(p) = log(p/(1-p)) and
 * expit(x) = 1/(1+exp(-x))
 */

#include <Python.h>
#include <math.h>

#include "numpy/npy_math.h"
#include "_logit.h"

/*
 * Inner loops for logit and expit
 */

#line 22

npy_float logitf(npy_float x)
{
    x /= 1 - x;
    return npy_logf(x);
}

npy_float expitf(npy_float x)
{
    if (x > 0) {
        x = npy_expf(x);
        return x / (1 + x);
    }
    else {
        return 1 / (1 + npy_expf(-x));
    }
}


#line 22

npy_double logit(npy_double x)
{
    x /= 1 - x;
    return npy_log(x);
}

npy_double expit(npy_double x)
{
    if (x > 0) {
        x = npy_exp(x);
        return x / (1 + x);
    }
    else {
        return 1 / (1 + npy_exp(-x));
    }
}


#line 22

npy_longdouble logitl(npy_longdouble x)
{
    x /= 1 - x;
    return npy_logl(x);
}

npy_longdouble expitl(npy_longdouble x)
{
    if (x > 0) {
        x = npy_expl(x);
        return x / (1 + x);
    }
    else {
        return 1 / (1 + npy_expl(-x));
    }
}



