#pragma once


#include <iostream>
#include <map>     // maybe replaced with C++11 unordered map?
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace numeric {

typedef std::vector<double> VectorType;
static std::map< int,
    std::map< int, VectorType >
> wiab;

namespace integrate {

/**************************************************************************/ /**
Use Mathematica to calculate correct weightings in arbitrary precision
\code{.nb}
xcds[n_, i_] := xcds[n, i] = i - Floor[(n - 1)/2]
A[n_, x_] := A[n, x] = Table[x[n, i]^j, {j, 0, n - 1}, {i, 0, n - 1}]
H[n_, h_] :=
 H[n, h] = Table[If[i == j, h^j/j!, 0], {i, 0, n - 1}, {j, 0, n - 1}]
T[n_, a_] :=
 T[n, a] = Table[a^(m + 1)/(m + 1)!, {m, 0, n - 1}, {i, 0, 0}]
\[CurlyPhi][n_, x0_, h_, x_, f_ : f] :=
 Table[f[x0 + x[n, i] h], {i, 0, n - 1}, {j, 0, 0}]
\[CapitalPhi][n_, x0_, h_, x_, f_ : f] :=
 Transpose[\[CurlyPhi][n, x0, h, x, f]].Inverse[A[n, x]].Inverse[
   H[n, h]]
S[n_, x0_, h_, x_, a_, b_, f_ : f] :=
 Simplify[(\[CapitalPhi][n, x0t, ht, x, f].(T[n, bt] - T[n, at]))[[1,
    1]]] //. {x0t -> x0, ht -> h, at -> a, bt -> b}
intnumcds[n_, N_, a_, h_] := intnumcds[n, N, a, h] = Module[{},
    S[n, a + Floor[(n - 1)/2] h, h, xcds, -Floor[(n - 1)/2] h, 0]
     + If[Floor[(n - 1)/2] != (N - 1) - Ceiling[(n - 1)/2] ,
      +S[n, a + Floor[(n - 1)/2] h, h, xcds, 0, h/2] +
       S[n, a + (N - 1) h - Ceiling[(n - 1)/2] h, h, xcds, -(h/2), 0],
       0]
     + \!\(
\*UnderoverscriptBox[\(\[Sum]\), \(i = Floor[
\*FractionBox[\(n - 1\), \(2\)]] + 1\), \(\((N - 1)\) - Ceiling[
\*FractionBox[\(n - 1\), \(2\)]] - 1\)]\(S[n, a + i\ h, h, xcds, \(-
\*FractionBox[\(h\), \(2\)]\),
\*FractionBox[\(h\), \(2\)]]\)\)
     + S[n, a + (N - 1) h - Ceiling[(n - 1)/2] h, h, xcds, 0,
      Ceiling[(n - 1)/2] h]];
wiab[n_, Nx_] := wiab[n, Nx] = Module[{a, h},
    Table[
     Coefficient[intnumcds[n, Nx, a, h]/h,
      Flatten[\[CurlyPhi][Nx, a, h, xfds]][[k + 1]]], {k, 0, Nx - 1}]];

nmax = 25;
outfile =
  OpenWrite["C:/Users/Hypatia/Desktop/initwiab.cpp",
   PageWidth -> \[Infinity]];
WriteString[outfile, "wiab[0][0].resize(1,false);\n"];
WriteString[outfile,
  "wiab[0][0][0] = " <> ToString[nmax] <>
   ";\n\n"];
WriteString[outfile,
  "for (int n = 1; n <= (int)wiab[0][0][0]; n++)\n"];
WriteString[outfile, "    for (int N = n; N <= 2*n+1; N++)\n"];
WriteString[outfile, "        wiab[n][N].resize(N,false);\n\n"];

For[n = 1, n <= nmax, n++,
 For[Nx = n, Nx <= 2 n + 1, Nx++,
  WriteString[outfile,
   "wiab[" <> ToString[n] <> "][" <> ToString[Nx] <> "] <<= "];
  wi = wiab[n, Nx];
  For[i = 1, i <= Length[wi] - 1, i++,
   WriteString[outfile, ToString[N[wi[[i]], 20], CForm] <> ","]
   ];
  WriteString[outfile,
   ToString[N[wi[[Length[wi]]], 20], CForm] <> ";\n"];]]
Close[outfile]
\endcode
*******************************************************************************/
void cacheIntegrationCoefficients(void);

/***************************************************************************/ /*
 * @brief Calculate the integrated area under function f with arbitrary error
 *        scaling
 * @param[in] f Vector like template data type. Needs method size() and
 *              access operator[] for this template to work!
 * @param[in] h Interval width. The values contained in vector f are considered
 *              to be evaluated at sampling points spaced h apart.
 * @param[in] n specifies order of the integration algorithm. Integration works
 *              by interpolating f with a polynomial, n specifies the order of
 *              that polynomial. n=0: takes the highest order possible with
 *              given parameters.
 *                 n=2: trapezoid rule
 *                 n=3: Simpson's rule
 *                 n=4: Simpson's 3/8 rule
 *                 n=5: Boole's rule
 *                 n>5: Newton-Cotes formula of order n-1
 * @result \f[ \int\limits_{ a=f^{-1}(f_0) }^{ b=f^{-1} (f_{N-1}) }
           f(x) \mathrm{d}x \f$ where N is the length of vector \f$ \vec{f} =
           \sum\limits_{i=0}^{N-1} f\left((a+i\frac{b-a}{N-1}\right) \hat{e}_i
           \f]
 ******************************************************************************/
template<typename T_PREC, typename T_VEC> T_PREC newtonCotes(const T_VEC & f, T_PREC h, int n = 0 );
template<typename T_PREC, typename T_VEC> T_PREC trapezoid  (const T_VEC & f, T_PREC h);
template<typename T_PREC, typename T_VEC> T_PREC simpson    (const T_VEC & f, T_PREC h);

} // namespace integrate

} // namespace numeric



#include "integrate.cpp"
