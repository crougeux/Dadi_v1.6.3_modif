C     -*- fortran -*-
C     This file is autogenerated with f2py (version:2)
C     It contains Fortran 77 wrappers to fortran functions.

      subroutine f2pywrapdlamch (dlamchf2pywrap, cmach)
      external dlamch
      character cmach
      double precision dlamchf2pywrap, dlamch
      dlamchf2pywrap = dlamch(cmach)
      end


      subroutine f2pywrapslamch (slamchf2pywrap, cmach)
      external wslamch
      character cmach
      real slamchf2pywrap, wslamch
      slamchf2pywrap = wslamch(cmach)
      end

