# puzzler

# dependencies:
#
#   pip3 install pycairo
#   pip3 install PyOpenGL
#   pip3 install pyopengltk
#   pip3 install bezier==2023.7.28     # avoid pulling in a dependency on numpy 2.0
#   pip3 install scikit-image

# 1000.json problems:
#   1. O28 (bad match to X38, Y37)
#   2. Q35 (bad match to Y4, Z5)
#   3. O27 (bad match to O28, Y36)

Todo:

* revise EdgeAligner to do ICP in order to get better alignment before
  measuring fit error, for example see 1000.json and fit of edge
  "A36:0,1=A37:0,0" (correct) vs "A36:0,1=A10:0,0" (incorrect, but
  better scoring)

* have a complete solve (tab and edge correspondence) for each puzzle
  to use as a reference to detect mistakes promptly

* revise RaftAligner to do ICP in order to get better alignment before
  measuring fit error, for example 1000.json and fit of
  "F17:3=F16:0,1,G30:0=G29:1,0" (which is admittely a false match...)
  "F17:3=F16:0,1,E17:0=E16:3,0" (correct match)

* abysmal fit, but not scored that poorly because fit only considered
  between edges that are proximate, deep overlaps are ignored
  
   C11:0=C12:0,D11:2=D12:0,C12:3=D12:1,C11:1=D11:1

  see puzzler.align.MultiTargetError for a better error measurement
