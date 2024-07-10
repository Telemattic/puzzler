# puzzler

# dependencies:
#
#   pip3 install pycairo
#   pip3 install PyOpenGL
#   pip3 install pyopengltk

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
