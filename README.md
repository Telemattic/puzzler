# puzzler

# dependencies:
#  python -m venv env
#  <activate env>
#  pip3 install ...
#      cachetools
#      numpy
#      opencv-python
#      scipy
#      palettable
#      networkx
#      tqdm
#      pycairo
#      Pillow
#      PyOpenGL
#      pyopengltk
#      scikit-image
#      matplotlib

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

  see puzzler.align.MultiTargetError for a better error measurement,
  better (correct I think) match is

   C11:1=C12:3,D11:2=D12:0,C12:2=D12:1


Profiling:

  # note, -s deals with python subprocesses which is vital in a virtualenv
  # note, -- makes sure that the python command line isn't parsed as part of the py-spy arguments
  
  py-spy record -f speedscope -o prof.ss -s -- python app.py -p 100.json align

Processing pipeline:

  app.yy -p NAME.json init
  app.py -p NAME.json scan add  (repeat as necessary)
  app.py -p NAME.json points update
  app.py -p NAME.json lint update
  app.py -p NAME.json features update
  app.py -p NAME.json tabs -o tabs_NAME.csv -n 28  # compute error for all possible tab pairs, and do it in parallel
  scripts/tabpairs_csv_to_mmap.py -p NAME.json -i tabs_NAME.csv -o tabs_NAME.mmap
