# dockerCHEM
dockerfile for ml4chem and graphdot python packages(available through pypi)

#training.py -> ml4chem test file(fix dask distributed issues dependencies seem in order otherwise)
-> this repo needs "data" folder with "miniset.cvs" file and "3D_generation" folder
#graphtest.py -> graphdot test(gpu compatability issues with docker expect to be fixed when run through shifter, could also be pathing issues with cuda couldn't resolve)

main.py -> builds intermediate python docker which users could run main.py(can omit and automate through main.py perhaps)
-> also builds and pushes dockerfile to dockerhub to be pulled for shifter use

can change from dockerhub elijahg/4928 to new one


edit main.py:dockerhub is elijah4928/temp
