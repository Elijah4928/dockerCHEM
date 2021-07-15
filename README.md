# dockerCHEM
dockerfile for ml4chem and graphdot python packages(available through pypi)



#test.py
Was writen in mind of goal of sending data to docker containers that would be used to retrain models, or send through existing ones
Buildon after dockerfile build of ml4chem/graphdot and test is successful

NOTE docker run -it --name.... for some reason doesn't like showing the packages/dependencies(python main.py -> after building in docker container, root:python --version results in command not found)
-> python main2.py
-> wait for dockerfile to build
-> $ exit
-> docker run -it installtest bash
-> $:src/
-> src folder in docker container now contains the files in which main.py was in
