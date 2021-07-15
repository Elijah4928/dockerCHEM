FROM ubuntu:20.04

WORKDIR /src

#needed for cuda installation
ENV TZ=America/Los_Angeles

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update
RUN apt-get -y install sudo
RUN apt-get -y install git
RUN apt-get -y install wget



RUN sudo apt -y install build-essential
RUN sudo apt-get -y install manpages-dev
RUN sudo apt -y install python3-pip


#install python below
RUN sudo apt-get -y install software-properties-common
RUN sudo add-apt-repository ppa:deadsnakes/ppa
RUN sudo apt-get -y install python3.8
###

###
RUN python3 -m pip install ml4chem
###


#cuda tookit installation reference https://developer.nvidia.com/cuda-downloads
#linux -> x86_64 -> Ubuntu -> 20.04 -> deb local

#cleanup if broken installation beforehand
#RUN sudo apt-get purge nvidia*
#RUN sudo apt-get autoremove
#RUN sudo apt-get autoclean
#RUN sudo rm -rf /usr/local/cuda*

#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#RUN sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#RUN wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
#RUN sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb
#RUN sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
#RUN sudo apt-get update
#DEBIAN_FRONTED=noninteractive(disables all settings that require input and default to first choice)
#RUN DEBIAN_FRONTEND=noninteractive apt-get -y --show-progress install cuda
#end of cuda toolkit

#rest of cuda drivers installation https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html





#end




#install visual studios c++(needed for pycuda)
RUN apt-get update
RUN sudo apt -y install software-properties-common apt-transport-https wget
RUN wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
RUN sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
RUN sudo apt -y install code


#RUN pip -y install graphdot


COPY . /src
