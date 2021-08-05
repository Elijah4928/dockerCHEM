  
import sys
import os

#reuse this code, after installing/testing of dockerfile
#basis for sending data + training command to docker/shifter

def send_data(file, updating):
	if updating:
		print("updating")
	else:
		print("calculating\n")
		containerID = "shifter"
		print("docker cp " + file + " " + containerID + ":" + file)
		os.system("docker cp " + file + " " + containerID + ":" + file)





def check_data(new_file):
	directory = new_file
	updating = False
	for file in os.listdir(directory):
		#request standardized method
		if file.endswith(".csv") or file.endswith(".pickle") or file.endswith(".pkl") or file.endswith(".p"):
			updating = True


	directory = os.getcwd()
	for root, subdirectories, files in os.walk(directory):
		for subdirectory in subdirectories:
			print(os.path.join(root, subdirectory))
		for file in files:
			print(os.path.join(root, file))


	send_data(new_file, updating)




def create(name, folderpath, push=True):
	#should name image same as 'dockerhub/repository'
	cmd = "docker build -t " + name + " " + folderpath
	os.system(cmd)

	if push:
		os.system("docker push " + name)
	run(name, folderpath)

def run(name, path):
	os.system("docker run -it " + name + " " + path)
	copy(name, path)

def copy(name, path, run=True):
	#docker cp localpath containername:'newfile/dir name'
	os.system("docker cp " + path + " " + name + ":" + name)

def start(name):
	os.system("docker start -i " + name):



def main(argv):
	#start in master folder(dockers)
	# =>(implies) python main.py
	#push to later pull for shifter
	path = os.getcwd() + "/shifter"
	create("elijahg/4928", path, push=False)

	#intermediate ubuntu with python
	#might be able to cut out intermediate docker and just assume python capability on user end if not
	#user can use intermediate docker to run this python script
	path2 = os.getcwd() + "/dock"
	create("elijahg/4928", path2, push=False)

"""
	#launch shifter
	account_name = "temp"
	cmd = "ssh "
	cmd += account_name
	cmd += "@cori.nersc.gov"
	os.system(cmd)
	#add some thing to wait for login/flag to let program continue
	os.system("shifter pull elijah4928/temp:latest")
	"""
	


if __name__ == "__main__":
	main(sys.argv)


