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
		if file.endswith(".csv") or file.endswith(".pickle") or file.endswith(".pkl") or file.endswith(".p"):
			updating = True


	directory = os.getcwd()
	for root, subdirectories, files in os.walk(directory):
		for subdirectory in subdirectories:
			print(os.path.join(root, subdirectory))
		for file in files:
			print(os.path.join(root, file))


	send_data(new_file, updating)






def createdocker(name, tag, volumename, imgpath, push=True, build=False):

	if build:
		#creates docker from imgpath with tagname(defualted to docker hub account) and pushed to hub
		#TODO change/finalize dockerhub account or other source of approved/secure webhosting if needed
		#TODO fix easier naming convention instead of id
		name2 = None
		if name =="docker":
			name2 = 'docker'
		else:
			name2 = 'shifter'


		try:
			cmd = "docker build -t " + name2 + ":" + tag + " " + imgpath
		finally:
			print("docker building name isn't working still")
			cmd = "docker build -t " + tag + " " + imgpath
		print(cmd)
		#os.system("docker rename d9c79b07ebe6 " + name)
		os.system(cmd)

		#pushes container to docker hub
		if False:
			cmd = "docker push " + tag
			os.system(cmd)

		os.system("echo \nFinsihed Building docker image\n")




	workingdirectory = "/src"
	command = "bash"


	imgid = None
	containerID = None
	if name=="shifter":
		imgid = "d9c79b07ebe6"
		containerID = "3e4d2338bce9e72488170402384716c3c41b94645854fc68ff7589860bc77267"
	else:
		imgid = "c3d0e325618f"
		containerID = "3d4875e5cafd80cc0a889f821eefbe2a84aa3fe4de8e48a089f06a7ee5e88a29"
	
		

	try:
		cmd2 = "docker run -it --name " + name + " -v " + imgpath + ":" + workingdirectory + " -w " + workingdirectory + " " + imgid + " " + command
		print(cmd2)
		os.system(cmd2)
	finally:
		cmd2 = "docker start -i " + containerID
		print(cmd2)
		os.system(cmd2)



	print("\n\n" + "docker name that exited is:" + name + "\n\n")
	os.system("docker stop " + name)









def main(argv):
	#python main.py filename(data) true
	print(argv)
	if len(argv) != 3 and len(argv) != 2:
		print("python main <new data filename> <set build to true>")
	try:
		check_data(argv[1])
	except:
		print("")
	if len(argv) >= 2:
		folder = os.getcwd()
		createdocker("shifter", "elijah4928temp", "moleculedata", folder, True, bool(argv[-1]))
		createdocker("docker", "elijah4928temp", "moleculedata", folder, True, bool(argv[-1]))
	else:
		folder = os.getcwd()
		createdocker("shifter", "elijah4928/temp", "moleculedata", folder, True)
		createdocker("docker", "elijah4928/temp", "moleculedata", folder, True)


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











