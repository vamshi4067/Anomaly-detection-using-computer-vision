import os
import subprocess
import pandas as pd
test_path = "/content/data/test_images"
test_samples = os.listdir(test_path)
for idx,exmpl in enumerate(test_samples):
	print("Testing %d image %s,"%(idx,exmpl))
	image_path="/content/data/test_images/"+exmpl
	yolov3_weights_path="/content/yolov3_9000.weights"
	cfg_path="/content/yolov3.cfg"
	data_path="/content/input.cfg"
	output_path="/content/output"
	image_name = os.path.basename(image_path)
	process = subprocess.Popen(['./darknet', 'detector','test',data_path, cfg_path, yolov3_weights_path, image_path],
						stdout=subprocess.PIPE,
						stderr=subprocess.PIPE)
	stdout, stderr = process.communicate()

	std_string = stdout.decode("utf-8")
	std_string = std_string.split(image_path)[1]
	count = 0
	outputList = []
	rowDict = {}
	for line in std_string.splitlines():

	   if count > 0:
		   if count%2 > 0:
			   obj_score = line.split(":")
			   obj = obj_score[0]
			   score = obj_score[1]
			   rowDict["object"] = obj
			   rowDict["score"] = score
		   else:
			   bbox = line.split(",")
			   rowDict["bbox"] = bbox
			   outputList.append(rowDict)
			   rowDict = {}
	   count = count +1
	rowDict["image"] = image_path
	rowDict["predictions"] = outputList

	df = pd.DataFrame(rowDict)
	df.to_json(output_path+"/"+image_name.replace(".jpg", ".json").replace(".png", ".json"),orient='records')
	print("Image %d written to output folder"%(idx))
