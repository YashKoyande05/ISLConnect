from roboflow import Roboflow

rf = Roboflow(api_key="fmhnBdjcmvVJFk1GseZF")
project = rf.workspace("indian-sign-language-detection").project("isl-words")
version = project.version(2)
dataset = version.download("multiclass")
print("Download successfully completed!")