from easydict import EasyDict as edict

configs = edict()
configs["cuda"] = True
configs["image_dir"] = "/home/yrj/Dataset/SceneText/English/Test_sets/CUTE80/crop/"
configs["val_list"] = "/home/yrj/Dataset/SceneText/English/Test_sets/CUTE80/crop/gt.txt"
configs["model_path"] = "./models/pren.pth"
configs["imgH"] = 64
configs["imgW"] = 256
configs["alphabet"] = "data/alphabet_en.txt"
configs["vert_test"] = True
configs["batchsize"] = 1
configs["display"] = True
configs["workers"] = 0
