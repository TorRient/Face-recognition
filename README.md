# Face recognition with insightface
# Use
git clone https://github.com/TorRient/Face-recognition.git

cd Face-recognition

git clone https://github.com/deepinsight/insightface.git

download Pretrain model https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0 put in /insightface/models

download Pretrain Retina https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip?dl=0 put in insightface/RetinaFace/model/

# Test
python recognizer_image.py --image_in 'path to input' --image_out 'path to out'

![Example](https://github.com/TorRient/Face-recognition/blob/master/data/test/cap_recog.jpg)
