import cv2
import os

img_dir_wh_l = './rir_evolutions/wh_noise/L_channel'

wh_l_files = sorted(os.listdir(img_dir_wh_l))

op_vid_wh_l = './rir_evolutions/wh_noise/wh_l.mp4'

frame_rate = 30

img_1 = cv2.imread(os.path.join(img_dir_wh_l, wh_l_files[0]))
# img_1 = './rir_evolutions/wh_noise/L'
h, w, chs = img_1.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_writer = cv2.VideoWriter(op_vid_wh_l, fourcc, frame_rate, (w, h), True)

for img in wh_l_files:
    img_path = os.path.join(img_dir_wh_l, img)
    frame = cv2.imread(img_path)
    vid_writer.write(frame)

vid_writer.release()

def make_vid(path, fr_rate, op_file):
    files = sorted(os.listdir(path))
    im1 = cv2.imread(os.path.join(path, files[0]))

    h, w, chs = im1.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    vid_writer = cv2.VideoWriter(op_file, fourcc, fr_rate, (w, h), True)

    for img in files:
        img_path = os.path.join(path, img)
        frame = cv2.imread(img_path)
        vid_writer.write(frame)

    vid_writer.release()

# wh_r_path = './rir_evolutions/wh_noise/R_channel/'
# op_wh_r_vid = './rir_evolutions/wh_noise/wh_R_vid.mp4'
# make_vid(wh_r_path, 5, op_wh_r_vid)
