#!/usr/bin/env python3
import numpy as np
import cv2, os, sys
import argparse
import imutils


def YUVtoRGB(args, yuv_file):
    stream = open(yuv_file, 'rb')

    e = 1280 * 720
    Y = stream[0:e]
    Y = np.reshape(Y, (args.height, args.width))

    s = e
    V = stream[s::2]
    V = np.repeat(V, 2, 0)
    V = np.reshape(V, (args.height/2, args.width))
    V = np.repeat(V, 2, 0)

    U = stream[s + 1::2]
    U = np.repeat(U, 2, 0)
    U = np.reshape(U, (args.height/2, args.width))
    U = np.repeat(U, 2, 0)

    RGBMatrix = (np.dstack([Y, U, V])).astype(np.uint8)
    RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB_NV12, 3)

    return RGBMatrix


def cvt_yuv_nv12_to_rgb(width, height, rotate, yuv_file):
    # yuv_file = 't.yuv'
    # print 'yuv_file = ',yuv_file
    stream = open(yuv_file, 'rb')
    # Seek to the fourth frame in the file
    # stream.seek(4 * width * height * 1.5)
    # Calculate the actual image size in the stream (accounting for rounding
    # of the resolution)
    fwidth = (width + 31) // 32 * 32
    fheight = (height + 15) // 16 * 16
    # Load the Y (luminance) data from the stream

    Y = np.fromfile(stream, dtype=np.uint8, count=fwidth * fheight). \
        reshape((fheight, fwidth))
    # Load the UV (chrominance) data from the stream, and double its size
    UV = np.fromfile(stream, dtype=np.uint8, count=2 * (fwidth // 2) * (fheight // 2)). \
        reshape((fheight // 2, fwidth // 2, 2)). \
        repeat(2, axis=0).repeat(2, axis=1)

    V = UV[:, :, 0]
    U = UV[:, :, 1]

    # Stack the YUV channels together, crop the actual resolution, convert to
    # floating point for later calculations, and apply the standard biases
    YUV = np.dstack((Y, U, V))[:height, :width, :].astype(np.float)
    YUV[:, :, 0] = YUV[:, :, 0] - 16  # Offset Y by 16
    YUV[:, :, 1:] = YUV[:, :, 1:] - 128  # Offset UV by 128
    # YUV conversion matrix from ITU-R BT.601 version (SDTV)
    # Note the swapped R and B planes!
    #              Y       U       V
    # M = np.array([
    #     [1.164, 0.000, 1.596],
    #               [1.164, 2.017, 0.000],  # B
    #
    #               [1.164, -0.392, -0.813]  # G
    #
    #
    # ])  # R

    M = np.array([[1.164, 2.017, 0.000],  # B
                  [1.164, -0.392, -0.813],  # G
                  [1.164, 0.000, 1.596]])  # R
    # Take the dot product with the matrix to produce BGR output, clamp the
    # results to byte range and convert to bytes
    BGR = YUV.dot(M.T).clip(0, 255).astype(np.uint8)

    RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)

    IMG = imutils.rotate_bound(RGB, rotate)
    return IMG


def process(args, yuv_list):
    succ = 0
    fail = 0

    for count, file in enumerate(yuv_list):

        image_path = os.path.realpath(os.path.dirname(file) + os.sep + os.path.basename(file).strip())
        # image_path = args.files_path + file.strip()
        if args.height and args.width:
            width = args.width
            height = args.height
        else:
            file_wh = file.split("_")[-1].split(".")[0].split()[0].split("x")
            if len(file_wh) != 2:
                print("[%d/%d] Fail: %s" % (count, len(yuv_list), image_path))
                print(f"Error: Could not get image width and height from filename!")
                fail += 1
                continue
            width = int(file_wh[0])
            height = int(file_wh[1])

        try:
            img = cvt_yuv_nv12_to_rgb(width, height, args.rotate, image_path)
            # img = YUVtoRGB(args, image_path)

            saved_image_path = image_path.rstrip('.yuv').rstrip('.YUV') + '.jpg'
            cv2.imwrite(saved_image_path, img)
            print("[%d/%d] Success: %s" % (count+1, len(yuv_list), image_path))
            succ += 1
        except Exception as err:
            print("[%d/%d] Fail: %s" % (count+1, len(yuv_list), image_path))
            print(f"Error: {err}")
            fail += 1

    print()
    print("Success %d times" % succ)
    print("Fail %d times" % fail)


def getYuvFileList(filesPath, file_type='yuv'):
    all_filesList = []
    yuv_filesList = []
    for (dirpath, dirnames, filenames) in os.walk(filesPath):
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            all_filesList.append(file_path)
        # all_filesList.extend(filenames)
    for file in all_filesList:
        if file.lower().endswith(file_type):
            yuv_filesList.append(file)

    return yuv_filesList

def parse_args():
    temp = r'C:\Users\mms00519\Downloads\case9'

    default_description = 'transfrom yuv images to jpg'
    parser = argparse.ArgumentParser(prog="nv12_to_jpg.py", description=default_description)
    parser.add_argument("-p", "--files_path", type=str, required=False, default='./', help='default ./')
    parser.add_argument("-hg", "--height", type=int, required=False, help='yuv height')  # 1792
    parser.add_argument("-w", "--width", type=int, required=False, help='yuv width')  # 2368
    # parser.add_argument("-a", "--auto", type=bool, required=False, default=True, help='default True')  # auto get height and width
    parser.add_argument("-r", "--rotate", type=int, required=False, default=0, help='default 0')
    return parser.parse_args()


def main():
    # Works for 8 BPP, semi planar NV12 YUVs
    args = parse_args()

    print()
    print('----------START----------')
    print()
    yuv_filesList = getYuvFileList(args.files_path, 'yuv')

    if len(yuv_filesList) is not 0:
        process(args, yuv_filesList)
    else:
        print('no yuv files found')

    print()
    print('----------END----------')
    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
