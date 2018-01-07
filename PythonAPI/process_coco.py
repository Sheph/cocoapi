#!/usr/bin/python2

import cv2, os
import numpy as np
from pycocotools.coco import COCO

go_auto = True

def box2yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0] + box[2]/2.0
    y = box[1] + box[3]/2.0
    w = box[2]
    h = box[3]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def process_coco(coco, img_path, out_img_path, out_label_path, out_list):
    global go_auto

    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(ids = img_ids)

    idx = 1

    out_img_path = os.path.abspath(out_img_path)

    txt_out_list = open(out_list, "w")

    want_red = False

    num_red = 0
    num_normal = 0

    iters = 0

    for img in imgs:
        iters += 1
        if iters > 1000:
            iters = 0
            print("processed", idx, "out of", len(imgs), "num_red", num_red, "num_normal", num_normal)

        frame = cv2.imread(img_path + "/" + img['file_name'])

        orig_frame = frame.copy()

        out_frame = orig_frame.copy()

        yolo_anns = []

        ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            got_red = False
            if ('bbox' in ann) and ('segmentation' in ann) and ('keypoints' in ann) and want_red:
                sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])-1
                sks = [sks[4], sks[5], sks[6], sks[7], sks[8], sks[9]]
                #sks = [sks[4], sks[5], sks[6], sks[7]]
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]

                found = False
                for sk in sks:
                    if np.all(v[sk] > 0):
                        found = True
                        break

                if found:
                    bbox = [int(xx) for xx in ann['bbox']]
                    msk = coco.annToMask(ann) * 255

                    part = orig_frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
                    grab_msk = np.zeros((bbox[3], bbox[2]), np.uint8)
                    grab_msk[msk[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] == 255] = 2

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)

                    #msk3 = cv2.merge((msk, msk, msk))
                    #frame = cv2.addWeighted(frame, 1.0, msk3, 0.5, 0)

                    for sk in sks:
                        if np.all(v[sk]>0):
                            cv2.polylines(frame, np.int32([zip(x[sk], y[sk])]), 0, (255,0,0), 2)
                            cv2.polylines(grab_msk, np.int32([zip(x[sk] - bbox[0], y[sk] - bbox[1])]), 0, 1, 2)
                    pts = zip(x[v>0], y[v>0])
                    for pt in pts:
                        cv2.circle(frame, pt, 2, (0,255,0), 2)
                    pts = zip(x[v>1], y[v>1])
                    for pt in pts:
                        cv2.circle(frame, pt, 2, (255,255,0), 2)

                    grab_msk = cv2.bitwise_and(grab_msk, msk[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]])

                    #cv2.imshow('grab_msk', grab_msk * 127)
                    #cv2.imshow('part', part)

                    if np.any(grab_msk == 1):
                        bgdModel = np.zeros((1,65),np.float64)
                        fgdModel = np.zeros((1,65),np.float64)
                        cv2.grabCut(part, grab_msk, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                        mask2 = np.where((grab_msk==2)|(grab_msk==0),0,1).astype('uint8')
                        img = part*mask2[:,:,np.newaxis]
                        #cv2.imshow('frame2', img)

                        msk3 = np.zeros(msk.shape, np.uint8)
                        msk3[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = mask2
                        #msk3 = cv2.merge((msk3 * 0, msk3 * 0, msk3 * 255))
                        out_frame[msk3 == 1] = (0,0, 255)
                        num_red += 1
                        yolo_anns.append((bbox, 1))
                        got_red = True
                        want_red = False
                        #out_frame = cv2.addWeighted(out_frame, 1.0, msk3, 1.0, 0)
            if not got_red and ('bbox' in ann):
                bbox = [int(xx) for xx in ann['bbox']]
                num_normal += 1
                yolo_anns.append((bbox, 0))
                want_red = True

        assert(len(yolo_anns) > 0)

        if not go_auto:
            cv2.imshow('frame', frame)
            cv2.imshow('out_frame', out_frame)

        cv2.imwrite(out_img_path + "/" + str(idx) + ".jpg", out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        txt_outfile = open(out_label_path + "/" + str(idx) + ".txt", "w")

        for ann in yolo_anns:
            bb = box2yolo((out_frame.shape[1], out_frame.shape[0]), ann[0])
            txt_outfile.write(str(ann[1]) + " " + " ".join([str(a) for a in bb]) + '\n')

        txt_outfile.close()

        txt_out_list.write(out_img_path + "/" + str(idx) + ".jpg" + '\n')

        idx += 1

        if not go_auto:
            k = cv2.waitKey(0) & 0xff
            if k == 27:
                break

    txt_out_list.close()

    print("Done. num_red", num_red, "num_normal", num_normal)

if __name__ == "__main__":
    process_coco(COCO("../annotations/person_keypoints_train2014.json"), "../images/train2014", "../my/images/train", "../my/labels/train", "../my_train.txt")
    process_coco(COCO("../annotations/person_keypoints_val2014.json"), "../images/val2014", "../my/images/val", "../my/labels/val", "../my_val.txt")
