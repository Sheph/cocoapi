#!/usr/bin/python2

import cv2
import numpy as np
from pycocotools.coco import COCO

def process_coco(coco):
    cat_ids = coco.getCatIds(catNms=['person'])
    print(cat_ids)
    img_ids = coco.getImgIds(catIds=cat_ids)
    imgs = coco.loadImgs(ids = img_ids)
    for img in imgs:
        frame = cv2.imread("../images/train2014/" + img['file_name'])

        orig_frame = frame.copy()

        ann_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ('bbox' in ann) and ('segmentation' in ann) and ('keypoints' in ann):
                sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])-1
                sks = [sks[4], sks[5], sks[6], sks[7], sks[8], sks[9]]
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]

                found = False
                for sk in sks:
                    if np.all(v[sk] > 0):
                        found = True
                        break

                if not found:
                    continue

                bbox = [int(xx) for xx in ann['bbox']]
                msk = coco.annToMask(ann) * 255

                grab_msk = np.zeros(frame.shape[:2], np.uint8)
                #grab_msk[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1
                grab_msk[msk == 255] = 2

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)

                msk = cv2.merge((msk, msk, msk))
                frame = cv2.addWeighted(frame, 1.0, msk, 0.5, 0)

                for sk in sks:
                    if np.all(v[sk]>0):
                        cv2.polylines(frame,  np.int32([zip(x[sk], y[sk])]), 0, (255,0,0), 2)
                        cv2.polylines(grab_msk,  np.int32([zip(x[sk], y[sk])]), 0, 1, 2)
                pts = zip(x[v>0], y[v>0])
                for pt in pts:
                    cv2.circle(frame, pt, 2, (0,255,0), 2)
                pts = zip(x[v>1], y[v>1])
                for pt in pts:
                    cv2.circle(frame, pt, 2, (255,255,0), 2)

                bgdModel = np.zeros((1,65),np.float64)
                fgdModel = np.zeros((1,65),np.float64)
                cv2.grabCut(orig_frame, grab_msk, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                mask2 = np.where((grab_msk==2)|(grab_msk==0),0,1).astype('uint8')
                img = orig_frame*mask2[:,:,np.newaxis]
                cv2.imshow('frame2', img)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break

if __name__ == "__main__":
    #process_coco(COCO("../annotations/instances_train2014.json"))
    #process_coco(COCO("../annotations/instances_val2014.json"))
    process_coco(COCO("../annotations/person_keypoints_train2014.json"))
