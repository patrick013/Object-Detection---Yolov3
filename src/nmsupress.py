import numpy as np

def _decode_box_coor(box):
    return (box.xmin, box.ymin, box.xmax, box.ymax)


def _iou(box1, box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = _decode_box_coor(box1)
    (box2_x1, box2_y1, box2_x2, box2_y2) = _decode_box_coor(box2)

    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = xi2 - xi1
    inter_height = yi2 - yi1
    inter_area = max(inter_height, 0) * max(inter_width, 0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou

def do_nms(data_dic, nms_thresh):
    final_boxes, final_scores, final_labels = list(), list(), list()
    for label in data_dic:
        scores_boxes = sorted(data_dic[label], reverse=True)
        for i in range(len(scores_boxes)):
            if scores_boxes[i][2] == 'removed': continue
            for j in range(i + 1, len(scores_boxes)):
                if _iou(scores_boxes[i][1], scores_boxes[j][1]) >= nms_thresh:
                    scores_boxes[j][2] = "removed"

        for e in scores_boxes:
            print(label + ' ' + str(e[0]) + " status: " + e[2])
            if e[2] == 'kept':
                final_boxes.append(e[1])
                final_labels.append(label)
                final_scores.append(e[0])

    return (final_boxes, final_labels, final_scores)


