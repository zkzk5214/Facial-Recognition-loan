import numpy as np


def filter_bbx(bbx_list):
    """from stupid wxp"""
    filter_bbx = []
    points = [[int(j) for j in i[:-1]] for i in bbx_list]

    for i in range(len(points)):

        bbx1_x0, bbx1_y0, bbx1_x1, bbx1_y1 = points[i]
        bbx1_w, bbx1_h = bbx1_x1 - bbx1_x0, bbx1_y1 - bbx1_y0

        # Restrict w/h of bbx
        if (40 < max([bbx1_w, bbx1_h]) < 300) and (30 < min([bbx1_w, bbx1_h]) < 220):
            flag = False
            skip_list = []

            for j in range(len(points[i:])):
                if j in skip_list:
                    continue

                bbx2_x0, bbx2_y0, bbx2_x1, bbx2_y1 = points[j]

                # if bbx2 inside the bbx1
                if bbx1_x0 < bbx2_x0 < bbx2_x1 < bbx1_x1 and bbx1_y0 < bbx2_y0 < bbx2_y1 < bbx1_y1:
                    skip_list.append(j)
                    flag = True

                if bbx1_x0 <= bbx2_x0 < bbx1_x1:
                        # bbx1 right-down intersect bbx2
                        if bbx1_y0 <= bbx2_y0 <= bbx1_y1:
                            # interesect area/ bbx1 > 0.3
                            if (bbx1_x1 - bbx2_x0) * (bbx1_y1 - bbx2_y0) / ((bbx1_x1 - bbx1_x0) * (bbx1_y1 - bbx1_y0)) > 0.3:
                                # if conf bbx1 > 0.6, flag=true
                                flag = True if bbx_list[i][-1] > 0.6 else flag

                                # if conf bbx1 > bbx2, flag=True append bbx2
                                flag = True if bbx_list[i][-1] > bbx_list[j][-1] else flag
                                skip_list = skip_list + [j] if bbx_list[i][-1] > bbx_list[j][-1] else skip_list

                        # bbx1 right-up intersect bbx2
                        elif bbx1_y0 < bbx2_y1 < bbx1_y1:
                            # interesect area/ bbx1 > 0.3
                            if (bbx1_x1 - bbx2_x0) * (bbx1_y1 - bbx2_y0) / ((bbx1_x1 - bbx1_x0) * (bbx1_y1 - bbx1_y0)) > 0.3:
                                # if conf bbx1 > 0.6, flag=true
                                flag = True if bbx_list[i][-1] > 0.6 else flag

                                # if conf bbx1 > bbx2, flag=True append bbx2
                                flag = True if bbx_list[i][-1] > bbx_list[j][-1] else flag
                                skip_list = skip_list + [j] if bbx_list[i][-1] > bbx_list[j][-1] else skip_list
                        # bbx1 right-upper intersect bbx2 or right-upper/down bbx2
                        else:
                            flag = True

                # Left-side condition
                elif bbx2_x0 <= bbx1_x0 < bbx2_x1:
                    if (bbx2_x1 - bbx1_x0) * (bbx1_y1 - bbx2_y0) / ((bbx1_x1 - bbx1_x0) * (bbx1_y1 - bbx1_y0)) > 0.3:
                        # if conf bbx1 > 0.6, flag=true
                        flag = True if bbx_list[i][-1] > 0.6 else flag

                        # if conf bbx1 > bbx2, flag=True append bbx2
                        flag = True if bbx_list[i][-1] > bbx_list[j][-1] else flag
                        skip_list = skip_list + [j] if bbx_list[i][-1] > bbx_list[j][-1] else skip_list

                    elif (bbx2_x1 - bbx1_x0) * (bbx2_y1 - bbx1_y0) / ((bbx1_x1 - bbx1_x0) * (bbx1_y1 - bbx1_y0)) > 0.3:
                        # if conf bbx1 > 0.6, flag=true
                        flag = True if bbx_list[i][-1] > 0.6 else flag

                        # if conf bbx1 > bbx2, flag=True append bbx2
                        flag = True if bbx_list[i][-1] > bbx_list[j][-1] else flag
                        skip_list = skip_list + [j] if bbx_list[i][-1] > bbx_list[j][-1] else skip_list
                    else:
                        flag = True
                else:
                    flag = True
            if flag:
                filter_bbx.append(i)
            if i == len(points) - 1:
                filter_bbx.append(i)

    return [bbx_list[i] for i in filter_bbx]


def filter_black(bbx_list, img):
    """ from stupid wxp"""
    filter_bbx = []
    points = [[int(j) for j in i[:-1]] + [i[-1]] for i in bbx_list if len(i) == 5]

    # Filter too light/dark in POI
    for p in range(len(points)):
        if 50 <= np.mean(img[points[p][1]:points[p][3], points[p][0]:points[p][2], :]) <= 230:
            filter_bbx.append(points[p])

    return filter_bbx
