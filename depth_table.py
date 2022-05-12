from matplotlib import pyplot as plt
import cv2
import pandas as pd
from skimage import draw
import numpy as np

COLS = ["Needle Number", "Hole Location", "Number of Seeds", "Distance last Seed to Grid (mm)", "Insertion Depth (mm)",
                 "Left over (mm)", "comments"]
IM_FOLDER_PATH = r"C:\Users\ilaym\Desktop\Dart\GridTrack\static\img"
font = cv2.FONT_HERSHEY_SIMPLEX

START_HOLE = 'A1'
START_COORDS = (379,66)
COL_STEP = 21.3
ROW_STEP = 18.5


def show_image(im_path):
    im = plt.imread(im_path)
    plt.imshow(im)
    plt.show()


def create_table(out_path):
    col_names = COLS
    holes = []
    for idx in range(ord('a'), ord('p') + 1):
        holes.extend([chr(idx).upper() + str(i) for i in range(1,19,2)])
        holes.extend([chr(idx) + str(i) for i in range(2, 19, 2)])
    df = pd.DataFrame(columns=col_names)
    df["Hole Location"] = holes

    writer = pd.ExcelWriter('%s/depth_table.xlsx' % out_path, engine='xlsxwriter')

    df.to_excel(writer, sheet_name='depth table', index=False)
    writer.save()
    return df


def mark_grid():
    im = (plt.imread("{}/grid.png".format(IM_FOLDER_PATH))*256).astype(np.uint8)
    df = pd.read_csv("static/items_example.csv")

    needles = df[df['Number of Seeds'].notnull()]
    to_insert = needles[needles["Needle Number"].isnull()]
    to_insert = to_insert[to_insert["Needle Number"].isnull()]
    holes_to_red = to_insert["Hole Location"]
    cells_to_red = holes_to_red.apply(lambda x: hole_to_cell(x,
                                            START_HOLE, START_COORDS, ROW_STEP, COL_STEP))
    depth = to_insert["Insertion Depth (mm)"]
    seed_num = to_insert["Number of Seeds"]
    inserted = needles[needles["Needle Number"].notnull()]
    leftover = inserted["Left over (mm)"]
    holes_to_green = inserted["Hole Location"]
    cells_to_green = holes_to_green.apply(lambda x: hole_to_cell(x,
                                            START_HOLE, START_COORDS, ROW_STEP, COL_STEP))
    im = highlight(cells_to_red, [255,0,0], im, depth, seed_num)
    im = highlight(cells_to_green, [0,125,0], im, leftover)
    plt.imsave("{}/map.png".format(IM_FOLDER_PATH), im)
    # fig = plt.figure
    # plt.imshow(im)
    # plt.show()

    # wb.sheets[1].pictures[0].delete()

    # print(ret)
    # if ret is not None:
    #     sht1.range('A1').value = 4


def fill_grid_locations(start, end, sheet):
    cell_num = 3
    for i in range(ord(start), ord(end)):
        letter = chr(i)
        for j in range(1,19):
            grid_location = letter + str(j)
            print(grid_location)
            cell = 'D' + str(cell_num)
            sheet.range(cell).value = grid_location
            cell_num += 1
        letter = chr(i).lower()
        for j in range(1, 19):
            grid_location = letter + str(j)
            print(grid_location)
            cell = 'D' + str(cell_num)
            sheet.range(cell).value = grid_location
            cell_num += 1


def hole_to_cell(hole, start_hole, start_coords, row_step, col_step):
    lower = hole[0].islower()
    letter_diff = ord(str(hole)[0].upper()) - ord(start_hole[0].upper())
    num_diff = int(hole[1:]) - int(start_hole[1])
    # print(hole , num_diff, letter_diff)
    cell = [start_coords[0] - row_step*num_diff, start_coords[1] + letter_diff * col_step]
    cell[1] = cell[1] + (col_step / 2) if lower else cell[1]
    return cell
   

# def insert_image(wb, fig):
#     sht1 = wb.sheets['Sheet1']
#     sht1.range('A1').value = 4
#     wb.sheets[1].pictures[0].delete()
#     wb.sheets[1].pictures.add(r'C:\Users\ilaym\Desktop\sites\Hadassa\HA-COMP-01\depth_table\grid_marked.jpeg', height=700, width=800)
#     pic = wb.sheets[1].pictures
#     return pic


def highlight(cells, color, im, depth=None, seed_num=None):
    rows = cells.index.array
    i = 0
    for cell in rows:
        center = (int(cells[cell][0]), int(cells[cell][1]))
        rr, cc = draw.disk((center[0], center[1]), 5)
        for ch in range(3):
            im[rr,cc,ch] = color[ch]
        # im = cv2.circle(im, center, 3, color, thickness=cv2.FILLED, lineType=cv2.FILLED)
        if depth is not None:
            depth_arr = np.array(depth)
            if seed_num is not None:
                seed_num = np.array(seed_num)
                # text = str(depth_arr[i] + "," + str(seed_num[i])
            # cv2.putText(img, 'Text on image goes here', (10, 500), font, 1, (255, 255, 255), 2)
            #     im = cv2.putText(im, str(depth_arr[i]) + ", " + str(int(seed_num[i])), (center[1] - 10, center[0] - 5), font, 0.3, (0,0,0), 1)
            # else:
            #     im = cv2.putText(im, str(depth_arr[i]), (center[1] - 10, center[0] - 10), font,
            #                      0.4, (0, 0, 0), 1)
            i += 1
    # plt.imshow(im)
    # plt.show()
    return im
