import json
import os


def group_data(keypoints: list[int], iteration_name):
    path = 'data.json'
    data_keys = [
        'left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y',
        'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 'left_eye_outer_corner_x',
        'left_eye_outer_corner_y', 'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
        'right_eye_outer_corner_x', 'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x',
        'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
        'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x',
        'right_eyebrow_outer_end_y', 'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y',
        'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
        'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
    ]

    if not os.path.exists(path):
        with open(path, 'w') as data_json:
            json.dump({}, data_json)
    with open(path, 'r') as data_json:
        old_data = json.load(data_json)
    final_data = {iteration_name: {}}
    for i in range(len(data_keys)):
        final_data[iteration_name][data_keys[i]] = keypoints[i]
    final_data.update(old_data)
    with open(path, 'w') as data_json:
        json.dump(final_data, data_json, indent=2)
        data_json.write('\n')


# data = [2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# data_name = ['left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y','left_eye_inner_corner_x','left_eye_inner_corner_y','left_eye_outer_corner_x','left_eye_outer_corner_y','right_eye_inner_corner_x','right_eye_inner_corner_y','right_eye_outer_corner_x','right_eye_outer_corner_y','left_eyebrow_inner_end_x','left_eyebrow_inner_end_y','left_eyebrow_outer_end_x','left_eyebrow_outer_end_y','right_eyebrow_inner_end_x','right_eyebrow_inner_end_y','right_eyebrow_outer_end_x','right_eyebrow_outer_end_y','nose_tip_x','nose_tip_y','mouth_left_corner_x','mouth_left_corner_y','mouth_right_corner_x','mouth_right_corner_y','mouth_center_top_lip_x','mouth_center_top_lip_y','mouth_center_bottom_lip_x','mouth_center_bottom_lip_y']
# group_data(data, data_name, '0')
