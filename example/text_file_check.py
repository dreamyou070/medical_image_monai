import os, json

text_dir = 'expert.json'
with open(text_dir, 'r') as f:
    json_data = json.load(f)

for data in json_data :
    # -----------------------------------------------
    expert_opinion = data['Description']
    img_info = data['External ID']
    label_info = data['Label']
    #classifications = label_info['classifications']
    objects = label_info['objects'][0]
    title = objects['title']

    if title != 'None' :
        first_selection = objects['title']
        classification = objects['classifications']
        for class_box in classification :
            if class_box['value'].lower() == 'level_one' :
                level_one_selection = class_box['answer']['value']
                if level_one_selection == 'ill_defined' :
                    print(f'img_info : {img_info} : {level_one_selection}')


    # -----------------------------------------------


    # -----------------------------------------------


