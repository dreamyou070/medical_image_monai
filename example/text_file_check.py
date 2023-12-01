import os, json
import numpy as np
from PIL import Image
"""
{'title': 'Periapical',
 'value': 'periapical',
 'classifications': 
 [{
 'featureId': 'ck2gj7kl4002e0q4n4jo9lr0n', 
 'schemaId': 'ck1r3eeqs21jm0757v6v8te4t', 
 'title': 'Level one', 'value': 'level_one', 'answer': {'featureId': 'ck2gj7klx002f0q4nxja9kkb6', 'schemaId': 'ck1r3ee9duodt0838gw5opzoz', 'title': 'Ill Defined', 'value': 'ill_defined'}}, 
 {'featureId': 'ck2gj7lhz002h0r6n4jx6acs9', 'schemaId': 'ck1r3eeqs21jn0757gp51g8wb', 
 
 'title': 'Level two', 'value': 'level_two', 'answer': None},
 {'featureId': 'ck2gj7okr002j0r6nbk8wdgqi', 'schemaId': 'ck1s9qd65oun40721eenuvijo', 'title': 'Level three', 'value': 'level_three', 'answers': [{'featureId': 'ck2gj7olm002k0r6n6rl0yhrq', 'schemaId': 'ck1s9qctm9poz0725d6zvcvgs', 'title': 'None', 'value': 'none'}]}, {'featureId': 'ck2gj7p28002h0r55wbh965tc', 'schemaId': 'ck1s6cfai17eq0811jz58gdza', 'title': 'Level four', 'value': 'level_four', 'answers': [{'featureId': 'ck2gj7p3h002i0r55lyiikmxs', 'schemaId': 'ck1s6cexq8duz0725ml8y9utg', 'title': 'Inflammation', 'value': 'inflammation'}]}], 'polygons': [[[921, 599], [921, 602], [920, 603], [920, 606], [919, 607], [919, 609], [918, 610], [917, 610], [917, 613], [916, 614], [916, 615], [915, 616], [915, 621], [914, 622], [914, 627], [913, 628], [913, 631], [912, 632], [911, 632], [911, 635], [910, 636], [910, 638], [911, 639], [912, 639], [914, 641], [915, 641], [917, 643], [918, 643], [923, 648], [931, 648], [932, 649], [955, 649], [956, 648], [957, 648], [958, 647], [959, 647], [959, 646], [968, 637], [969, 637], [970, 636], [971, 636], [971, 635], [975, 631], [975, 628], [976, 627], [976, 626], [977, 625], [977, 616], [976, 615], [976, 614], [975, 615], [975, 617], [974, 618], [974, 619], [970, 623], [970, 624], [967, 627], [967, 628], [963, 632], [963, 633], [961, 635], [960, 635], [960, 636], [959, 637], [956, 637], [956, 638], [954, 640], [952, 640], [951, 641], [949, 641], [948, 642], [933, 642], [932, 641], [932, 640], [931, 639], [931, 637], [930, 636], [930, 611], [929, 610], [929, 607], [927, 605], [927, 604], [926, 603], [925, 603], [924, 602], [923, 602], [922, 601], [922, 600]]]}

"""
base_mask_dir = r'../../medical_data/dental/turf/Expert/mask'
teeth_mask_base_dir = r'../../medical_data/dental/turf/Segmentation/teeth_mask'
text_dir = 'expert_trim.json'
with open(text_dir, 'r') as f:
    json_data = json.load(f)
base_caption_dir = r'../../medical_data/dental/turf/Expert/caption'
os.makedirs(base_caption_dir, exist_ok=True)
# Within normal limits
"""
for data in json_data :
    # -----------------------------------------------
    img_info = data['External ID']
    name = img_info.split('.')[0]
    expert_opinion = data['Description']
    label_info = data['Label']['objects']

    if expert_opinion != 'Within normal limits' :
        info = label_info[0]
        infos = info['classifications']
        level_threes = []
        for info in infos :
            if info['title'] == 'Level three' :
                try :
                    answer = info['answers'][0]['title'].lower()
                except :
                    answer = info['answer']['title'].lower()
                if answer not in level_threes :
                    level_threes.append(answer)

        level_fours = []
        for info in infos:
            if info['title'] == 'Level four':
                try:
                    answer = info['answers'][0]['title'].lower()
                except:
                    answer = info['answer']['title'].lower()
                if answer not in level_fours :
                    level_fours.append(answer)
        disease = level_threes + level_fours
        if 'none' in disease :
            disease.remove('none')
        #print(disease)
    else :
        disease = ['normal']

    teeth_mask_dir = os.path.join(teeth_mask_base_dir, name + '.JPG')
    teeth_mask_pil = Image.open(teeth_mask_dir)
    teeth_mask = np.array(teeth_mask_pil)
    teeth_standard = teeth_mask.sum()
    if teeth_standard > 0 :
        disease.append('teeth')

    caption_dir = os.path.join(base_caption_dir, name + '.txt')
    with open(caption_dir, 'w') as f:
        for i, e in enumerate(disease) :
            if i == len(disease) - 1 :
                f.write(f'{e}')
            else :
                f.write(f'{e}, ')
"""
img_base_dir = r'../../medical_data/dental/turf/Segmentation/teeth_mask'




                    #print(title)
                #answer = info['answers']
                #print(answer)
        #if len(label_info) == 2 :

            #first_selection = label_info[0]['title'].lower()
            #second_selection = label_info[1]['title'].lower()
            #if first_selection != second_selection :
            #    continue
            #else :
            #    #label_info =
            #    print(label_info[0])
            #    print(label_info[1])
            #    print('------------------------------------------------------------')
            #    data['Label']['objects'] = [label_info[0]]

"""
text_dir = 'expert_trim.json'
with open(text_dir, 'w') as f:
    json.dump(new, f)




    if title != 'None' :
        first_selection = objects['title']
        classification = objects['classifications']
        caption_list = []
        for class_box in classification :
            if class_box['value'].lower() == 'level_one' :
                level_one_selection = class_box['answer']['value'].lower().strip()
                if level_one_selection != 'none' :
                    caption_list.append(level_one_selection)
                #if level_one_selection == 'ill_defined' :
                #    print(f'img_info : {img_info} : {level_one_selection}')
            if class_box['value'].lower() == 'level_two':
                try :
                    level_two_selection = class_box['answer']['value'].lower().strip()
                    if level_two_selection != 'none' :
                        caption_list.append(level_two_selection)
                except :
                    continue
            if class_box['value'].lower() == 'level_three':
                if 'answers' in class_box.keys() :
                    level_three_selection = class_box['answers'][0]['value'].lower().strip()
                    if level_three_selection != 'none':
                        caption_list.append(level_three_selection)
                else:
                    level_three_selection = class_box['answer']['value'].lower().strip()
                    if level_three_selection != 'none':
                        caption_list.append(level_three_selection)
            if class_box['value'].lower() == 'level_four' :
                if 'answers' in class_box.keys() :
                    level_four_selection = class_box['answers'][0]['value'].lower().strip()
                    if level_four_selection != 'none':
                        caption_list.append(level_four_selection)
                else:
                    level_four_selection = class_box['answer']['value'].lower()

        caption = ', '.join(caption_list)
        caption = caption.strip()
    else :
        caption = 'normal'
    name, ext = os.path.splitext(img_info)
    os.makedirs('Radiographs_masked_text', exist_ok=True)
    text_dir = os.path.join('Radiographs_masked_text', name + '.txt')
    with open(text_dir, 'w') as f:
        f.write(caption)
"""