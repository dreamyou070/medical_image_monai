import os, json

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
def file_check(data_dir) :
    with open(data_dir, 'r') as f:
        file_data = f.readlines()
    new_file_data = []
    for elem in file_data :
        new_file_data.append(elem.strip())
    return new_file_data

normal_data_dir = 'normal_check.txt'
normal_data = file_check(normal_data_dir)
ood_data_dir = 'oob_check.txt'
ood_data = file_check(ood_data_dir)

text_dir = 'expert.json'
with open(text_dir, 'r') as f:
    json_data = json.load(f)

for data in json_data:
    # -----------------------------------------------
    expert_opinion = data['Description']
    img_info = data['External ID']
    label_info = data['Label']
    # classifications = label_info['classifications']
    objects = label_info['objects'][0]
    title = objects['title']
    if expert_opinion.lower() == 'Within normal limits'.lower() :
        if img_info not in normal_data :
            print(img_info)
    else :
        if img_info not in ood_data :
            print(f'[{img_info}] except opinion : {expert_opinion.lower()}')
            if img_info in normal_data :
                print(f'[{img_info}] normal data')

