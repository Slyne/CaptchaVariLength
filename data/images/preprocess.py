#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import json
input_directories = ["five_digit/", "four_digit/", "six_digit/","seven_digit/"]

pictures = []
ch_index = {}
labels = []

index = 0
image_no = 0
for directory in input_directories:
    for pic in os.listdir(directory):
        suffix_index = pic.find(".")
        cur_pic_label = pic[0:suffix_index]
        labels.append(cur_pic_label)
        for l in cur_pic_label:
            if l not in ch_index:
                ch_index[l] = index
                index += 1
        img = Image.open(directory + pic)
        img = img.resize((250,60),Image.BILINEAR) # all convert to Height = 60, width=250, channel = 3
        img = np.array(img)
        pictures.append(np.rollaxis(img, 2, 0))
        image_no += 1
        if image_no%1000 == 0:
            print image_no

print "save pictures"
with open("pic", "wb") as inp:
    np.save(inp, pictures)

ch_index['<EOF>'] = index

with open("ch_index", "w") as inp:
    json.dump(ch_index, inp)

max_caption_length = 7 + 1
id_labels = []
for label in labels:
    template = [ch_index['<EOF>']] * max_caption_length
    for (i,c) in enumerate(label):
        template[i] = ch_index[c]
    id_labels.append(template)
print "save labels"
with open("labels", "wb") as inp:
    np.save(inp, id_labels)