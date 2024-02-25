import os
import xml.etree.ElementTree as ET

path1 = 'C:\\Stuffing\\ImPing\\ObjectDetection\\annotations'
path2 = 'C:\\Stuffing\\ImPing\\ObjectDetection\\images'
annots = os.listdir(path1)
imgs = os.listdir(path2)
filename = 'testname'

for i in range(len(annots)):
    filename = 'image_' + str(i+1)
    filejpg = filename + '.jpg'
    os.rename(f'images\\{imgs[i]}', filejpg)

    mytree = ET.parse(f'annotations\\{annots[i]}')
    root = mytree.getroot()

    for child in root:
        if child.tag == 'filename':
            child.text = filejpg
            child.set('updated', 'yes')
        if child.tag == 'path':
            child.text = filejpg
            child.set('updated', 'yes')

    mytree.write(f'{filename}.xml')


    