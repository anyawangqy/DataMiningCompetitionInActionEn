import os
import random
def generate_name_title_list(ann_file, namelist_file, titlelist_file):
    with open(ann_file, 'r') as fin:
        with open(namelist_file, 'w') as fout_name:
            with open(titlelist_file, 'w') as fout_title:
                for line in fin:
                    filename, label = line.strip().split()
                    name = os.path.basename(filename)[:os.path.basename(filename).rfind('.avi')]
                    title = ''.join([chr(random.randint(97, 122)) for _ in range(random.randint(3, 32))])
                    fout_name.write(name + '\n')
                    fout_title.write(title + '\n')
