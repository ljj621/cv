import os

def prepare_pcn_data_list():
    split = ['train', 'test', 'val']
    root = f'/home/lj/MyProject/DATASET/Completion/PCN'
    for s in split:
        classes = os.listdir(os.path.join(root, s, 'partial'))
        file_list = open(f'/home/lj/MyProject/DATASET/Completion/PCN/{s}.list', 'w')
        for c in classes:
            dirs = os.listdir(os.path.join(root, s, 'partial', c))
            dirs = ''.join([os.path.join(c, d)+'\n' for d in dirs])
            file_list.write(dirs)
        file_list.close()



if __name__ == '__main__':
    split = ['train', 'test', 'val']

    prepare_pcn_data_list()

    
