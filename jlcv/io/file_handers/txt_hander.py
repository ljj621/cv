class TxtHander:
    @staticmethod
    def load(file_path, mode='r', split=' '):
        with open(file_path, mode) as f:
            contents = f.readlines()
            for i, c in enumerate(contents):
                tokens = c.strip().split(split)
                if len(tokens) > 0:
                    contents[i] = tokens[0] if len(tokens) == 1 else tokens

        return contents
    
    @staticmethod
    def dump(contents, file_path):
        if isinstance(contents, str):
            contents = [contents]
        assert isinstance(contents, list)
        if '\n' not in contents[0]:
            contents = [c+'\n' for c in contents]

        with open(file_path, 'w') as f:
            f.writelines(contents)