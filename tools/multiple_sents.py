def get_lens(lens_file='data/mica/lens.txt'):
    with open(lens_file) as fhand:
        for line in fhand:
            lens = map(int, line.split())
    return lens

def copy_sents(input_name='data/sents/dev.txt', output_name='data/mica/sents4micastags'):
    lens = get_lens()
    with open(input_name) as fhand:
        with open('sents4micastagseqs.txt', 'wt') as fwrite:
            for i, line in enumerate(fhand):
                line = line.rstrip()
                for _ in xrange(lens[i]):
                    fwrite.write(line)
                    fwrite.write('\n')
if __name__ == '__main__':
    copy_sents()
