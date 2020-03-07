log_file_path = "/s/neptune/a/nobackup/kartikay/extracted_events_data/log.txt"
target = '/s/chopin/l/grad/kartikay/log.txt'
f = open(log_file_path , 'r')
f_2 = open(target, 'w')

for line in f:
    substr = 'Character Issue'
    s1 = 'Ctrl+[,_S'
    if substr in line:
        if s1 in line:
            continue
        word = line.split('*****')[1]
        f_2.write(word)
        f_2.write('\n\n')

f_2.close()
