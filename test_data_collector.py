import json

log_file_path = "/s/neptune/a/nobackup/kartikay/extracted_events_data/log.txt"

f = open(log_file_path , 'r')
condition = 2
for line in f:
    if condition == 1:
        substr = 'Character Issue'
    else:
        substr = 'FIND_EVENT'

    if substr in line:
        file_name = line.split('*****')[-1]
        #eliminate trailing new line character
        file_name = file_name[:-1]
        data = json.load(open(file_name))
        import pdb; pdb.set_trace()
