import json
import os

'''
<Choice value="EN"/>
<Choice value="ES"/>
<Choice value="JA"/>
<Choice value="FR"/>
<Choice value="PTBR"/>
<Choice value="DE"/>
<Choice value="IT"/>
<Choice value="PL"/>
<Choice value="Mangled VAD utterance"/>
<Choice value="Speech - Unintelligible"/>
<Choice value="Speech - Filler - uh/umm/mhm"/>
<Choice value="Speech from electronic devices"/>
<Choice value="Speech + Music"/>
<Choice value="Speech + Noise"/>
<Choice value="NoSpeech - Sighs/coughs/grunts"/>
<Choice value="NoSpeech - Laughter"/>
<Choice value="Music - Singing"/>
<Choice value="Music - Other"/>
<Choice value="Noise"/>
<Choice value="Silence"/>
'''

# process iroro's labels into our csv format
# narcos / pinegap / rodneycarr / weeds
def generate_label(dir_path, des_path):
    start_list, end_list, label_list = [], [], []
    noise = ['Silence', 'Noise', 'NoSpeech - Laughter', 'NoSpeech - Sighs/coughs/grunts', 'Speech + Noise']
    music = ['Music - Other', 'Music - Singing', 'Speech + Music']
    speech = ['Speech + Music', 'Speech + Noise', 'Speech from electronic devices', 'Speech - Unintelligible', 'Mangled VAD utterance', 'EN', 'ES', 'JA', 'FR', 'PTBR', 'DE', 'IT', 'PL']
    for file in sorted(os.listdir(dir_path))[:]:
        try:
            with open(os.path.join(dir_path, file), "r") as read_file:
                json_data = json.load(read_file)
                labels = json_data['completions'][0]['result'][0]['value']['choices']
                url = os.path.basename(json_data['task_path'])[:-4].split('_')
                start = url[-2]
                end = url[-1]
                for label in labels:
                    if label in music:
                        start_list.append(start)
                        end_list.append(end)
                        label_list.append('m')
                    if label in speech:
                        start_list.append(start)
                        end_list.append(end)
                        label_list.append('s')
        except: pass

    start_list = list(map(float, start_list))
    end_list = list(map(float, end_list))
    start_list, end_list, label_list = zip(*sorted(zip(start_list, end_list, label_list)))

    if not os.path.exists(des_path):
        os.makedirs(des_path)
    oup_file = open(os.path.join(des_path, dir_path.split('/')[-2] + '.csv'), "w")
    for start, end, label in zip(start_list, end_list, label_list):
        oup_file.write(str(start) + '\t' + str(end) + '\t' + label + '\n')
        print(start, end, label)
    oup_file.close()

generate_label('/root/iroro_sample/raw_labels/weeds/completions', '/root/iroro_sample/labels_s/')