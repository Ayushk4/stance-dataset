import json, random
from collections import Counter

j = json.load(open('wtwt_ids.json'))

print('Stance:', Counter(x['stance'] for x in j))
print('Merger:', Counter(x['merger'] for x in j))
id_2_merger_stances = {xx: [] for xx in set([x['tweet_id'] for x in j])}

for data in j:
    cnvrt = lambda x: 'DIS_FOX' if x == 'FOXA_DIS' else x
    id_2_merger_stances[data['tweet_id']].append({'tweet_id': data['tweet_id'],
                                                    'merger': cnvrt(data['merger']), 
                                                    'stance': data['stance']
                                                    })


inverse_stance = {'support': 'refute', 'refute': 'support'}

# 1. Sentiment invert
print("============= Step 1 =============")

for tid, this_tid_data in id_2_merger_stances.items():
    to_append = []
    for data in this_tid_data:
        assert tid == data['tweet_id']
        if data['stance'] in ['support', 'refute']:
            to_append.append({'tweet_id': tid,
                            'merger': "NEG_" + data['merger'],
                            'stance': inverse_stance[data['stance']],
                            }
                        )
    for more_data in to_append:
        id_2_merger_stances[tid].append(more_data)


print(len([y for x in id_2_merger_stances.values() for y in x]))
print('Stance:', Counter(x['stance'] for d in id_2_merger_stances.values() for x in d))
print('Merger:', Counter(x['merger'] for d in id_2_merger_stances.values() for x in d))
json.dump([y for x in id_2_merger_stances.values() for y in x], open('wtwt_1.json', 'w+'), indent=2)

print('Repeat:', len([y for x in id_2_merger_stances.values() for y in x])-
                len(list(set(x['tweet_id'] + "||" + x['merger'] + "||" + x['stance']  
                for x in [x for d in id_2_merger_stances.values() for x in d]))))


# 2. Lexicon bias free
print("============= Step 2 =============")
num_added = 0
for tid, data in id_2_merger_stances.items():
    if len(data) == 1:
        if data[0]['stance'] == "unrelated":
            continue
        num_added += 1
        unrelated_merger = random.choice([x for x in ["CVS_AET", "AET_HUM", "ANTM_CI", "CI_ESRX", "DIS_FOX"]
                                          if x != data[0]['merger']])
        id_2_merger_stances[tid].append({'tweet_id': tid,
                                        'merger': unrelated_merger,
                                        'stance': 'unrelated'
                                    })


print("Num added:", num_added)
print('Stance:', Counter(x['stance'] for d in id_2_merger_stances.values() for x in d))
print('Merger:', Counter(x['merger'] for d in id_2_merger_stances.values() for x in d))
json.dump([y for x in id_2_merger_stances.values() for y in x], open('wtwt_2.json', 'w+'), indent=2)
print(len([y for x in id_2_merger_stances.values() for y in x]))
print('Repeat:', len([y for x in id_2_merger_stances.values() for y in x])-
                len(list(set(x['tweet_id'] + "||" + x['merger'] + "||" + x['stance']  
                for x in [x for d in id_2_merger_stances.values() for x in d]))))
print('Num_label_per_tweet:', Counter(len(x) for x in id_2_merger_stances.values() if len(x) != 1 or x[0]['stance'] != "unrelated"))

# 3. Balance Class dist
print("============= Step 3 =============")

num_unrel_added, num_comment_addded = 0, 0
for tid, data in id_2_merger_stances.items():
    if len(data) == 1: # add unrelated
        assert data[0]['stance'] == "unrelated"
        if random.random() > 0.5 and random.random() > 0.5:
            unrelated_merger = random.choice([x for x in ["CVS_AET", "AET_HUM", "ANTM_CI",
                                                        "CI_ESRX", "DIS_FOX"]
                                                        if x != data[0]['merger']
                                            ])
            id_2_merger_stances[tid].append({'tweet_id': tid,
                                    'merger': unrelated_merger,
                                    'stance': data[0]['stance'],
                                })
        else:
            num_unrel_added += 1
            id_2_merger_stances[tid].append({'tweet_id': tid,
                                            'merger': "NEG_" + data[0]['merger'],
                                            'stance': data[0]['stance'],
                                    })
    elif len(data) == 2:
        to_append = []
        for d in data: # comment
            if len(to_append) > 0:
                break
            if d['stance'] == 'comment':
                if random.random() > 0.5:
                    continue
                num_comment_addded += 1
                to_append.append({'tweet_id': tid,
                                    'merger': "NEG_" + d['merger'],
                                    'stance': d['stance']
                            })
        id_2_merger_stances[tid].extend(to_append)


print("Num added:", num_unrel_added, ",", num_comment_addded)
print('Stance:', Counter(x['stance'] for d in id_2_merger_stances.values() for x in d))

sorted(Counter(x['stance']+"||"+x['merger'] for d in id_2_merger_stances.values() for x in d).items(), key = lambda x: x[0])

list_format = [x for d in id_2_merger_stances.values() for x in d]
print(len(list_format))
print('Stance:', Counter(x['stance'] for x in list_format))
print('Merger:', Counter(x['merger'] for x in list_format))
print('Tweet_id:', Counter(Counter(x['tweet_id'] for x in list_format).values()))

print("Removing repeats")

all_dataset_no_repeat = list(set(x['tweet_id'] + "||" + x['merger'] + "||" + x['stance']  
                for x in list_format))

new_dataset = []
for new_datapoint in all_dataset_no_repeat:
    tid, mgr, stnc = new_datapoint.split("||")
    new_dataset.append({"tweet_id": tid,
                        "merger": mgr,
                        "stance": stnc
                    })

print(len(new_dataset))
print('Stance:', Counter(x['stance'] for x in new_dataset))
print('Merger:', Counter(x['merger'] for x in new_dataset))
print('Tweet_id:', Counter(Counter(x['tweet_id'] for x in new_dataset).values()))

json.dump(new_dataset, open("wtwt_new.json", "w+"), indent=4)

print(new_dataset[0])

max(Counter(x['tweet_id']+'|-|'+x['merger'] for x in new_dataset).values())

